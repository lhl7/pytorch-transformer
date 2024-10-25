import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causual_mask
from model import build_transformer

from config import get_config, get_weights_file_path, latest_weights_file_path
 
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm

import warnings

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 推理过程中，编码器只需要运行一次，保留结果，多次运行解码器即可
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    #encoder只需要调用一次
    encoder_output = model.encode(source, source_mask)
    #初始化decoder输入，一开始是一个单个的标记
 
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        decoder_mask = causual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        #取出最后一个token（当前预测的新词）
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        #和之前生成的拼在一起
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],dim=1)
        
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)
        
        
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    
    #模型输入
    source_texts = []
    #标签
    expected = []
    #模型输出
    predicted = []
    
    console_width = 80
    
    # 表示不训练梯度，只推理
    with torch.no_grad():
        for batch in validation_ds:
            count+=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) == 1 , "batch size must be 1 for validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            #tqdm中的打印方式
            print_msg('-'*console_width)
            print_msg(f"SOURCE:{source_text}")
            print_msg(f"TARGET:{target_text}")
            print_msg(f"PREDICTED:{model_out_text}")
            
            if count == num_examples:
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
        


def get_all_sentence(ds, lang):
    for item in ds:
        yield item['translation'][lang]

#  ds数据集 lang表示语言,ds的格式为：[{ "en": "There was no possibility of taking a walk that day.", "it": "I. In quel giorno era impossibile passeggiare." } , .......]
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequncy = 2)
        tokenizer.train_from_iterator(get_all_sentence(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer
        
def get_ds(config):
    ds_raw  = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")
    
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])
    
    # 分割数据集，其中0.9用于训练
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    #print(f"tokenizersrc,{tokenizer_src} ")
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config['lang_src']]).ids
        
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"max len of src sentence: {max_len_src}")
    print(f"max len of tgt sentence: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"])
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # TensorBoard
    
    writer = SummaryWriter(config["experiment_name"])
    
    #定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    #加载之前的
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    # 标签平滑用于随机丢弃概率最高的预测的一定概率
    
    #训练主循环
    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        #tpdm 用来显示进度条
        batch_iterator = tqdm(train_dataloader, desc=f"processiung epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) #(B,Seq_len)
            decoder_input = batch["decoder_input"].to(device) #(B,Seq_len)
            encoder_mask = batch["encoder_mask"].to(device) #(B,1,1,Seq_len)
            decoder_mask = batch["decoder_mask"].to(device) #(B,1,Seq_len,Seq_len)
            
            encoder_output = model.encode(encoder_input, encoder_mask) #(B,Seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(B, Seq_len, d_model)
            proj_ouput = model.project(decoder_output) #(B, Seq_len, tgt_vocab_size)
            
            label = batch["label"].to(device) #(B,seq_Len)
            
            #
            loss = loss_fn(proj_ouput.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step+=1
        
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch':epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step':global_step
        }, model_filename)
        
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
            
            