import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os 
import sys
import yaml
import time
import pickle
import argparse
from functools import partial

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from tokenizer.tokenizer import BPETokenizer
from dataset.dataset import NalanDataset
from model.gpt import GPT

def batch_proc(batch, tokenizer_cfg):
    BOS, EOS, PAD = tokenizer_cfg['BOS'], tokenizer_cfg['EOS'], tokenizer_cfg['PAD']
    bos_ids, _ = tokenizer.encode(BOS)
    eos_ids, _ = tokenizer.encode(EOS)
    pad_ids, _ = tokenizer.encode(PAD)
    
    batch_x = []
    batch_chatml = []
    # bpe encode
    for sample in batch:
        ids,chatml = sample
        ids = bos_ids + ids + eos_ids
        batch_x.append(ids)
        batch_chatml.append(chatml)
    
    # padding
    max_len = max([len(ids) for ids in batch_x])
    for ids in batch_x:
        if len(ids) < max_len:
            ids.extend(pad_ids * (max_len - len(ids)))
    batch_x = torch.tensor(batch_x, dtype=torch.long)
    
    # padding mask
    batch_padding_mask = (batch_x == pad_ids[0])
    return batch_x, batch_padding_mask, batch_chatml

def train(model, data_loader, optimizer, epoch, DEVICE):
    progress_bar = tqdm(data_loader, leave=False)
    for batch in tqdm(data_loader):
        batch_ids,  batch_padding_mask, batch_chatml = batch

        batch_ids = batch_ids.to(DEVICE)
        batch_padding_mask = batch_padding_mask.to(DEVICE)
        
        logtis = model(batch_ids, batch_padding_mask)  # (batch,seq,vocab)
        
        probs = logtis[:, :-1, :]   # (batch,seq-1,vocab)
        targets = batch_ids[:, 1:] # (batch,seq-1)
        loss = F.cross_entropy(probs.reshape(-1,probs.size(2)), targets.reshape(-1), ignore_index=pad_ids[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix(loss=loss.detach().item())  # 显示当前损失

    progress_bar.close()  # 关闭进度条

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--yaml_path', type=str, help='Path of yaml file.')
    args = parser.parse_args()

    # 读取 YAML 配置文件
    with open(args.yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    data_cfg = config['data']
    tokenizer_cfg = config['tokenizer']
    model_cfg = config['model']
    train_cfg = config['train']

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = os.path.join(data_cfg['data_dir'], data_cfg['data_name'], '{}.bin'.format(config['GPT_MODE']))
    ds = NalanDataset(config)
    dataset = ds.load_dataset(data_path)

    tokenizer_path = os.path.join(tokenizer_cfg['tokenizer_dir'], data_cfg['data_name'], 'tokenizer.bin')
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    

    pad_ids, _ = tokenizer.encode(tokenizer_cfg['PAD'])
    model = GPT(d_model=model_cfg['GPT_DIM'], nhead=model_cfg['GPT_HEAD'], feedforward=model_cfg['GPT_FF'], blocks_num=model_cfg['GPT_BLOCKS'], 
            vocab_size=tokenizer.vocab_size(), seq_max_len=tokenizer_cfg['MAX_SEQ_LEN']).to(DEVICE) # 模型
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    saved_path = os.path.join(train_cfg['model_dir'], data_cfg['data_name'], 'checkpoint.path')
    start_epoch = 0
    try:
        checkpoint = torch.load(saved_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    except Exception as e:
        print(e)
    
    # dataloader = DataLoader(dataset, batch_size=train_cfg['BATCH_SIZE'], shuffle=True, num_workers=8, persistent_workers=True)
    data_loader = DataLoader(dataset, batch_size=train_cfg['BATCH_SIZE'], shuffle=True, num_workers=2, persistent_workers=True, collate_fn=partial(batch_proc, tokenizer_cfg=tokenizer_cfg))

    os.makedirs(os.path.join(train_cfg['model_dir'], data_cfg['data_name']), exist_ok=True)
    for epoch in range(start_epoch + 1, train_cfg['EPOCHS']):
        print('==> training...')
        time1 = time.time()
        train(model, data_loader, optimizer, epoch, DEVICE)
        time2 = time.time()
        print('Epoch {}, train time {:.2f}'.format(epoch, time2 - time1))
        checkpoint = {'epoch':epoch, 'model':model.state_dict(), 'optimizer':optimizer.state_dict()}
        saved_path = os.path.join(train_cfg['model_dir'], data_cfg['data_name'], 'checkpoint.path')
        torch.save(checkpoint, saved_path)
