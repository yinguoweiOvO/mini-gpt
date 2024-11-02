import torch 
import torch.nn.functional as F
import random
import os
import sys
import yaml
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model.gpt import GPT
from tokenizer.tokenizer import BPETokenizer

def chat(query, tokenizer, model, DEVICE, config, tokenizer_cfg, inference_cfg):
    if config['GPT_MODE'] == 'chat':
        inputs = f"{tokenizer_cfg['BOS']}{tokenizer_cfg['IM_START']}user\n{query}\n{tokenizer_cfg['IM_END']}\n{tokenizer_cfg['IM_START']}assistant\n"
    else:
        inputs = f"{tokenizer_cfg['BOS']}{query}"
    ids, _ = tokenizer.encode(inputs)
    
    while len(ids) < tokenizer_cfg['MAX_SEQ_LEN']:
        batch_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        batch_paddding_mask = torch.tensor([[0]*len(ids)],dtype=torch.bool).to(DEVICE)
        
        with torch.no_grad():
            logits = model(batch_ids, batch_paddding_mask) # (batch,seq,vocab)
            # 多样性控制
            logits = logits[0,-1,:] / inference_cfg['TEMPERATURE']
            topk_logits,topk_ids = torch.topk(logits, k=inference_cfg['TOP_K'])
            topk_logits,topk_ids = topk_logits.cpu(), topk_ids.cpu()
            # 从topk中随机挑1个token
            topk_probs = F.softmax(topk_logits, dim=-1)
            rnd = random.random()
            cumsum = 0
            for i in range(inference_cfg['TOP_K']):
                if rnd < cumsum + topk_probs[i]:
                    next_id = topk_ids[i].item()
                    break
                cumsum += topk_probs[i]

        if next_id in eos_ids + pad_ids + im_end_ids:
            break
        ids = ids + [next_id]
    return tokenizer.decode(ids[1:])
        
    
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
    inference_cfg = config['inference']

    # 设备
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu' 

    # 分词器
    tokenizer_path = os.path.join(tokenizer_cfg['tokenizer_dir'], data_cfg['data_name'], 'tokenizer.bin')
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    # 加载模型
    model = GPT(d_model=model_cfg['GPT_DIM'], nhead=model_cfg['GPT_HEAD'], feedforward=model_cfg['GPT_FF'], blocks_num=model_cfg['GPT_BLOCKS'], 
            vocab_size=tokenizer.vocab_size(), seq_max_len=tokenizer_cfg['MAX_SEQ_LEN']).to(DEVICE) # 模型
    saved_path = os.path.join(train_cfg['model_dir'], data_cfg['data_name'], 'checkpoint.path')
    try:  
        checkpoint = torch.load(saved_path)
        model.load_state_dict(checkpoint['model'])
    except:
        pass

    model.eval()

    # 可能的结束符
    EOS, PAD, IM_END = tokenizer_cfg['EOS'], tokenizer_cfg['PAD'], tokenizer_cfg['IM_END']
    eos_ids,_=tokenizer.encode(EOS)
    pad_ids,_=tokenizer.encode(PAD)
    im_end_ids,_=tokenizer.encode(IM_END)
    while True:
        query=input('>')
        if query=='exit':
            break
        
        resp = chat(query, tokenizer, model, DEVICE, config, tokenizer_cfg, inference_cfg)
        print('<',resp)
