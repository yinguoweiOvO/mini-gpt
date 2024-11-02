import os
import sys 
import json 
import yaml
import argparse

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)

from tokenizer import BPETokenizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--yaml_path', type=str, help='Path of yaml file.')
    args = parser.parse_args()

    # 读取 YAML 配置文件
    with open(args.yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    data_cfg = config['data']
    tokenizer_cfg = config['tokenizer']

    with open(data_cfg['raw_data'], 'r', encoding='utf-8') as fp:
        ds = json.loads(fp.read())

    text_list = []
    sample_count = 0
    for sample in ds:
        text_list.append(sample['title'])
        for p in sample['para']: 
            text_list.append(p)
        sample_count += 1
    print('共加载%d条数据'%sample_count)

    # 训练词表
    tokenizer = BPETokenizer()  
    tokenizer.train(text_list, tokenizer_cfg['VOCAB_SIZE'])
    tokenizer.add_special_tokens([tokenizer_cfg['IM_START'], tokenizer_cfg['IM_END'], tokenizer_cfg['BOS'], tokenizer_cfg['EOS'], tokenizer_cfg['PAD']])
    
    os.makedirs(os.path.join(tokenizer_cfg['tokenizer_dir'], data_cfg['data_name']), exist_ok=True)
    tokenizer_path = os.path.join(tokenizer_cfg['tokenizer_dir'], data_cfg['data_name'], 'tokenizer.bin')
    tokenizer.save(tokenizer_path)