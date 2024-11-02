from torch.utils.data import Dataset
import os
import sys
import json 
import pickle
from tqdm import tqdm

from tokenizer.tokenizer import BPETokenizer

# 数据集
class NalanDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.data_cfg = config['data']
        self.tokenizer_cfg = config['tokenizer']
        self.GPT_MODE = config['GPT_MODE']
    
    def build_train_data(self, file_path):
        tokenizer = BPETokenizer()
        tokenizer_path = os.path.join(self.tokenizer_cfg['tokenizer_dir'], self.data_cfg['data_name'], 'tokenizer.bin')
        tokenizer.load(tokenizer_path)
        
        with open(self.data_cfg['raw_data'], 'r', encoding='utf-8') as fp:
            raw_ds = json.loads(fp.read())
        self.data = []
        for sample in tqdm(raw_ds, desc='building dataset'):
            try:
                text = '\n'.join(sample['para'])
                if self.GPT_MODE == 'chat':
                    inputs = f"{self.tokenizer_cfg['IM_START']}user\n{sample['title']}\n{self.tokenizer_cfg['IM_END']}\n{self.tokenizer_cfg['IM_START']}assistant\n{text}\n{self.tokenizer_cfg['IM_END']}"
                else:
                    inputs = f"{text}"
                ids, _ = tokenizer.encode(inputs)
                if len(ids) > self.tokenizer_cfg['MAX_SEQ_LEN'] - 2:  # 留出BOS和EOS的token
                    continue
                self.data.append((ids, inputs))
            except Exception as e:
                print(e)
                continue
        with open(file_path, 'wb') as fp:
            pickle.dump(self.data, fp)
    
    def load_dataset(self, path):
        with open(path, 'rb') as fp:
            dataset = pickle.load(fp)
        return dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
