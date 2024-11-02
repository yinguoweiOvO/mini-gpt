import pickle
import os 
import sys 
import yaml
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dataset import NalanDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--yaml_path', type=str, help='Path of yaml file.')
    args = parser.parse_args()

    # 读取 YAML 配置文件
    with open(args.yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    data_cfg = config['data']
    tokenizer_cfg = config['tokenizer']

    os.makedirs(os.path.join(data_cfg['data_dir'], data_cfg['data_name']), exist_ok=True)
    file_path = os.path.join(data_cfg['data_dir'], data_cfg['data_name'], '{}.bin'.format(config['GPT_MODE']))

    dataset = NalanDataset(config)
    dataset.build_train_data(file_path)
    print(f'{file_path}已生成')

    dataset = dataset.load_dataset(file_path)
    print(f'{file_path}，训练集大小：{len(dataset)}，样例数据如下：')
    ids, text = dataset[5]
    print(ids)
    print(text)