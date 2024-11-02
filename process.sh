
python tokenizer/train_tokenizer.py --yaml_path config/nalanxingde.yaml

python dataset/build_dataset.py --yaml_path config/nalanxingde.yaml

python tools/train_gpt.py --yaml_path config/nalanxingde.yaml

python tools/inference.py --yaml_path config/nalanxingde.yaml