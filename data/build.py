from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import os

def build_dataset(args, is_train=True):
    # 判断是否使用本地数据集
    if args.use_local_data:
        # 加载本地数据集
        file_path = args.train_file if is_train else args.val_file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件：{file_path}")
        
        df = pd.read_csv(file_path)
        dataset = load_dataset('csv', data_files=file_path)['train']
    else:
        # 加载Hugging Face数据集
        split = 'train' if is_train else 'validation'
        dataset = load_dataset(args.dataset_name, split=split)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 定义预处理函数
    def preprocess_function(examples):
        # 根据数据集的具体结构调整这里的字段名
        texts = (examples['text'] if 'text' in examples 
                 else (examples['sentence1'], examples['sentence2']))
        return tokenizer(texts, truncation=True, padding='max_length', max_length=args.max_length)
    
    # 应用预处理
    dataset = dataset.map(preprocess_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return dataset

def make_datasets(args):
    train_dataset = build_dataset(args, is_train=True)
    eval_dataset = build_dataset(args, is_train=False)
    return train_dataset, eval_dataset