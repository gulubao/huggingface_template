from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
import pandas as pd
import os
import json

def build_dataset(args, load_eval_only=False):
    if not load_eval_only:
        if not args.train_path.exists():
            raise FileNotFoundError(f"找不到文件：{args.train_path}")
        
    if not args.eval_path.exists():
        raise FileNotFoundError(f"找不到文件：{args.eval_path}")

    if not load_eval_only:
        train_dataset = load_dataset('csv', data_files=args.train_path._str, split='train')
    else:
        train_dataset = None

    eval_dataset = load_dataset('csv', data_files=args.eval_path._str, split='train')
    
    def preprocess_function(examples):
        pass
        return examples
    
    if not load_eval_only:
        train_dataset = train_dataset.map(preprocess_function
                                            , batched=True
                                            , load_from_cache_file=False)
        
    eval_dataset = eval_dataset.map(preprocess_function
                                    , batched=True
                                    , load_from_cache_file=False)

    info = json.load(open(args.info_path, 'r', encoding='utf-8'))
    return train_dataset, eval_dataset 

def build_data_collator(args):
    return DefaultDataCollator()