from datasets import load_dataset, Dataset
import os
import json
import numpy as np
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import random
import torch

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

    train_dataset = train_dataset.remove_columns(["label_xxx", "xxx"])
    eval_dataset_df = eval_dataset.to_pandas()
    eval_dataset  = Dataset.from_pandas(eval_dataset_df, preserve_index=False)
    return train_dataset, eval_dataset 

def build_data_collator():
    return torch_custom_data_collator

InputDataClass = NewType("InputDataClass", Any)
def torch_custom_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    """
    自定义数据整理函数,用于处理批次数据。
    
    该函数重写了 from transformers.data.data_collator import torch_default_data_collator

    参数:
    features (List[InputDataClass]): 包含多个样本特征的列表

    返回:
    Dict[str, Any]: 整理后的批次数据字典
    """
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch