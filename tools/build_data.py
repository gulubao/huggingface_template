import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 5)

import os, sys
from pathlib import Path
from tqdm import tqdm
import json
sys.path.append('.')
from config.defaults import default_parser
from utils.utils_preprocess import label_data

sys.path.append('../')
from house_unit_match_ml.utils_data import load_organize_tables, split_data

def _build_data(args):
    args.logger.info("Loading and organizing data...")
    output = load_organize_tables(args)    
    data = output["data_organized"]

    # 分离特征
    house_features = output["house_columns"] + output["person_columns"]
    unit_features = output["unit_columns"]
    gt_map_columns = output["gt_mapping_columns"]
    gt_map = data[gt_map_columns]

    data = data[house_features + unit_features + gt_map_columns]
    args.logger.info("Labeling data...")    
    data = label_data(data)

    with open(args.gt_map_path.parent / 'features.jsonl', 'w', encoding='utf-8') as f:
        json.dump(
            {
                "house_features": house_features, 
                "house_label": ["label_h"],
                "unit_features": unit_features, 
                "unit_label": ["label_u"],
                "gt_map_columns": gt_map_columns
            }, 
            f, ensure_ascii=False, indent=4
        )

    # 分割训练集和测试集
    args.logger.info("Splitting data...")
    # TODO: 根据house_id分割数据, 而不是原始id, 因为可能存在重复的条目.
    X_train, X_test = split_data(data, test_size=args.train_val_split_ratio, random_state=args.seed, args=args)

    X_train.to_csv(args.train_path, index=False)
    X_test.to_csv(args.eval_path, index=False)
    gt_map.to_csv(args.gt_map_path, index=False)
    return 0 


def main():
    """提前进行数据预处理"""
    args = default_parser()

    if args.debug:
        import debugpy
        try:
            # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
            debugpy.listen(("localhost", 9501))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
        except Exception as e:
            pass

    seed = args.seed
    rank = 0
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    _build_data(args)
    return 0

if __name__ == "__main__":
    main()