# encoding: utf-8
import argparse
import numpy as np
from pathlib import Path
from io import StringIO
import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed
)

def setup_logger(args, training_args):
    logger = logging.getLogger(args.experiment_name)
    # log_level = training_args.get_process_log_level()
    log_level = logging.INFO

    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    ch.setFormatter(formatter)
    ch.setLevel(log_level)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(training_args.output_dir, "log.log"), mode='a+')
    fh.setFormatter(formatter)
    fh.setLevel(log_level)
    logger.addHandler(fh)

    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    return logger

def log_args_in_chunks(args, N=4, logger=None):
    """
    将参数以每行N个参数的方式组织成一个字符串，然后一次性记录日志。
    每个参数包括其名称和值。
    
    参数:
    args (argparse.Namespace): 要记录的参数。
    N (int): 每行参数的数量。
    logger (logging.Logger): 用于记录日志的Logger对象。
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    args_name = args.__class__.__name__
    if not isinstance(args, dict):
        args = vars(args)
    
    # 将args字典转换为字符串列表，并找出最长的参数字符串
    args_list = []
    max_param_length = 0
    for k, v in args.items():
        param_str = f"{k} = {v}"
        args_list.append(param_str)
        max_param_length = max(max_param_length, len(param_str))
    
    # 使用StringIO来构建最终的日志消息
    log_message = StringIO()
    log_message.write(f"运行配置 - {args_name}:\n")
    
    # 将参数列表分割成大小为N的块
    for i in range(0, len(args_list), N):
        chunk = args_list[i:i+N]
        # 对齐每个参数
        formatted_chunk = [f"{param:<{max_param_length}}" for param in chunk]
        log_message.write("    " + "  ".join(formatted_chunk) + "\n")
    
    # 一次性记录整个日志消息
    logger.info(log_message.getvalue().strip())

@dataclass
class CustomArguments:
    # -----------------------------------------------------------------------------
    # Experiment Parameters
    # -----------------------------------------------------------------------------
    experiment_name: str = field(default="experiment_debug", metadata={"help": "The name of the experiment."})
    logger: logging.Logger = field(default=None, metadata={"help": "The logger of the experiment."})
    mydebug: bool = field(default=False, metadata={"help": "Whether to use my debug mode."})
    # -----------------------------------------------------------------------------
    # Raw file paths
    # -----------------------------------------------------------------------------
    input_house_raw_file: str = field(default="../dataset/row_data/psam_h10.csv", metadata={"help": "Raw csv household-unit data file"})
    input_person_raw_file: str = field(default="../dataset/row_data/psam_p10.csv", metadata={"help": "Raw csv person data file"})
    interest_column_file: str = field(default="../dataset/row_data/interest_variables_2.xlsx", metadata={"help": "File containing the interested columns"})

    # -----------------------------------------------------------------------------
    # processed file paths
    # -----------------------------------------------------------------------------
    train_path: str = field(default="logs/{experiment_name}/propossed/train.csv", metadata={"help": "Processed csv household data file"})
    eval_path: str = field(default="logs/{experiment_name}/propossed/eval.csv", metadata={"help": "Processed csv household data file"})
    gt_map_path: str = field(default="logs/{experiment_name}/propossed/gt_map.csv", metadata={"help": "Processed csv ground truth mapping file"})

    # -----------------------------------------------------------------------------
    # processing parameters
    # -----------------------------------------------------------------------------
    numeric_features: List[str] = field(default_factory=lambda: ["SetBelow"], metadata={"help": "Numeric features. SetBelow"})
    ordinal_features: Dict[str, List[int]] = field(default_factory=lambda: {"SetBelow": [0, 1, 2, 3]}, metadata={"help": "Ordinal features. SetBelow"})
    categorical_features: List[str] = field(default_factory=lambda: ["SetBelow"], metadata={"help": "Categorical features. SetBelow"})
    # -----------------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------------
    retrive_batch_size: int = field(default=256, metadata={"help": "Batch size for retriving the topk accuracy"})

def default_parser():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, training_args = parser.parse_args_into_dataclasses()

    # -----------------------------------------------------------------------------
    # Update parser
    # -----------------------------------------------------------------------------
    training_args.output_dir = Path(training_args.output_dir).resolve() / args.experiment_name
    training_args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logger = setup_logger(args, training_args)
    set_seed(training_args.seed)
    args.train_path = training_args.output_dir / "propossed" / "train.csv"
    args.eval_path = training_args.output_dir / "propossed" / "eval.csv"
    args.gt_map_path = training_args.output_dir / "propossed" / "gt_map.csv"
    args.gt_map_path.parent.mkdir(parents=True, exist_ok=True)
    args.numeric_features = [
        "NP", "GRNTP", "GRPIP", "HHLDRAGEP", "HINCP", "BDSP", "MRGP",
        "RMSP", "RNTP", "VALP", "TAXAMT"
    ]
    args.ordinal_features = {
        "ACR": [0, 1, 2, 3], 
        "VEH": [0, 1, 2, 3, 4, 5, 6], 
        "YRBLT": [1939, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2021], 
        "SCHL": [1, 2, 3, 4, 5, 6, 7, 8]
    }
    args.categorical_features = [
        "TEN_H", "HHL", "HHLDRRAC1P", "HUPAC", "R65", "BLD", "TEN_U", "DIS"
    ]
    
    log_args_in_chunks(args, N=4, logger=args.logger)
    log_args_in_chunks(training_args, N=4, logger=args.logger)

    return args, training_args

if __name__ == "__main__":
    """
    测试
    conda activate tf
    cd ~/code/research/house_unit_match/house_unit_match_clip
    python config/defaults.py --mydebug True --output_dir logs
    """
    # import debugpy
    # try:
    #     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    #     debugpy.listen(("localhost", 9501))
    #     print("Waiting for debugger attach")
    #     debugpy.wait_for_client()
    # except Exception as e:
    #     pass
    args, training_args = default_parser()
    print(args)
    print(training_args)