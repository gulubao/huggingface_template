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

from config_util import log_args_in_chunks, setup_logger_tf
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

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="logs", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})

def default_parser():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((CustomArguments, CustomTrainingArguments))
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
    args.logger = setup_logger_tf(args, training_args)
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
    python config/defaults.py
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