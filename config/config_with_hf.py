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
    input_file: str = field(default="../dataset/xxx.csv", metadata={"help": "Raw csv household-unit data file"})

    # -----------------------------------------------------------------------------
    # processed file paths
    # -----------------------------------------------------------------------------
    train_path: str = field(default="logs/{experiment_name}/propossed/train.csv", metadata={"help": "Processed csv household data file"})
    eval_path: str = field(default="logs/{experiment_name}/propossed/eval.csv", metadata={"help": "Processed csv household data file"})

    # -----------------------------------------------------------------------------
    # processing parameters
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------------
    retrive_batch_size: int = field(default=256, metadata={"help": "Batch size for retriving the topk accuracy"})

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="logs",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    remove_unused_columns: Optional[bool] = field(
        default=False, 
        metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    ) # 在debug时不删除无用列

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
    args.train_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_args_in_chunks(args, N=4, logger=args.logger)
    log_args_in_chunks(training_args, N=4, logger=args.logger)

    return args, training_args

if __name__ == "__main__":
    """
    测试
    conda activate tf
    cd ~/code/research/xxx
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