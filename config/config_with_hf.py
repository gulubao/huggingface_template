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
from typing import Optional, Dict, List, Union, Any

import transformers
from transformers.training_args import default_logdir
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
    experiment_name: str = field(default="{output_dir}.name", metadata={"help": "The name of the experiment."})
    logger: logging.Logger = field(default=None, metadata={"help": "The logger of the experiment."})
    mydebug: bool = field(default=False, metadata={"help": "Whether to use my debug mode."})
    # -----------------------------------------------------------------------------
    # Raw file paths
    # -----------------------------------------------------------------------------
    input_file: str = field(default="../dataset/xxx.csv", metadata={"help": "Raw csv household-unit data file"})

    # -----------------------------------------------------------------------------
    # processed file paths
    # -----------------------------------------------------------------------------
    train_path: str = field(default="{output_dir}/propossed/train.csv", metadata={"help": "Processed csv household data file"})
    eval_path: str = field(default="{output_dir}/propossed/eval.csv", metadata={"help": "Processed csv household data file"})

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
        default="logs/experiment_debug",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    remove_unused_columns: Optional[bool] = field(
        default=False, 
        metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    ) # 在debug时不删除无用列
    # -----------------------------------------------------------------------------
    # DataLoader Parameters
    # -----------------------------------------------------------------------------
    per_device_train_batch_size: int = field(
        default=1024, 
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1024, 
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    dataloader_num_workers: int = field(
        default=8, 
        metadata={"help": "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process."}
    )
    dataloader_drop_last: bool = field(
        default=True, 
        metadata={"help": "Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size) or not."}
    )
    # -----------------------------------------------------------------------------
    # optimizer
    # -----------------------------------------------------------------------------
    learning_rate: float = field(
        default=5e-5,  # 5e-5
        metadata={"help": "The initial learning rate for [`AdamW`] optimizer."}
    )
    weight_decay: float = field(
        default=1e-5, 
        metadata={"help": "The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`] optimizer."}
    )
    num_train_epochs: float = field(
        default=500, 
        metadata={"help": "Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training)."}
    )
    lr_scheduler_type: str = field(
        default="cosine_with_min_lr", 
        metadata={"help": "The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values."}
    )
    lr_scheduler_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"min_lr": 1e-6},
        metadata={"help": "Arguments for the scheduler."}
    )

    # -----------------------------------------------------------------------------
    # training strategy
    # -----------------------------------------------------------------------------
    label_smoothing_factor: float = field(
        default=0.1, 
        metadata={"help": "The label smoothing factor to use. Zero means no label smoothing, otherwise the underlying onehot-encoded labels are changed from 0s and 1s to `label_smoothing_factor/num_labels` and `1 - label_smoothing_factor + label_smoothing_factor/num_labels` respectively."}
    )

    # -----------------------------------------------------------------------------
    # logging & save
    # -----------------------------------------------------------------------------
    log_level: str = field(
        default="info", 
        metadata={"help": "Logger log level to use on the main process. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and keeps the current log level for the Transformers library (which will be `'warning'` by default)."}
    )
    logging_first_step: bool = field(
        default=True, 
        metadata={"help": "Whether to log the first `global_step` or not."}
    )
    logging_steps: int = field(
        default=0.1, 
        metadata={"help": "Number of update steps between two logs if `logging_strategy='steps'`."}
    )
    eval_on_start: bool = field(
        default=False, 
        metadata={"help": "Whether to perform a evaluation step (sanity check) before the training to ensure the validation steps works correctly."}
    )
    eval_strategy: str = field(
        default="steps", 
        metadata={"help": "Evaluation strategy. Literal['epoch', 'steps']"}
    )
    eval_steps: int = field(
        default=0.1, 
        metadata={"help": "Number of update steps between two evaluations if `eval_strategy='steps'`."}
    )

    save_strategy: str = field(
        default="steps", 
        metadata={"help": "The checkpoint save strategy to adopt during training. Literal['no', 'epoch', 'steps']"}
    )
    save_steps: int = field(
        default=0.1, 
        metadata={"help": "Number of update steps before two checkpoint saves if `save_strategy='steps'`."}
    )
    save_total_limit: int = field(
        default=3, 
        metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to `metric_for_best_model` will always be retained in addition to the most recent ones. For example, for `save_total_limit=5` and `load_best_model_at_end`, the four last checkpoints will always be retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end`, it is possible that two checkpoints are saved: the last one and the best one (if they are different)."}
    )
    load_best_model_at_end: bool = field(
        default=True, 
        metadata={"help": "Whether to load the best model at the end of training."}
    )
    report_to: str = field(
        default="tensorboard", 
        metadata={"help": "The list of integrations to report the results and logs to. Supported platforms are `'azure_ml'`, `'clearml'`, `'codecarbon'`, `'comet_ml'`, `'dagshub'`, `'dvclive'`, `'flyte'`, `'mlflow'`, `'neptune'`, `'tensorboard'`, and `'wandb'`. Use `'all'` to report to all integrations installed, `'none'` for no integrations."}
    )
    # -----------------------------------------------------------------------------
    # accelerate
    # -----------------------------------------------------------------------------
    jit_mode_eval: bool = field(
        default=True, 
        metadata={"help": "Whether to use PyTorch jit trace for inference."}
    )
    fp16: bool = field( 
        default=False, 
        metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."}
    )
    fp16_full_eval: bool = field(
        default=False, 
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit. This will be faster and save memory but can harm metric values."}
    )
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
    # 更新地址相关参数
    # -----------------------------------------------------------------------------
    training_args.output_dir = Path(training_args.output_dir).resolve()
    training_args.output_dir.mkdir(parents=True, exist_ok=True)
    args.experiment_name = training_args.output_dir.name
    args.logger = setup_logger_tf(args, training_args)
    set_seed(training_args.seed)
    args.train_path = training_args.output_dir / "propossed" / "train.csv"
    args.eval_path = training_args.output_dir / "propossed" / "eval.csv"
    args.train_path.parent.mkdir(parents=True, exist_ok=True)
    training_args.output_dir = str(training_args.output_dir)
    # 更新xxx参数


    # -----------------------------------------------------------------------------
    # 条件限制
    # -----------------------------------------------------------------------------
    # assert pass

    # -----------------------------------------------------------------------------
    # 打印参数
    # -----------------------------------------------------------------------------
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