import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from io import StringIO
import textwrap

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed
)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters or not."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input testing data file (a jsonlines file)."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

@dataclass
class ExperimentArguments:
    experiment_name: str = field(default="default", metadata={"help": "The name of the experiment."})
    logger: logging.Logger = field(default=None, metadata={"help": "The logger of the experiment."})
    def __post_init__(self):
        pass

def default_parser():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ExperimentArguments, ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        exp_args, model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        exp_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # -----------------------------------------------------------------------------
    # Update parser
    # -----------------------------------------------------------------------------
    training_args.output_dir = Path("logs").resolve() / exp_args.experiment_name
    training_args.output_dir.mkdir(parents=True, exist_ok=True)
    exp_args.logger = setup_logger(exp_args, training_args, 0)
    set_seed(training_args.seed)
    
    
    log_args_in_chunks(exp_args, N=4, logger=exp_args.logger)
    log_args_in_chunks(model_args, N=4, logger=exp_args.logger)
    log_args_in_chunks(data_args, N=4, logger=exp_args.logger)
    log_args_in_chunks(training_args, N=4, logger=exp_args.logger)  
    
    return exp_args, model_args, data_args, training_args

def setup_logger(exp_args, training_args, distributed_rank):
    logger = logging.getLogger(exp_args.experiment_name)

    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(training_args.output_dir, "log.log"), mode='a+')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    return logger

def log_args_in_chunks(args, N=4, logger=None):
    """
    将参数以每行N个参数的方式组织成一个字符串，然后一次性记录日志。
    
    参数:
    args (argparse.Namespace): 要记录的参数。
    N (int): 每行参数的数量。
    logger (logging.Logger): 用于记录日志的Logger对象。
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 如果args不是字典，将其转换为字典
    if not isinstance(args, dict):
        args = vars(args)
    
    # 将args字典转换为字符串列表
    args_list = ["{}={}".format(k, v) for k, v in args.items()]
    
    # 将列表分割成大小为N的块
    chunks = [args_list[i:i + N] for i in range(0, len(args_list), N)]
    
    # 使用StringIO来构建最终的日志消息
    log_message = StringIO()
    log_message.write("Running with config:\n")
    
    for chunk in chunks:
        chunk_str = ", ".join(chunk)
        wrapped_lines = textwrap.wrap(chunk_str, width=120)
        log_message.write("\n\t".join(wrapped_lines) + "\n")
    
    # 一次性记录整个日志消息
    logger.info("{}".format(log_message.getvalue().strip()))
