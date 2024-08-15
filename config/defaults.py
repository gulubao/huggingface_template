import argparse
import numpy as np
from pathlib import Path
from io import StringIO
import logging
import os
import sys
import textwrap

def default_parser():
    parser = argparse.ArgumentParser(description="默认配置")
    # -----------------------------------------------------------------------------
    # Experiment Parameters
    # -----------------------------------------------------------------------------
    parser.add_argument("--seed", default=42, help="Seed for random number generator", type=int)
    parser.add_argument("--experiment_name", default="experiment_debug", help="Experiment name", type=str)
    parser.add_argument("--output_dir", default="INFER", help="Output directory. Inferred based on experiment_name. logs/experiment_debug", type=str)
    parser.add_argument("--debug", default=False, help="Debug mode", type=bool)

    # 数据集和模型参数
    parser.add_argument("--dataset_name", type=str, default="glue/mrpc", help="Hugging Face数据集名称")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Hugging Face模型名称")
    parser.add_argument("--num_labels", type=int, default=2, help="分类标签数量")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    # 添加本地数据集相关参数
    parser.add_argument("--use_local_data", action="store_true", help="是否使用本地数据集")
    parser.add_argument("--train_file", type=str, default="data/train.csv", help="本地训练数据文件路径")
    parser.add_argument("--val_file", type=str, default="data/val.csv", help="本地验证数据文件路径")

    # 添加自定义模型相关参数
    parser.add_argument("--use_custom_model", action="store_true", help="是否使用自定义模型")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--custom_model_type", type=str, default="custom1", choices=["custom1", "custom2"], help="自定义模型类型")

    # 添加自定义模型配置参数
    parser.add_argument("--hidden_size", type=int, default=768, help="隐藏层大小")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="隐藏层数量")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="注意力头数量")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="中间层大小")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="隐藏层dropout概率")
    parser.add_argument("--num_labels", type=int, default=2, help="标签数量")

    # 训练参数
    parser.add_argument("--train_batch_size", type=int, default=32, help="训练批次大小")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="评估批次大小")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=500, help="预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="恢复训练的检查点路径")


    # 与 Accelerate 相关的参数
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="是否使用混合精度训练")
    parser.add_argument("--cpu", action="store_true", help="是否强制使用CPU")
    parser.add_argument("--num_processes", type=int, default=1, help="用于分布式训练的进程数")

    # -----------------------------------------------------------------------------
    # Update parser
    # -----------------------------------------------------------------------------
    args = parser.parse_args()
    args.output_dir = Path("logs").resolve() / args.experiment_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logger = setup_logger("H-U Match ML:", args.output_dir, 0)
    
    log_args_in_chunks(args, N=4, logger=args.logger)
    return args

def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.log"), mode='a+')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

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