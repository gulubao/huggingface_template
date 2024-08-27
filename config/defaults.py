import argparse
import numpy as np
from pathlib import Path
from io import StringIO
import logging
import os
import sys
import textwrap
from config_util import setup_logger, log_args_in_chunks

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


    # 添加自定义模型配置参数

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
    args.logger = setup_logger("H-U Match ML:", args.output_dir)
    
    log_args_in_chunks(args, N=4, logger=args.logger)
    return args

