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
    args.logger = setup_logger("H-U Match ML:", args.output_dir)
    
    log_args_in_chunks(args, N=4, logger=args.logger)
    return args

