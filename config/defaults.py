import argparse

def default_parser():
    parser = argparse.ArgumentParser(description="默认配置")

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

    # 输出和日志参数
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--logging_dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志记录步数")

    # 与 Accelerate 相关的参数
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="是否使用混合精度训练")
    parser.add_argument("--cpu", action="store_true", help="是否强制使用CPU")
    parser.add_argument("--num_processes", type=int, default=1, help="用于分布式训练的进程数")
    return parser.parse_args()