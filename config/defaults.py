from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from config_util import setup_logger, log_args_in_chunks
from transformers import HfArgumentParser

@dataclass
class DefaultArguments:
    # 实验参数
    seed: int = field(default=42, metadata={"help": "随机数种子"})
    experiment_name: str = field(default="experiment_debug", metadata={"help": "实验名称"})
    output_dir: str = field(default="INFER", metadata={"help": "输出目录,基于experiment_name推断"}) 
    debug: bool = field(default=False, metadata={"help": "是否为调试模式"})
    
    # 训练参数
    train_batch_size: int = field(default=32, metadata={"help": "训练批次大小"})
    eval_batch_size: int = field(default=64, metadata={"help": "评估批次大小"})
    num_epochs: int = field(default=3, metadata={"help": "训练轮数"})
    learning_rate: float = field(default=5e-5, metadata={"help": "学习率"})
    warmup_steps: int = field(default=500, metadata={"help": "预热步数"})
    weight_decay: float = field(default=0.01, metadata={"help": "权重衰减"})
    
    resume_from_checkpoint: Optional[str] = field(
        default=None, 
        metadata={"help": "恢复训练的检查点路径"}
    )
    
    # Accelerate相关参数
    mixed_precision: str = field(
        default="no",
        metadata={
            "help": "是否使用混合精度训练",
            "choices": ["no", "fp16", "bf16"]
        }
    )
    cpu: bool = field(default=False, metadata={"help": "是否强制使用CPU"})
    num_processes: int = field(default=1, metadata={"help": "分布式训练进程数"})
    
    # 内部字段
    logger: Optional[object] = field(default=None, metadata={"help": "日志器实例"})
    
    def __post_init__(self):
        """初始化后的处理"""
        # 处理输出目录
        if self.output_dir == "INFER":
            self.output_dir = Path("logs").resolve() / self.experiment_name
        else:
            self.output_dir = Path(self.output_dir).resolve()
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志器
        if self.logger is None:
            self.logger = setup_logger("H-U Match ML:", self.output_dir)
            
        # 验证参数
        self._validate_args()
        
    def _validate_args(self):
        """验证参数的有效性"""
        # 添加验证逻辑
        pass

def default_parser():
    """解析命令行参数"""
    parser = HfArgumentParser(DefaultArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args

