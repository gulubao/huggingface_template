# encoding: utf-8
import logging
import os
import sys
import textwrap
from io import StringIO
import transformers

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.log"), mode='a+')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def setup_logger_tf(args, training_args):
    logger = logging.getLogger(args.experiment_name)
    # log_level = training_args.get_process_log_level()
    log_level = logging.DEBUG

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
    将参数以每行N个参数的方式组织成一个字符串, 然后一次性记录日志。
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