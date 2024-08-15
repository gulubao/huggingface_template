from config.defaults import default_parser
from modeling import build_model
from data import make_datasets
from engine import do_train
from accelerate import Accelerator
import torch
import numpy as np
import random

def main(args):
    accelerator = Accelerator()
    accelerator.print(f"当前设备: {accelerator.device}")

    model = build_model(args)
    accelerator.print(f"使用{'自定义' if args.use_custom_model else '预训练'}模型")

    train_dataset, eval_dataset = make_datasets(args)
    
    eval_result = do_train(args, model, train_dataset, eval_dataset, accelerator)
    
    accelerator.print("最终评估结果:", eval_result)

if __name__ == '__main__':
    args = default_parser()
    if args.debug:
        import debugpy
        try:
            # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
            debugpy.listen(("localhost", 9501))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
        except Exception as e:
            pass
    
    seed = args.seed
    rank = 0
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    main(args)