from config.defaults import default_parser
from modeling import build_model
from data import make_datasets
from engine import do_train
from accelerate import Accelerator

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
    main(args)