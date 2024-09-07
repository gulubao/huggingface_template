import sys
import os
from matplotlib import font_manager as fm, pyplot as plt
font_path = '/mnt/c/Windows/Fonts/calibri.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='Calibri')

sys.path.append('.') # 命令行当前所在的路径

from config.defaults import default_parser
from engine.trainer import do_train
from modeling import build_model
from data import build_dataset, build_data_collator

def main(args, training_args):
    train_dataset, eval_dataset, dataset_info = build_dataset(args)
    data_collator = build_data_collator()
    args.dataset_info = dataset_info
    args.logger.info(f"\ndataset_info: {dataset_info}")

    model = build_model(args, training_args)
    do_train(
        training_args=training_args, 
        model=model, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        data_collator=data_collator, 
    )

if __name__ == '__main__':
    args, training_args = default_parser()
    if args.mydebug:
        import debugpy
        try:
            # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
            debugpy.listen(("localhost", 9501))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
        except Exception as e:
            pass

    main(args, training_args)
