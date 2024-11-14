import sys
import os
from matplotlib import font_manager as fm, pyplot as plt
from modeling.configuration_custom import CustomModelConfig
from modeling.modeling_custom import CustomModel
from engine.trainer import CustomTrainer
from data import build_dataset, build_data_collator
from config.defaults import default_parser

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager as fm, pyplot as plt
font_path = '/mnt/c/Windows/Fonts/calibri.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='Calibri')
# Update the font path to Comic Sans MS
# font_path = '/mnt/c/Windows/Fonts/comic.ttf'  # Ensure this path is correct for your system
# if os.path.exists(font_path):
#     fm.fontManager.addfont(font_path)
#     plt.rc('font', family='Comic Sans MS')

sys.path.append('.')  # 命令行当前所在的路径

def main(args, training_args):
    train_dataset, eval_dataset, dataset_info = build_dataset(args)
    data_collator = build_data_collator()
    args.dataset_info = dataset_info
    args.logger.info(f"\ndataset_info: {dataset_info}")

    # Model initialization logic, similar to tune.py
    config = CustomModelConfig(
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        hidden_act=args.hidden_act,
        layer_norm_eps=args.layer_norm_eps,
        hidden_dropout_prob=args.hidden_dropout_prob,
        image_loss_weight=args.image_loss_weight,
        num_labels_info=args.num_labels_info,
        logit_scale=args.logit_scale,
        logit_bias=args.logit_bias,
        queue_size=args.queue_size,
        momentum=args.momentum,
        alpha=args.alpha,
        label_smoothing_factor=training_args.label_smoothing_factor,
    )
    model = CustomModel(config=config)

    # Training logic, similar to tune.py
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        logger=args.logger
    )

    trainer.train()

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
