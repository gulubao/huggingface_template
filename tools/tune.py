# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import sys
import os
sys.path.append('.')
from config.defaults import default_parser
from engine.trainer import do_train
from modeling import build_model
from data import build_dataset, build_data_collator
from ray import tune
# from ray.tune.search.hebo import HEBOSearch

from modeling.modeling_custom import CustomModel
from modeling.configuration_custom import CustomModelConfig
from engine.trainer import CustomTrainer
from typing import Dict

args, training_args = default_parser()

train_dataset, eval_dataset, args.num_labels_info = build_dataset(args)
data_collator = build_data_collator()

def model_init(trial=None):
    if trial is None:   
        trial = {}

    if "model_scale" in trial:
        if trial["model_scale"] == "small":
            embedding_dim = 32
            hidden_size = 512
            intermediate_size = 2048
            num_hidden_layers = 12
        elif trial["model_scale"] == "medium":
            embedding_dim = 64
            hidden_size = 1024
            intermediate_size = 4096
            num_hidden_layers = 24
        elif trial["model_scale"] == "large":
            embedding_dim = 128
            hidden_size = 2048
            intermediate_size = 8192
            num_hidden_layers = 36

        elif trial["model_scale"] == "small_wide": # 宽度增加, 深度减小
            embedding_dim = 512
            hidden_size = 1024
            intermediate_size = 4096
            num_hidden_layers = 6
        elif trial["model_scale"] == "medium_wide":
            embedding_dim = 1024
            hidden_size = 2048
            intermediate_size = 8192
            num_hidden_layers = 8
        elif trial["model_scale"] == "large_wide":
            embedding_dim = 2048
            hidden_size = 4096
            intermediate_size = 16384
            num_hidden_layers = 12
        else:
            raise ValueError(f"Invalid model scale: {trial['model_scale']}")
    else:
        embedding_dim = trial["embedding_dim"] if "embedding_dim" in trial else args.embedding_dim
        hidden_size = trial["hidden_size"] if "hidden_size" in trial else args.hidden_size
        intermediate_size = trial["intermediate_size"] if "intermediate_size" in trial else args.intermediate_size
        num_hidden_layers = trial["num_hidden_layers"] if "num_hidden_layers" in trial else args.num_hidden_layers

    hidden_act = trial["hidden_act"] if "hidden_act" in trial else args.hidden_act
    hidden_dropout_prob = trial["hidden_dropout_prob"] if "hidden_dropout_prob" in trial else args.hidden_dropout_prob
    logit_scale = trial["logit_scale"] if "logit_scale" in trial else args.logit_scale
    logit_bias = trial["logit_bias"] if "logit_bias" in trial else args.logit_bias
    image_loss_weight = trial["image_loss_weight"] if "image_loss_weight" in trial else args.image_loss_weight
    queue_size = trial["queue_size"] if "queue_size" in trial else args.queue_size
    momentum = trial["momentum"] if "momentum" in trial else args.momentum
    alpha = trial["alpha"] if "alpha" in trial else args.alpha
    label_smoothing_factor = trial["label_smoothing_factor"] if "label_smoothing_factor" in trial else training_args.label_smoothing_factor
    # 使用trial进行超参数搜索
    config = CustomModelConfig(
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        hidden_act=hidden_act,
        layer_norm_eps=args.layer_norm_eps,
        hidden_dropout_prob=hidden_dropout_prob,
        image_loss_weight=image_loss_weight,
        num_labels_info=args.num_labels_info,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        queue_size=queue_size,
        momentum=momentum,
        alpha=alpha,
        label_smoothing_factor=label_smoothing_factor,
    )
    return CustomModel(config=config)

def ray_hp_space(trial): # 与模型大小无关的超参数. 计划进行 60 次实验
    return {
        "learning_rate": tune.loguniform(1e-5, 5e-4), # 2e-4
        "hidden_act": tune.choice(["relu", "gelu", "tanh", "quick_gelu"]), # quick_gelu
        "hidden_dropout_prob": tune.uniform(0.0, 0.1), # 0.02
        "logit_scale": tune.uniform(0.1, 10.0), # 9
        "logit_bias": tune.uniform(-10.0, -1.0), # -7
        "momentum": tune.uniform(0.9, 0.999), # 0.95
        "alpha": tune.uniform(0.1, 0.9), # 0.14
        "label_smoothing_factor": tune.uniform(0.0, 0.2), # 0.05
    }

def ray_hp_space_trail_2(trial): # 平衡权重. 计划进行 10 次实验
    return {
        "image_loss_weight": tune.uniform(0.1, 0.9)
    }

def ray_hp_space_trail_3(trial): # 模型大小调整, 计划测试所有的模型大小, 目前为 6 个.
    # wide 表示模型宽度增加, 深度减小
    return{
        "model_scale": tune.grid_search(["small", "medium", "large", "small_wide", "medium_wide", "large_wide"])
    }

trainer = CustomTrainer(model=None, args=training_args, model_init=model_init, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, logger=args.logger)

def compute_objective(metrics: Dict[str, float]) -> float:
    return metrics["eval_top1_avg"]

# hebo = HEBOSearch(metric="eval_top1_avg", mode="max") # 使用 HEBO 搜索. 建议至少进行 16 次实验.
best_run = trainer.hyperparameter_search(
    hp_space=ray_hp_space_trail_2,
    n_trials=10,
    # search_alg=hebo,
    compute_objective=compute_objective, direction="maximize", backend="ray"
)

ray_results = best_run.run_summary.results_df
ray_results.to_csv(os.path.join(training_args.output_dir, "ray_results.csv"))