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

from modeling.configuration_blip_2 import Blip2Config
from modeling.modeling_blip_2 import Blip2Model
from engine.trainer import CustomTrainer
from typing import Dict

args, training_args = default_parser()

train_dataset, eval_dataset, args.num_labels_info = build_dataset(args)
data_collator = build_data_collator()

def model_init(trial=None):
    if trial is None:
        hidden_act=args.hidden_act, # quick_gelu        
        hidden_dropout_prob=args.hidden_dropout_prob, # 0.1
        logit_scale=args.logit_scale, # 10
        logit_bias=args.logit_bias, # -10
        momentum=args.momentum, # 0.995
        alpha=args.alpha, # 0.4
        label_smoothing_factor=training_args.label_smoothing_factor, # 0.1
    else:
        hidden_act = trial["hidden_act"]
        hidden_dropout_prob = trial["hidden_dropout_prob"]
        logit_scale = trial["logit_scale"]
        logit_bias = trial["logit_bias"]
        momentum = trial["momentum"]
        alpha = trial["alpha"]
        label_smoothing_factor = trial["label_smoothing_factor"]

    embedding_dim = args.embedding_dim # 32
    hidden_size = args.hidden_size # 512
    intermediate_size = args.intermediate_size # 2048
    num_hidden_layers = args.num_hidden_layers # 12
    queue_size=args.queue_size, # 8192 # bs: 2048
    layer_norm_eps=args.layer_norm_eps, # 1e-5
    num_labels_info=args.num_labels_info,

    config = Blip2Config(
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        hidden_act=hidden_act,
        layer_norm_eps=layer_norm_eps,
        hidden_dropout_prob=hidden_dropout_prob,
        num_labels_info=num_labels_info,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        queue_size=queue_size,
        momentum=momentum,
        alpha=alpha,
        label_smoothing_factor=label_smoothing_factor,
    )
    return Blip2Model(config=config)

def ray_hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-5, 5e-4),
        # "embedding_dim": tune.randint(32, 512),
        # "hidden_size": tune.randint(256, 1024),
        # "intermediate_size": tune.randint(1024, 4096),
        # "num_hidden_layers": tune.randint(4, 18),
        "hidden_act": tune.choice(["relu", "gelu", "tanh", "quick_gelu"]),
        "hidden_dropout_prob": tune.uniform(0.0, 0.1),
        "logit_scale": tune.uniform(0.1, 10.0),
        "logit_bias": tune.uniform(-10.0, -1.0),
        "momentum": tune.uniform(0.9, 0.999),
        "alpha": tune.uniform(0.1, 0.9),
        "label_smoothing_factor": tune.uniform(0.0, 0.2),
    }

trainer = CustomTrainer(
    model=None,
    args=training_args,
    model_init=model_init,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator, 
    logger=args.logger
)

def compute_objective(metrics: Dict[str, float]) -> float:
    return metrics["eval_top1_avg"]

best_run = trainer.hyperparameter_search(
    hp_space=ray_hp_space,
    compute_objective=compute_objective,
    n_trials=60,
    direction="maximize",
    backend="ray"
)

ray_results = best_run.run_summary.results_df
ray_results.to_csv(os.path.join(training_args.output_dir, "ray_results.csv"))