from transformers import AutoModelForSequenceClassification
from .custom_models import CustomModel1, CustomModel2
from .custom_config import CustomConfig

def build_model(args):
    if args.use_custom_model:
        if args.custom_model_type == "custom1":
            return CustomModel1(args)
        elif args.custom_model_type == "custom2":
            config = CustomConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads=args.num_attention_heads,
                intermediate_size=args.intermediate_size,
                hidden_dropout_prob=args.hidden_dropout_prob,
                num_labels=args.num_labels
            )
            return CustomModel2.from_pretrained(args.model_name, config=config)
        else:
            raise ValueError(f"未知的自定义模型类型: {args.custom_model_type}")
    else:
        return AutoModelForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=args.num_labels
        )