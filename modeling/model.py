
from .configuration_blip_2 import Blip2Config
from .modeling_blip_2 import Blip2ForConditionalGeneration


def build_model(args):
    config = Blip2Config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_labels=args.num_labels
    )
    return Blip2ForConditionalGeneration.from_pretrained(args.model_name, config=config)
