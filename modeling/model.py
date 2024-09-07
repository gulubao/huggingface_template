
from .configuration_custom import Blip2Config
from .modeling_custom import Blip2Model


def build_model(args):
    config = Blip2Config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_labels=args.num_labels
    )
    return Blip2Model(config=config)