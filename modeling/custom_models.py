import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class CustomModel(nn.Module):
    def __init__(self, args):
        super(CustomModel, self).__init__()
        self.config = AutoConfig.from_pretrained(args.model_name)
        self.base_model = AutoModel.from_pretrained(args.model_name)
        
        # 自定义层
        self.dropout = nn.Dropout(args.dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, args.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
from transformers import PreTrainedModel, AutoModel
from .custom_config import CustomConfig

class CustomModel2(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.base_model = AutoModel.from_config(config)
        
        # 自定义层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, config.num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = CustomConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        model.base_model = AutoModel.from_pretrained(pretrained_model_name_or_path, config=config)
        return model