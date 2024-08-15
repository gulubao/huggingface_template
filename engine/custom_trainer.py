from transformers import Trainer
import torch
from typing import Dict, Union, Any

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        自定义损失计算方法
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        自定义训练步骤
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def log(self, logs: Dict[str, float]) -> None:
        """
        自定义日志记录方法
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        print(f"Step: {self.state.global_step}, Loss: {logs.get('loss', 'N/A')}")

    def create_optimizer(self):
        """
        自定义优化器创建方法
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int):
        """
        自定义学习率调度器创建方法
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_training_steps)
        return self.lr_scheduler