from transformers import Trainer, TrainingArguments
from accelerate import Accelerator

from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} ended")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print(f"Step {state.global_step}: loss = {logs['loss']:.4f}")

def do_train(args, model, train_dataset, eval_dataset, accelerator: Accelerator):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # 启用 Accelerate 的功能
        fp16=accelerator.state.use_fp16,
        deepspeed=accelerator.state.deepspeed_plugin,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[CustomCallback()]
    )

    # 使用 Accelerator 包装 Trainer
    trainer = accelerator.prepare(trainer)

    trainer.train()

    # 评估
    eval_result = trainer.evaluate()
    return eval_result