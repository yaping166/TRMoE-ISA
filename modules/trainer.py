import os

import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers.trainer_callback import TrainerCallback


class FFNMoEDataCollator(DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        task_ids = torch.tensor([int(f["task_id"]) for f in features], dtype=torch.long)
        core_features = [
            {k: f[k] for k in ("input_ids", "attention_mask", "labels")}
            for f in features
        ]
        batch = super().__call__(core_features, return_tensors=return_tensors)
        batch["task_id"] = task_ids
        return batch


class BestModelSaver(TrainerCallback):

    def __init__(self, best_model_path, metric_name="eval_combined_f1"):
        self.best_model_path = best_model_path
        self.metric_name = metric_name
        self.best_metric = float("-inf")

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics") or {}
        val = metrics.get(self.metric_name)
        if val is None or val <= self.best_metric:
            return
        self.best_metric = float(val)
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        torch.save(kwargs["model"].state_dict(), self.best_model_path)


class FFNMoEMultiTaskTrainer(Seq2SeqTrainer):

    def __init__(self, lambda_sep, **kwargs):
        super().__init__(**kwargs)
        self.lambda_sep = float(lambda_sep)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss + self.lambda_sep * model._last_task_separation_loss
        if return_outputs:
            return loss, outputs
        return loss
