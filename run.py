import json
import os

import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments

from modules.ffn_moe import FFNMoET5ForConditionalGeneration
from modules.trainer import BestModelSaver, FFNMoEDataCollator, FFNMoEMultiTaskTrainer
from isa_data.isa import build_tokenize_function, prepare_isa_dataset
from isa_data.metrics import build_compute_metrics, compute_polarity_metrics
from utils import decode_sequences, remove_best_checkpoint


def run(args, logger):

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = load_data(args, logger, tokenizer)

    model = load_model(args, logger)
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    trainer, best_model_path = build_trainer(args, logger, model, tokenizer, tokenized)

    if args.do_train:
        logger.info("Starting training!")
        trainer.train()

        if os.path.exists(best_model_path):
            logger.info("Loading best checkpoint from {}".format(best_model_path))
            state = torch.load(best_model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)

    if args.do_predict:
        logger.info("Running prediction on test set")
        evaluate(args, logger, trainer, tokenizer, tokenized["test"])

    remove_best_checkpoint(args.output_dir)


def load_data(args, logger, tokenizer):
    logger.info("Loading ISA dataset from {} / {}".format(args.train_path, args.test_path))
    datasets = prepare_isa_dataset(
        args.train_path,
        args.test_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    logger.info("#train={}, #valid={}, #test={}".format(
        len(datasets["train"]), len(datasets["valid"]), len(datasets["test"])
    ))
    tokenize_fn = build_tokenize_function(
        tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
    )
    return datasets.map(tokenize_fn, batched=True)


def load_model(args, logger):
    logger.info("Building FFN-MoE Flan-t5 model from {}".format(args.model_name))
    return FFNMoET5ForConditionalGeneration(
        model_name=args.model_name,
        num_experts=args.num_experts,
        encoder_moe_layers=args.encoder_moe_layers,
        decoder_moe_layers=args.decoder_moe_layers,
        task_embedding_dim=args.task_embedding_dim,
        router_hidden_dim=args.router_hidden_dim,
        top_k=args.top_k,
    )


def build_trainer(args, logger, model, tokenizer, tokenized):
    data_collator = FFNMoEDataCollator(tokenizer=tokenizer, model=model.base_model)
    best_model_path = os.path.join(args.output_dir, "best_model.pt")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_seed=args.seed,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        save_strategy="no",
        predict_with_generate=True,
        generation_max_length=args.max_output_length,
        report_to=[],
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        load_best_model_at_end=False,
        bf16=args.bf16,
    )

    trainer = FFNMoEMultiTaskTrainer(
        lambda_sep=args.lambda_sep,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["valid"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer, tokenized["valid"]),
        callbacks=[BestModelSaver(best_model_path=best_model_path)],
    )

    return trainer, best_model_path


def evaluate(args, logger, trainer, tokenizer, tokenized_test):
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    trainer.compute_metrics = None
    results = trainer.predict(tokenized_test)
    pred_texts = decode_sequences(tokenizer, results.predictions)

    detailed_rows = []
    pol_pred, pol_gold, pol_implicit_idx = [], [], []

    for idx, ex in enumerate(tokenized_test):
        pred_text = pred_texts[idx].strip()
        task = ex["task_dataset"]
        detailed_rows.append({
            "instance_id": ex["instance_id"],
            "task_dataset": task,
            "text": ex["text"],
            "term": ex["term"],
            "gold_target": ex["target"],
            "pred_target": pred_text,
            "implicit": bool(ex["implicit"]),
        })

        if task == "polarity":
            pol_gold.append(ex["target"].strip().lower())
            pol_pred.append(pred_text.lower())
            if bool(ex["implicit"]):
                pol_implicit_idx.append(len(pol_gold) - 1)

    overall = compute_polarity_metrics(pol_pred, pol_gold)
    implicit_metrics = None
    if pol_implicit_idx:
        implicit_metrics = compute_polarity_metrics(
            [pol_pred[i] for i in pol_implicit_idx],
            [pol_gold[i] for i in pol_implicit_idx],
        )

    summary = {
        "overall": {"accuracy": overall["accuracy"], "f1": overall["f1"]},
        "implicit_true": (
            {"accuracy": implicit_metrics["accuracy"], "f1": implicit_metrics["f1"]}
            if implicit_metrics is not None
            else None
        ),
        "implicit_true_count": len(pol_implicit_idx),
        "split_count": len(tokenized_test),
        "polarity_count": overall["count"],
    }
    with open(os.path.join(results_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(results_dir, "detailed_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(detailed_rows, f, indent=2, ensure_ascii=False)

    logger.info("[Eval] Test polarity overall: %s", summary["overall"])
    if implicit_metrics is not None:
        logger.info(
            "[Eval] Test polarity on implicit subset (n=%d): %s",
            len(pol_implicit_idx), summary["implicit_true"],
        )
