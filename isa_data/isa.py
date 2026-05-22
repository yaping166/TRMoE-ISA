import json
import random
import re

from datasets import Dataset, DatasetDict


TASK_NAMES = ["polarity", "implicit", "rationale"]
TASK_TO_ID = {name: idx for idx, name in enumerate(TASK_NAMES)}

PROMPT_TEMPLATES = {
    "polarity": (
        "Determine the sentiment polarity towards the given aspect term in the review.\n"
        "Review:\n{text}\n\nAspect:\n{term}\n\nAnswer:\n"
    ),
    "implicit": (
        "Determine whether the sentiment towards the given aspect term is explicit or implicit.\n"
        "Review:\n{text}\n\nAspect:\n{term}\n\nAnswer:\n"
    ),
    "rationale": (
        "Explain in one short sentence why the speaker holds their attitude "
        "toward the given aspect term, grounded in the review.\n"
        "Review:\n{text}\n\nAspect:\n{term}\n\nAnswer:\n"
    ),
}


EXTRA_COLUMNS = (
    "task_dataset",
    "instance_id",
    "term",
    "text",
    "target",
    "polarity",
    "implicit",
)


def build_rows(raw_examples):
    rows = []
    for example_id, ex in enumerate(raw_examples):
        text = re.sub(r"\s+", " ", ex["text"]).strip()
        for label_id, lab in enumerate(ex["labels"]):
            term = lab["term"].strip()
            polarity = lab["polarity"].strip().lower()
            implicit = int(bool(lab["implicit"]))
            rationale_text = lab["rationale"].strip()

            targets = {
                "polarity": polarity,
                "implicit": "implicit" if implicit else "explicit",
                "rationale": rationale_text,
            }
            base = {
                "instance_id": f"{example_id}-{label_id}",
                "example_id": example_id,
                "label_id": label_id,
                "text": text,
                "term": term,
                "polarity": polarity,
                "implicit": implicit,
                "rationale": rationale_text,
            }
            for task_name in TASK_NAMES:
                rows.append({
                    **base,
                    "task_dataset": task_name,
                    "input": PROMPT_TEMPLATES[task_name].format(term=term, text=text),
                    "target": targets[task_name],
                })
    return rows


def prepare_isa_dataset(train_path, test_path, val_ratio=0.1, seed=42):
    with open(train_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_raw = json.load(f)

    indices = list(range(len(train_raw)))
    random.Random(seed).shuffle(indices)
    val_size = int(len(indices) * val_ratio)
    valid_raw = [train_raw[i] for i in indices[:val_size]]
    train_split = [train_raw[i] for i in indices[val_size:]]

    return DatasetDict({
        "train": Dataset.from_list(build_rows(train_split)),
        "valid": Dataset.from_list(build_rows(valid_raw)),
        "test": Dataset.from_list(build_rows(test_raw)),
    })


def build_tokenize_function(tokenizer, max_input_length, max_output_length):
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=max_output_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["task_id"] = [TASK_TO_ID[name] for name in examples["task_dataset"]]
        for key in EXTRA_COLUMNS:
            model_inputs[key] = examples[key]
        return model_inputs

    return tokenize_function
