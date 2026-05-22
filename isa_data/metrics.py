import re
import string
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


POLARITY_LABELS = ["positive", "negative", "neutral"]
IMPLICIT_LABELS = ["implicit", "explicit"]


def _norm(text):
    return (text or "").strip().lower()


def compute_polarity_metrics(pred_labels, true_labels):
    preds = [_norm(x) for x in pred_labels]
    trues = [_norm(x) for x in true_labels]
    if not trues:
        return {"accuracy": 0.0, "f1": 0.0, "count": 0}
    acc = accuracy_score(trues, preds)
    f1 = f1_score(
        trues, preds, labels=POLARITY_LABELS, average="macro", zero_division=0
    )
    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "count": len(trues)}


def compute_implicit_metrics(pred_labels, true_labels):
    preds = [_norm(x) for x in pred_labels]
    trues = [_norm(x) for x in true_labels]
    if not trues:
        return {"accuracy": 0.0, "f1": 0.0, "count": 0}
    acc = accuracy_score(trues, preds)
    f1 = f1_score(
        trues,
        preds,
        labels=IMPLICIT_LABELS,
        average="macro",
        zero_division=0,
    )
    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "count": len(trues)}


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_PUNCT = set(string.punctuation)


def _squad_normalize(text):
    s = (text or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _squad_token_f1(pred, gold):
    pred_tokens = _squad_normalize(pred).split()
    gold_tokens = _squad_normalize(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rationale_metrics(pred_rationales, true_rationales):
    if not true_rationales:
        return {"f1": 0.0, "count": 0}
    scores = [_squad_token_f1(p, g) for p, g in zip(pred_rationales, true_rationales)]
    f1 = float(np.mean(scores)) if scores else 0.0
    return {"f1": round(f1, 4), "count": len(scores)}


def _bucket_by_task(eval_dataset, decoded_preds, decoded_labels):
    buckets = {"polarity": ([], []), "implicit": ([], []), "rationale": ([], [])}
    n = min(len(eval_dataset), len(decoded_preds), len(decoded_labels))
    task_field = eval_dataset["task_dataset"]
    for i in range(n):
        task = task_field[i]
        if task not in buckets:
            continue
        buckets[task][0].append(decoded_preds[i])
        buckets[task][1].append(decoded_labels[i])
    return buckets


def compute_task_metrics(eval_dataset, decoded_preds, decoded_labels):
    buckets = _bucket_by_task(eval_dataset, decoded_preds, decoded_labels)

    pol = compute_polarity_metrics(*buckets["polarity"])
    imp = compute_implicit_metrics(*buckets["implicit"])
    rat = compute_rationale_metrics(*buckets["rationale"])

    task_f1s = []
    if pol["count"] > 0:
        task_f1s.append(pol["f1"])
    if imp["count"] > 0:
        task_f1s.append(imp["f1"])
    if rat["count"] > 0:
        task_f1s.append(rat["f1"])
    combined_f1 = round(float(np.mean(task_f1s)), 4) if task_f1s else 0.0

    return {
        "polarity_accuracy": pol["accuracy"],
        "polarity_f1": pol["f1"],
        "polarity_count": pol["count"],
        "implicit_accuracy": imp["accuracy"],
        "implicit_f1": imp["f1"],
        "implicit_count": imp["count"],
        "rationale_f1": rat["f1"],
        "rationale_count": rat["count"],
        "combined_f1": combined_f1,
    }


def build_compute_metrics(tokenizer, eval_dataset):
    pad_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions == -100, pad_id, predictions)
        labels = np.where(labels == -100, pad_id, labels)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return compute_task_metrics(eval_dataset, decoded_preds, decoded_labels)

    return compute_metrics
