"""
Evaluation harness. Accuracy alone is a weak metric for a language model
(with ~4800 vocab words, top-1 accuracy is intentionally hard and doesn't
tell you how "close" wrong predictions were). This module adds the
metrics an interviewer would actually expect:

- Perplexity: exp(cross-entropy). Standard LM metric; lower is better.
- Top-k accuracy: was the true word in the model's top 1/3/5 guesses.
- A comparison table across all registered architectures.
"""

import time
import numpy as np


def perplexity(model, x, y, batch_size: int = 256) -> float:
    losses = []
    for i in range(0, len(x), batch_size):
        xb, yb = x[i : i + batch_size], y[i : i + batch_size]
        preds = model.predict(xb, verbose=0)
        # sparse categorical cross-entropy per example
        true_probs = preds[np.arange(len(yb)), yb.astype(int)]
        true_probs = np.clip(true_probs, 1e-9, 1.0)
        losses.append(-np.log(true_probs))
    mean_loss = np.mean(np.concatenate(losses))
    return float(np.exp(mean_loss))


def top_k_accuracy(model, x, y, k: int = 5, batch_size: int = 256) -> float:
    correct = 0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i : i + batch_size], y[i : i + batch_size]
        preds = model.predict(xb, verbose=0)
        topk = np.argsort(-preds, axis=1)[:, :k]
        correct += sum(yb[j] in topk[j] for j in range(len(yb)))
    return correct / len(x)


def evaluate_model(model, x_test, y_test) -> dict:
    return {
        "perplexity": perplexity(model, x_test, y_test),
        "top1_acc": top_k_accuracy(model, x_test, y_test, k=1),
        "top3_acc": top_k_accuracy(model, x_test, y_test, k=3),
        "top5_acc": top_k_accuracy(model, x_test, y_test, k=5),
    }


def compare_models(trained_models: dict, x_test, y_test) -> "list[dict]":
    """trained_models: {name: (model, n_params, train_seconds)}"""
    rows = []
    for name, (model, n_params, train_seconds) in trained_models.items():
        t0 = time.time()
        metrics = evaluate_model(model, x_test, y_test)
        eval_seconds = time.time() - t0
        rows.append(
            {
                "model": name,
                "params": n_params,
                "train_time_s": round(train_seconds, 1),
                "eval_time_s": round(eval_seconds, 1),
                **{k: round(v, 4) for k, v in metrics.items()},
            }
        )
    return rows


def print_comparison_table(rows: "list[dict]"):
    if not rows:
        print("No results.")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))
