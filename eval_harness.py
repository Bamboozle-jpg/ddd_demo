import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

OUTPUT_DIR = "eval_outputs"
INPUT_FILE = os.path.join(OUTPUT_DIR, "raw_predictions.json")


def load_predictions(path):
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def compute_metrics(df):
    y_true = df["true_label"]
    y_pred = df["pred_label"]
    labels = ["POSITIVE", "NEGATIVE"]

    accuracy = round(accuracy_score(y_true, y_pred), 4)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels
    )

    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels
    )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "positive_precision": round(per_class_p[0], 4),
        "positive_recall": round(per_class_r[0], 4),
        "positive_f1": round(per_class_f1[0], 4),
        "negative_precision": round(per_class_p[1], 4),
        "negative_recall": round(per_class_r[1], 4),
        "negative_f1": round(per_class_f1[1], 4),
        "total_examples": len(df),
    }
    return metrics


def save_metrics_csv(metrics, path):
    rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def save_confusion_matrix(df, path):
    labels = ["POSITIVE", "NEGATIVE"]
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - DistilBERT SST-2")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confidence_distribution(df, path):
    correct = df[df["correct"]]["pred_score"]
    incorrect = df[~df["correct"]]["pred_score"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct, bins=20, range=(0.5, 1.0), color="green", alpha=0.6, label="Correct")
    ax.hist(incorrect, bins=20, range=(0.5, 1.0), color="red", alpha=0.6, label="Incorrect")
    ax.set_title("Prediction Confidence Distribution")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_latex_table(metrics, path):
    acc = metrics["accuracy"]
    rows = [
        ("Positive", metrics["positive_precision"], metrics["positive_recall"], metrics["positive_f1"]),
        ("Negative", metrics["negative_precision"], metrics["negative_recall"], metrics["negative_f1"]),
        ("Macro Avg", metrics["macro_precision"], metrics["macro_recall"], metrics["macro_f1"]),
    ]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Evaluation Results: DistilBERT on SST-2 Validation Set}",
        r"\label{tab:eval-results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Class & Precision & Recall & F1-Score \\",
        r"\midrule",
    ]
    for cls, p, r, f1 in rows:
        lines.append(f"{cls} & {p:.4f} & {r:.4f} & {f1:.4f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption*{{Overall Accuracy: {acc:.4f}}}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    df = load_predictions(INPUT_FILE)
    metrics = compute_metrics(df)

    outputs = {
        "metrics.csv": os.path.join(OUTPUT_DIR, "metrics.csv"),
        "confusion_matrix.png": os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        "confidence_distribution.png": os.path.join(OUTPUT_DIR, "confidence_distribution.png"),
        "metrics_table.tex": os.path.join(OUTPUT_DIR, "metrics_table.tex"),
    }

    save_metrics_csv(metrics, outputs["metrics.csv"])
    save_confusion_matrix(df, outputs["confusion_matrix.png"])
    save_confidence_distribution(df, outputs["confidence_distribution.png"])
    save_latex_table(metrics, outputs["metrics_table.tex"])

    print("Evaluation complete. Files created:")
    for name, path in outputs.items():
        print(f"  {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
