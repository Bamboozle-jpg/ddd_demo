# run small Transformer model on sentiment analysis dataset (SST-2)
# micromamba env create -n eval_demo python=3.12
# pip install transformers datasets torch

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import json
import os


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Load model
classifier = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME, device=-1)
print("Model loaded successfully.\n")

# Load dataset
dataset = load_dataset("glue", "sst2", split="validation")
print(f"Dataset size: {len(dataset)} examples\n")
LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}

# batch setup
NUM_EXAMPLES = 200  # bump to len(dataset) for full eval
samples = dataset.select(range(NUM_EXAMPLES))
texts = list(samples["sentence"])  # plain list[str] required by tokenizer
true_labels = [LABEL_MAP[label] for label in samples["label"]]

# Batch predict
predictions = classifier(texts, batch_size=32, truncation=True)
pred_labels = [p["label"] for p in predictions]
pred_scores = [p["score"] for p in predictions]

# Save predictions
OUTPUT_DIR = "eval_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []
for i in range(NUM_EXAMPLES):
    result = {
        "id": i,
        "text": texts[i],
        "true_label": true_labels[i],
        "pred_label": pred_labels[i],
        "pred_score": pred_scores[i],
        "correct": true_labels[i] == pred_labels[i],
    }
    results.append(result)
results_path = os.path.join(OUTPUT_DIR, "raw_predictions.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Raw predictions saved to: {results_path}")

# ──────────────────────────────────────────────
# 5. Quick Sanity Check
# ──────────────────────────────────────────────
correct = sum(1 for r in results if r["correct"])
print(f"\nQuick check: {correct}/{NUM_EXAMPLES} correct ({correct/NUM_EXAMPLES:.1%} accuracy)")
print("\n--- Sample Predictions ---")
for r in results[:5]:
    status = "✓" if r["correct"] else "✗"
    print(f"  {status} [{r['true_label']:>8} → {r['pred_label']:>8} ({r['pred_score']:.3f})] {r['text'][:80]}")

print(f"""
{'='*60}
Setup complete! You now have:

  1. Model: {MODEL_NAME}
  2. Dataset: SST-2 validation ({NUM_EXAMPLES} examples)
  3. Raw predictions: {results_path}

Next step: Use your spec to build the evaluation harness
that computes metrics, generates plots, and outputs a
LaTeX table from raw_predictions.json.
{'='*60}
""")