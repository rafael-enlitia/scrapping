"""Evaluate classifications against a human-labeled gold dataset.

Gold file format (JSONL): one JSON object per line with keys:
  - review_id: str (must match a classified review in the DB)
  - sentiment: str
  - topics: list[str]
"""

import argparse
import json
import logging
import sys

import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from src.db.queries import get_reviews_df
from src.llm.taxonomy import SENTIMENT_VALUES, TOPIC_VALUES

logger = logging.getLogger(__name__)


def load_gold(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _evaluate_method(gold: list[dict], df: pd.DataFrame, method_label: str, save_plots: bool = False):
    """Run sentiment + topic evaluation for a single classification method."""
    classified = df[df["sentiment"].notna()].copy()
    if "review_id" not in classified.columns:
        print(f"[{method_label}] No review_id column — skipping.")
        return

    classified = classified.set_index("review_id")

    matched = []
    missing = []
    for item in gold:
        rid = item["review_id"]
        if rid in classified.index:
            row = classified.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            pred_topics = row["topics"]
            if isinstance(pred_topics, str):
                try:
                    pred_topics = json.loads(pred_topics)
                except (json.JSONDecodeError, TypeError):
                    pred_topics = []
            matched.append({
                "review_id": rid,
                "gold_sentiment": item["sentiment"],
                "pred_sentiment": row["sentiment"],
                "gold_topics": item["topics"],
                "pred_topics": pred_topics if isinstance(pred_topics, list) else [],
            })
        else:
            missing.append(rid)

    if missing:
        logger.warning("[%s] Gold reviews not found or unclassified: %s", method_label, missing)

    if not matched:
        print(f"[{method_label}] No matching classified reviews found.\n")
        return

    print(f"\n{'=' * 60}")
    print(f"{method_label.upper()} EVALUATION — {len(matched)} / {len(gold)} gold reviews matched")
    print("=" * 60)

    # --- Sentiment ---
    gold_sent = [m["gold_sentiment"] for m in matched]
    pred_sent = [m["pred_sentiment"] for m in matched]

    print(f"\nAccuracy: {accuracy_score(gold_sent, pred_sent):.3f}")
    print(f"Macro F1: {f1_score(gold_sent, pred_sent, average='macro', zero_division=0):.3f}")
    print()
    print(classification_report(gold_sent, pred_sent, labels=SENTIMENT_VALUES, zero_division=0))

    labels_present = sorted(set(gold_sent + pred_sent))
    cm = confusion_matrix(gold_sent, pred_sent, labels=labels_present)
    print("Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=labels_present, columns=labels_present)
    print(cm_df.to_string())
    print()

    if save_plots:
        prefix = f"data/{method_label}"
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels_present,
            y=labels_present,
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig.update_layout(title=f"{method_label} Sentiment Confusion Matrix")
        fig.write_html(f"{prefix}_sentiment_confusion_matrix.html")
        fig.write_image(f"{prefix}_sentiment_confusion_matrix.png")
        print(f"Saved: {prefix}_sentiment_confusion_matrix.html/.png")

    # --- Topics (multilabel) ---
    if method_label == "nlp":
        print("(Topic evaluation skipped for NLP — LDA topics are unsupervised and don't map 1:1 to gold labels)\n")
        return

    mlb = MultiLabelBinarizer(classes=TOPIC_VALUES)
    gold_topics_bin = mlb.fit_transform([m["gold_topics"] for m in matched])
    pred_topics_bin = mlb.transform([m["pred_topics"] for m in matched])

    print("TOPIC EVALUATION (multilabel)")
    print("-" * 40)
    print(f"Micro F1:  {f1_score(gold_topics_bin, pred_topics_bin, average='micro', zero_division=0):.3f}")
    print(f"Macro F1:  {f1_score(gold_topics_bin, pred_topics_bin, average='macro', zero_division=0):.3f}")
    print(f"Precision: {precision_score(gold_topics_bin, pred_topics_bin, average='micro', zero_division=0):.3f}")
    print(f"Recall:    {recall_score(gold_topics_bin, pred_topics_bin, average='micro', zero_division=0):.3f}")
    print()

    per_label = pd.DataFrame({
        "topic": TOPIC_VALUES,
        "gold_count": gold_topics_bin.sum(axis=0),
        "pred_count": pred_topics_bin.sum(axis=0),
        "f1": [
            f1_score(gold_topics_bin[:, i], pred_topics_bin[:, i], zero_division=0)
            for i in range(len(TOPIC_VALUES))
        ],
    })
    print(per_label.to_string(index=False))
    print()

    if save_plots:
        prefix = f"data/{method_label}"
        fig = px.bar(per_label, x="topic", y="f1", title=f"{method_label} Per-topic F1 Score")
        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(f"{prefix}_topic_f1_scores.html")
        fig.write_image(f"{prefix}_topic_f1_scores.png")
        print(f"Saved: {prefix}_topic_f1_scores.html/.png")


def evaluate(gold_path: str, method: str = "both", save_plots: bool = False, app_id: str | None = None):
    gold = load_gold(gold_path)
    if not gold:
        print("Gold dataset is empty.")
        sys.exit(1)

    methods_to_eval = []
    if method in ("llm", "both"):
        methods_to_eval.append("llm")
    if method in ("nlp", "both"):
        methods_to_eval.append("nlp")

    for m in methods_to_eval:
        df = get_reviews_df(app_id=app_id, method=m)
        _evaluate_method(gold, df, method_label=m, save_plots=save_plots)

    if method == "both" and len(methods_to_eval) == 2:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)

        for m in methods_to_eval:
            df = get_reviews_df(app_id=app_id, method=m)
            classified = df[df["sentiment"].notna()].set_index("review_id")
            gold_sent, pred_sent = [], []
            for item in gold:
                if item["review_id"] in classified.index:
                    row = classified.loc[item["review_id"]]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    gold_sent.append(item["sentiment"])
                    pred_sent.append(row["sentiment"])

            if gold_sent:
                acc = accuracy_score(gold_sent, pred_sent)
                f1 = f1_score(gold_sent, pred_sent, average="macro", zero_division=0)
                print(f"  {m.upper():>4}  Accuracy={acc:.3f}  Macro-F1={f1:.3f}  (n={len(gold_sent)})")
            else:
                print(f"  {m.upper():>4}  No matching reviews")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifications against gold labels")
    parser.add_argument("--gold", required=True, help="Path to gold JSONL file")
    parser.add_argument("--method", default="both", choices=["llm", "nlp", "both"], help="Which method to evaluate")
    parser.add_argument("--app-id", default=None, help="Scope evaluation to a specific app (e.g. com.whatsapp)")
    parser.add_argument("--save-plots", action="store_true", help="Save confusion matrix and charts as HTML/PNG")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    evaluate(args.gold, method=args.method, save_plots=args.save_plots, app_id=args.app_id)


if __name__ == "__main__":
    main()
