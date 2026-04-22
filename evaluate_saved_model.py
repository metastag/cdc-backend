import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

from model import get_analyzer


def evaluate_distortion_metrics(
    csv_path: str,
    text_col: str = "text",
    label_col: str = "label",
    healthy_label: str = "No Distortion",
    test_size: float = 0.2,
    random_state: int = 42,
    limit: int = 0,
) -> dict:
    df = pd.read_csv(csv_path)

    required_cols = {text_col, label_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)

    y = (df[label_col] != healthy_label).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if limit > 0:
        X_test = X_test.iloc[:limit]
        y_test = y_test.iloc[:limit]

    analyzer = get_analyzer()

    y_pred = []
    for text in X_test:
        analysis = analyzer.analyze_entry(text)
        y_pred.append(1 if analysis.get("is_distorted", False) else 0)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics = {
        "dataset": str(Path(csv_path).name),
        "rows_used": int(len(X_test)),
        "healthy_label": healthy_label,
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        },
        "confusion_matrix": {
            "labels": ["healthy(0)", "distorted(1)"],
            "matrix": cm.tolist(),
        },
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved journal model without retraining."
    )
    parser.add_argument("--csv", default="combined_dataset.csv", help="Path to labeled CSV")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument(
        "--healthy-label",
        default="No Distortion",
        help="Label value representing the healthy/non-distorted class",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Held-out split ratio"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for split"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only first N test rows (0 = all)",
    )

    args = parser.parse_args()

    results = evaluate_distortion_metrics(
        csv_path=args.csv,
        text_col=args.text_col,
        label_col=args.label_col,
        healthy_label=args.healthy_label,
        test_size=args.test_size,
        random_state=args.random_state,
        limit=args.limit,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
