from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_scores(csv_path: Path, min_score: float, max_score: float) -> list[float]:
    scores: list[float] = []

    for chunk in pd.read_csv(csv_path, usecols=["Similarity_Score"], chunksize=500_000):
        filtered = chunk["Similarity_Score"].dropna()
        filtered = filtered[(filtered >= min_score) & (filtered <= max_score)]
        scores.extend(filtered.astype(float).tolist())

    return scores


def plot_histogram(scores: list[float], min_score: float, max_score: float, output_path: Path) -> None:
    if not scores:
        raise ValueError(
            f"Khong tim thay gia tri nao trong khoang [{min_score}, {max_score}] de ve bieu do."
        )

    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=50, range=(min_score, max_score), color="#2a9d8f", edgecolor="white")
    plt.title(f"Phan phoi Similarity_Score tu {min_score} den {max_score}")
    plt.xlabel("Similarity_Score")
    plt.ylabel("So luong cap subreddit")
    plt.xlim(min_score, max_score)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ve histogram phan phoi Similarity_Score.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="subreddit_similarity_results.csv",
        help="Duong dan toi file CSV dau vao.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="Can duoi cua khoang diem can ve.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.99,
        help="Can tren cua khoang diem can ve.",
    )
    parser.add_argument(
        "--output",
        default="similarity_histogram_0p3_0p99.png",
        help="File anh dau ra.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_path = Path(args.output)

    scores = load_scores(csv_path, args.min_score, args.max_score)
    plot_histogram(scores, args.min_score, args.max_score, output_path)
    print(f"Da luu bieu do tai: {output_path}")
    print(f"Tong so mau trong khoang [{args.min_score}, {args.max_score}]: {len(scores)}")


if __name__ == "__main__":
    main()