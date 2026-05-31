from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path(__file__).with_name("community_result.csv")
IMAGE_PATH = Path(__file__).with_name("community_result_summary.png")


def load_community_result(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {csv_path}")

    frame = pd.read_csv(csv_path)
    required_columns = {"subreddit", "community_id", "community_size"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Thieu cot: {sorted(missing_columns)}")

    frame = frame.copy()
    frame["community_id"] = pd.to_numeric(frame["community_id"], errors="coerce")
    frame["community_size"] = pd.to_numeric(frame["community_size"], errors="coerce")
    frame = frame.dropna(subset=["subreddit", "community_id", "community_size"])
    frame["community_id"] = frame["community_id"].astype(int)
    frame["community_size"] = frame["community_size"].astype(int)
    return frame


def print_statistics(frame: pd.DataFrame) -> None:
    community_summary = (
        frame.groupby("community_id")
        .agg(
            community_size=("community_size", "first"),
            num_subreddits=("subreddit", "count"),
        )
        .reset_index()
        .sort_values(["community_size", "num_subreddits"], ascending=[False, False])
    )

    total_subreddits = len(frame)
    num_communities = community_summary["community_id"].nunique()
    largest_community = community_summary.iloc[0] if not community_summary.empty else None

    print("=== Tong quan ===")
    print(f"Tong so subreddit: {total_subreddits}")
    print(f"Tong so community: {num_communities}")
    if largest_community is not None:
        print(
            "Largest community: "
            f"ID={int(largest_community['community_id'])}, "
            f"size={int(largest_community['community_size'])}, "
            f"subreddits={int(largest_community['num_subreddits'])}"
        )

    print("\n=== Phan bo kich thuoc community ===")
    size_distribution = community_summary["community_size"].value_counts().sort_index(ascending=False)
    for size, count in size_distribution.items():
        print(f"size={int(size):>4}: {int(count)} community")

    print("\n=== Top 10 community lon nhat ===")
    print(community_summary.head(10).to_string(index=False))

    print("\n=== Top 20 subreddit theo community_size ===")
    top_subreddits = frame.sort_values(["community_size", "subreddit"], ascending=[False, True]).head(20)
    print(top_subreddits.to_string(index=False))


def save_summary_image(frame: pd.DataFrame, image_path: Path = IMAGE_PATH) -> None:
    community_summary = (
        frame.groupby("community_id")
        .agg(
            community_size=("community_size", "first"),
            num_subreddits=("subreddit", "count"),
        )
        .reset_index()
        .sort_values("community_size", ascending=False)
    )

    size_counts = community_summary["community_size"].value_counts().sort_index(ascending=False)
    top_communities = community_summary.head(12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    axes[0].bar(
        [str(size) for size in size_counts.index],
        size_counts.values,
        color="#4C78A8",
    )
    axes[0].set_title("Distribution of Community Sizes")
    axes[0].set_xlabel("Community size")
    axes[0].set_ylabel("Number of communities")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(
        top_communities["community_id"].astype(str),
        top_communities["community_size"],
        color="#F58518",
    )
    axes[1].set_title("Top 12 Largest Communities")
    axes[1].set_xlabel("Community ID")
    axes[1].set_ylabel("Community size")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Community Result Summary", fontsize=16)
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDa luu anh thong ke tai: {image_path}")


def main() -> None:
    community_frame = load_community_result()
    print_statistics(community_frame)
    save_summary_image(community_frame)


if __name__ == "__main__":
    main()
