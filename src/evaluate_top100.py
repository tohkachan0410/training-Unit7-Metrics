from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    average_precision_score,
)


ROOT = Path("/home/u5450760/train_6")

SCORE_DIR = ROOT / "outputs/scores"
RANK_DIR = ROOT / "outputs/rankings"
METRIC_DIR = ROOT / "outputs/metrics"

RANK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)


combined = pd.read_csv(
    SCORE_DIR / "combined_scores.csv"
)

combined["image_id"] = (
    combined["image_id"].astype(str)
)


reference = pd.read_csv(
    ROOT /
    "data/annotations/"
    "ranking_reference_top100.csv"
)

reference_ids = set(
    reference["image_id"].astype(str)
)


assert len(reference_ids) == 100
assert len(combined) == 500


methods = {
    "resnet18_raw":
        "resnet18_raw",

    "vgg16_raw":
        "vgg16_raw",

    "resnet50_raw":
        "resnet50_raw",

    "resnet18_confidence":
        "resnet18_confidence",

    "vgg16_confidence":
        "vgg16_confidence",

    "resnet50_confidence":
        "resnet50_confidence",

    "ensemble":
        "ensemble_score",
}


metric_rows = []


combined["human_relevant"] = (
    combined["image_id"]
    .isin(reference_ids)
    .astype(int)
)


for method_name, score_column in methods.items():

    ranked = combined.sort_values(
        by=[
            score_column,
            "image_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    ranked["rank"] = (
        ranked.index + 1
    )

    top100 = ranked.head(100).copy()

    model_ids = set(
        top100["image_id"]
    )

    intersection_ids = (
        reference_ids & model_ids
    )

    intersection = len(
        intersection_ids
    )

    overlap = (
        intersection / 100.0
    )

    union_size = len(
        reference_ids | model_ids
    )

    jaccard = (
        intersection / union_size
        if union_size > 0
        else 0.0
    )

    ap = average_precision_score(
        combined["human_relevant"],
        combined[score_column],
    )

    top100[
        [
            "rank",
            "image_id",
            "image_path",
            score_column,
        ]
    ].to_csv(
        RANK_DIR /
        f"{method_name}_top100.csv",
        index=False,
    )

    metric_rows.append({
        "method":
            method_name,

        "intersection":
            intersection,

        "overlap":
            overlap,

        "jaccard":
            jaccard,

        "AP":
            ap,
    })

    print(
        f"{method_name:25s} | "
        f"I={intersection:3d} | "
        f"Overlap={overlap:.4f} | "
        f"Jaccard={jaccard:.4f} | "
        f"AP={ap:.4f}"
    )


metrics = pd.DataFrame(
    metric_rows
)

metrics = metrics.sort_values(
    "AP",
    ascending=False
)

metrics.to_csv(
    METRIC_DIR /
    "top100_metrics.csv",
    index=False,
)

print()
print("=" * 70)
print("FINAL METRICS")
print("=" * 70)
print(metrics.to_string(index=False))
