from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/u5450760/train_6")
SCORE_DIR = ROOT / "outputs/scores"


models = [
    "resnet18",
    "vgg16",
    "resnet50",
]


dfs = {}

for model in models:
    path = SCORE_DIR / f"{model}_scores.csv"

    df = pd.read_csv(path)

    assert len(df) == 500
    assert df["image_id"].is_unique
    assert df["raw_quality_score"].notna().all()
    assert df["tta_std"].notna().all()

    std = df["tta_std"].astype(float)

    std_min = std.min()
    std_max = std.max()

    if std_max > std_min:
        uncertainty_norm = (
            (std - std_min)
            / (std_max - std_min)
        )
    else:
        uncertainty_norm = np.zeros(
            len(df)
        )

    df["uncertainty_normalized"] = (
        uncertainty_norm
    )

    df["confidence_adjusted_score"] = (
        df["tta_mean"]
        * (
            1.0
            - df["uncertainty_normalized"]
        )
    )

    df.to_csv(
        SCORE_DIR /
        f"{model}_scores_with_confidence.csv",
        index=False,
    )

    dfs[model] = df


base = dfs["resnet18"][
    ["image_id", "image_path"]
].copy()


for model in models:
    df = dfs[model].copy()

    score = df["raw_quality_score"]

    minimum = score.min()
    maximum = score.max()

    if maximum > minimum:
        normalized = (
            (score - minimum)
            / (maximum - minimum)
        )
    else:
        normalized = np.zeros(
            len(df)
        )

    temp = pd.DataFrame({
        "image_id":
            df["image_id"].astype(str),

        f"{model}_raw":
            df["raw_quality_score"],

        f"{model}_confidence":
            df[
                "confidence_adjusted_score"
            ],

        f"{model}_normalized":
            normalized,
    })

    base["image_id"] = (
        base["image_id"].astype(str)
    )

    base = base.merge(
        temp,
        on="image_id",
        how="inner",
    )


assert len(base) == 500


base["ensemble_score"] = (
    base["resnet18_normalized"]
    + base["vgg16_normalized"]
    + base["resnet50_normalized"]
) / 3.0


outfile = (
    SCORE_DIR /
    "combined_scores.csv"
)

base.to_csv(
    outfile,
    index=False
)

print("Saved:", outfile)
print("Rows:", len(base))
