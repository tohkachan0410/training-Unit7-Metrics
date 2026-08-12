import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms

from dataset import resolve_image_path
from models import build_model


ROOT = Path("/home/u5450760/train_6")


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def predict(model, image, transform, device):
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor).squeeze().item()

    # training target 是 /100
    score = output * 100.0

    # 最終品質定義限制 0~100
    score = float(
        np.clip(score, 0.0, 100.0)
    )

    return score


def build_tta_images(image):
    """
    固定、可重現的 5 種輕量 TTA。
    """
    return [
        image,

        ImageOps.mirror(image),

        ImageEnhance.Brightness(
            image
        ).enhance(0.95),

        ImageEnhance.Brightness(
            image
        ).enhance(1.05),

        ImageEnhance.Contrast(
            image
        ).enhance(1.05),
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet18",
            "vgg16",
            "resnet50",
        ],
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model = build_model(
        args.model,
        pretrained=False,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    train = pd.read_csv(
        ROOT / "data/splits/train.csv"
    )

    val = pd.read_csv(
        ROOT / "data/splits/val.csv"
    )

    ref = pd.read_csv(
        ROOT /
        "data/annotations/"
        "ranking_reference_top100.csv"
    )

    # 不建立 candidate_pool.csv
    # 直接記憶體 concat
    df = pd.concat(
        [train, val, ref],
        ignore_index=True,
    )

    assert len(df) == 500
    assert df["image_id"].is_unique

    transform = get_transform()

    rows = []

    print(
        f"Scoring {len(df)} images "
        f"with {args.model}"
    )

    for index, row in df.iterrows():
        path = resolve_image_path(
            row["image_path"]
        )

        image = Image.open(path).convert("RGB")

        # Raw
        raw_score = predict(
            model,
            image,
            transform,
            device,
        )

        # TTA
        tta_images = build_tta_images(image)

        tta_scores = [
            predict(
                model,
                img,
                transform,
                device,
            )
            for img in tta_images
        ]

        tta_mean = float(
            np.mean(tta_scores)
        )

        tta_std = float(
            np.std(tta_scores)
        )

        rows.append({
            "image_id":
                str(row["image_id"]),

            "image_path":
                row["image_path"],

            "model":
                args.model,

            "raw_quality_score":
                raw_score,

            "tta_mean":
                tta_mean,

            "tta_std":
                tta_std,
        })

        if (
            (index + 1) % 50 == 0
            or index + 1 == len(df)
        ):
            print(
                f"{index + 1}/{len(df)}"
            )

    output = pd.DataFrame(rows)

    outdir = (
        ROOT /
        "outputs" /
        "scores"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    outfile = (
        outdir /
        f"{args.model}_scores.csv"
    )

    output.to_csv(
        outfile,
        index=False
    )

    print("Saved:", outfile)


if __name__ == "__main__":
    main()
