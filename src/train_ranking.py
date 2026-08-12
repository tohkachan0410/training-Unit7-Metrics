import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
import wandb

from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import (
    RankingDataset,
    get_train_transform,
    get_eval_transform,
)
from models import build_model


PROJECT_ROOT = Path("/home/u5450760/train_6")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True
            )

            target = batch["target"].float().to(
                device,
                non_blocking=True
            )

            output = model(images).squeeze(1)

            loss = criterion(
                output,
                target
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions.extend(
                output.detach().cpu().numpy()
            )

            targets.extend(
                target.detach().cpu().numpy()
            )

    val_loss = total_loss / len(loader.dataset)

    # 轉回 0~100
    pred_100 = np.array(predictions) * 100.0
    target_100 = np.array(targets) * 100.0

    mae = mean_absolute_error(
        target_100,
        pred_100
    )

    rmse = mean_squared_error(
        target_100,
        pred_100
    ) ** 0.5

    if len(np.unique(pred_100)) <= 1:
        spearman = 0.0
    else:
        spearman = spearmanr(
            target_100,
            pred_100
        ).statistic

        if np.isnan(spearman):
            spearman = 0.0

    return (
        val_loss,
        mae,
        rmse,
        spearman,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"]
    )

    parser.add_argument(
        "--run-name",
        required=True
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("Model:", cfg["model_name"])
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("=" * 60)

    train_dataset = RankingDataset(
        csv_path=PROJECT_ROOT / cfg["train_csv"],
        transform=get_train_transform(
            cfg.get("input_size", 224)
        ),
        target_column=cfg.get(
            "target_column",
            "quality_score"
        ),
    )

    val_dataset = RankingDataset(
        csv_path=PROJECT_ROOT / cfg["val_csv"],
        transform=get_eval_transform(
            cfg.get("input_size", 224)
        ),
        target_column=cfg.get(
            "target_column",
            "quality_score"
        ),
    )

    batch_size = int(
        cfg.get("batch_size", 32)
    )

    num_workers = int(
        cfg.get("num_workers", 4)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = build_model(
        cfg["model_name"],
        pretrained=cfg.get(
            "pretrained",
            True
        ),
    ).to(device)

    criterion = nn.SmoothL1Loss()

    optimizer = AdamW(
        model.parameters(),
        lr=float(
            cfg.get(
                "learning_rate",
                1e-4
            )
        ),
        weight_decay=float(
            cfg.get(
                "weight_decay",
                1e-4
            )
        ),
    )

    wandb.init(
        project=cfg.get(
            "wandb_project",
            "training-unit7-ranking"
        ),
        name=args.run_name,
        mode=args.wandb_mode,
        config=cfg,
    )

    checkpoint_dir = (
        PROJECT_ROOT /
        "outputs" /
        "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        checkpoint_dir /
        f"{cfg['model_name']}_best.pt"
    )

    epochs = int(
        cfg.get("epochs", 30)
    )

    patience = int(
        cfg.get(
            "early_stopping_patience",
            5
        )
    )

    best_val_loss = float("inf")
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(
                device,
                non_blocking=True
            )

            target = batch["target"].float().to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(images).squeeze(1)

            loss = criterion(
                output,
                target
            )

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

        train_loss = (
            running_loss /
            len(train_loader.dataset)
        )

        (
            val_loss,
            val_mae,
            val_rmse,
            val_spearman,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_MAE={val_mae:.3f} | "
            f"val_RMSE={val_rmse:.3f} | "
            f"val_Spearman={val_spearman:.4f}"
        )

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/mae": val_mae,
            "val/rmse": val_rmse,
            "val/spearman": val_spearman,
            "learning_rate": current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0

            torch.save({
                "model_name":
                    cfg["model_name"],

                "model_state_dict":
                    model.state_dict(),

                "val_loss":
                    val_loss,

                "epoch":
                    epoch,

                "config":
                    cfg,
            }, checkpoint_path)

            print(
                "  儲存最佳權重：",
                checkpoint_path
            )

        else:
            no_improvement += 1

            print(
                f"  Early stopping counter: "
                f"{no_improvement}/{patience}"
            )

        if no_improvement >= patience:
            print(
                f"Early stopping at epoch "
                f"{epoch}"
            )
            break

    print("=" * 60)
    print("Training finished")
    print("Best checkpoint:", checkpoint_path)
    print("Best val loss:", best_val_loss)
    print("=" * 60)

    wandb.finish()


if __name__ == "__main__":
    main()
