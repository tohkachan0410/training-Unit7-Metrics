from pathlib import Path
import shutil

import pandas as pd


ROOT = Path("/home/u5450760/train_6")

METRICS_CSV = ROOT / "outputs/metrics/top100_metrics.csv"
RANKINGS_DIR = ROOT / "outputs/rankings"
REFERENCE_CSV = (
    ROOT
    / "data"
    / "annotations"
    / "ranking_reference_top100.csv"
)

OUTPUT_DIR = ROOT / "outputs/cases"


def resolve_image_path(path_str):
    """
    嘗試找到實際圖片位置。
    """
    p = Path(str(path_str))

    # 1. CSV 本身就是絕對路徑
    if p.is_absolute() and p.exists():
        return p

    # 2. 以 train_6 為根目錄
    candidate = ROOT / p
    if candidate.exists():
        return candidate

    # 3. data/raw/檔名
    candidate = ROOT / "data" / "raw" / p.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"找不到圖片：{path_str}"
    )


def choose_best_method(metrics):
    """
    主要依 AP 最大選最佳 method。
    若 AP 相同，再比較 overlap，
    再比較 jaccard。
    """
    required = {
        "method",
        "intersection",
        "overlap",
        "jaccard",
        "AP",
    }

    missing = required - set(metrics.columns)

    if missing:
        raise ValueError(
            f"top100_metrics.csv 缺少欄位：{sorted(missing)}"
        )

    metrics = metrics.copy()

    metrics = metrics.sort_values(
        by=["AP", "overlap", "jaccard"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return metrics.iloc[0]


def get_ranking_file(method):
    """
    method 名稱和我們 evaluate_top100.py 輸出的檔名一致。
    例如：
    vgg16_raw
    -> outputs/rankings/vgg16_raw_top100.csv
    """
    path = RANKINGS_DIR / f"{method}_top100.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"找不到 ranking 檔案：{path}"
        )

    return path


def prepare_output_dirs():
    dirs = {
        "agreement": OUTPUT_DIR / "agreement",
        "human_only": OUTPUT_DIR / "human_only",
        "model_only": OUTPUT_DIR / "model_only",
    }

    for d in dirs.values():
        if d.exists():
            shutil.rmtree(d)

        d.mkdir(
            parents=True,
            exist_ok=True
        )

    return dirs


def copy_case_images(case_df, outdir, prefix):
    """
    複製前 5 張案例圖。
    """
    selected = case_df.head(5).copy()

    copied_rows = []

    for i, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):
        src = resolve_image_path(
            row["image_path"]
        )

        suffix = src.suffix

        dst = (
            outdir
            / f"{prefix}_{i:02d}_{row['image_id']}{suffix}"
        )

        shutil.copy2(
            src,
            dst
        )

        copied_rows.append({
            "case_type": prefix,
            "case_order": i,
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "copied_path": str(dst),
        })

    return copied_rows


def main():
    print("=" * 70)
    print("UNIT7 CASE ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------
    # 1. 讀 metrics，自動找最佳 method
    # ------------------------------------------------
    metrics = pd.read_csv(
        METRICS_CSV
    )

    best = choose_best_method(
        metrics
    )

    best_method = str(
        best["method"]
    )

    print("\nBest method:")
    print("Method       :", best_method)
    print("Intersection :", best["intersection"])
    print("Overlap      :", best["overlap"])
    print("Jaccard      :", best["jaccard"])
    print("AP           :", best["AP"])

    # ------------------------------------------------
    # 2. 找到該 method 的 Top100
    # ------------------------------------------------
    ranking_file = get_ranking_file(
        best_method
    )

    model_top100 = pd.read_csv(
        ranking_file
    )

    reference = pd.read_csv(
        REFERENCE_CSV
    )

    model_top100["image_id"] = (
        model_top100["image_id"]
        .astype(str)
    )

    reference["image_id"] = (
        reference["image_id"]
        .astype(str)
    )

    # ------------------------------------------------
    # 3. 建立 H / M
    # ------------------------------------------------
    human_ids = set(
        reference["image_id"]
    )

    model_ids = set(
        model_top100["image_id"]
    )

    agreement_ids = (
        human_ids & model_ids
    )

    human_only_ids = (
        human_ids - model_ids
    )

    model_only_ids = (
        model_ids - human_ids
    )

    print("\nCase counts:")
    print(
        "Agreement H ∩ M :",
        len(agreement_ids)
    )
    print(
        "Human only H - M:",
        len(human_only_ids)
    )
    print(
        "Model only M - H:",
        len(model_only_ids)
    )

    # ------------------------------------------------
    # 4. Agreement
    #
    # 保留模型 ranking 順序，
    # 所以取模型排名最高的共同案例。
    # ------------------------------------------------
    agreement = (
        model_top100[
            model_top100["image_id"]
            .isin(agreement_ids)
        ]
        .copy()
    )

    # ------------------------------------------------
    # 5. Human-only
    #
    # Reference 本身沒有模型 rank，
    # 所以沿 reference CSV 順序。
    # ------------------------------------------------
    human_only = (
        reference[
            reference["image_id"]
            .isin(human_only_ids)
        ]
        .copy()
    )

    # ------------------------------------------------
    # 6. Model-only
    #
    # 取模型 ranking 最高但人工沒選的案例。
    # ------------------------------------------------
    model_only = (
        model_top100[
            model_top100["image_id"]
            .isin(model_only_ids)
        ]
        .copy()
    )

    # ------------------------------------------------
    # 7. 確保有 image_path
    # ------------------------------------------------
    for name, df in [
        ("agreement", agreement),
        ("human_only", human_only),
        ("model_only", model_only),
    ]:
        if "image_path" not in df.columns:
            raise ValueError(
                f"{name} 缺少 image_path"
            )

    # ------------------------------------------------
    # 8. 建立輸出資料夾
    # ------------------------------------------------
    dirs = prepare_output_dirs()

    # ------------------------------------------------
    # 9. 複製每組前 5 張
    # ------------------------------------------------
    summary_rows = []

    summary_rows += copy_case_images(
        agreement,
        dirs["agreement"],
        "agreement",
    )

    summary_rows += copy_case_images(
        human_only,
        dirs["human_only"],
        "human_only",
    )

    summary_rows += copy_case_images(
        model_only,
        dirs["model_only"],
        "model_only",
    )

    # ------------------------------------------------
    # 10. 儲存完整三組 CSV
    # ------------------------------------------------
    agreement.to_csv(
        OUTPUT_DIR / "agreement_all.csv",
        index=False
    )

    human_only.to_csv(
        OUTPUT_DIR / "human_only_all.csv",
        index=False
    )

    model_only.to_csv(
        OUTPUT_DIR / "model_only_all.csv",
        index=False
    )

    # ------------------------------------------------
    # 11. 儲存報告使用的 15 張摘要
    # ------------------------------------------------
    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_DIR / "case_summary.csv",
        index=False
    )

    # ------------------------------------------------
    # 12. 寫一個文字摘要
    # ------------------------------------------------
    summary_txt = OUTPUT_DIR / "case_analysis_summary.txt"

    with open(
        summary_txt,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "Training Unit7 Case Analysis\n"
        )
        f.write(
            "============================\n\n"
        )

        f.write(
            f"Best method: {best_method}\n"
        )
        f.write(
            f"Intersection: {best['intersection']}\n"
        )
        f.write(
            f"Overlap: {best['overlap']}\n"
        )
        f.write(
            f"Jaccard: {best['jaccard']}\n"
        )
        f.write(
            f"AP: {best['AP']}\n\n"
        )

        f.write(
            f"Agreement count: {len(agreement)}\n"
        )
        f.write(
            f"Human-only count: {len(human_only)}\n"
        )
        f.write(
            f"Model-only count: {len(model_only)}\n"
        )

    print("\nGenerated:")
    print(
        OUTPUT_DIR / "agreement"
    )
    print(
        OUTPUT_DIR / "human_only"
    )
    print(
        OUTPUT_DIR / "model_only"
    )
    print(
        OUTPUT_DIR / "case_summary.csv"
    )
    print(
        OUTPUT_DIR / "case_analysis_summary.txt"
    )

    print("\nCASE ANALYSIS: PASS")


if __name__ == "__main__":
    main()
