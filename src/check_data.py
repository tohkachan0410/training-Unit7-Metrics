from pathlib import Path
import pandas as pd


ROOT = Path("/home/u5450760/train_6")

TRAIN_CSV = ROOT / "data/splits/train.csv"
VAL_CSV = ROOT / "data/splits/val.csv"
REF_CSV = ROOT / "data/annotations/ranking_reference_top100.csv"


print("===== Checking CSV paths =====")
print("Train:", TRAIN_CSV)
print("Val:", VAL_CSV)
print("Reference:", REF_CSV)

print()
print("Train exists:", TRAIN_CSV.exists())
print("Val exists:", VAL_CSV.exists())
print("Reference exists:", REF_CSV.exists())


if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"找不到 Train CSV: {TRAIN_CSV}")

if not VAL_CSV.exists():
    raise FileNotFoundError(f"找不到 Val CSV: {VAL_CSV}")

if not REF_CSV.exists():
    raise FileNotFoundError(f"找不到 Reference CSV: {REF_CSV}")


print("\n===== Reading CSV =====")

train = pd.read_csv(TRAIN_CSV)
val = pd.read_csv(VAL_CSV)
ref = pd.read_csv(REF_CSV)


print("Train:", len(train))
print("Val:", len(val))
print("Reference:", len(ref))
print("Total:", len(train) + len(val) + len(ref))


assert len(train) == 320, f"Train 數量錯誤: {len(train)}"
assert len(val) == 80, f"Val 數量錯誤: {len(val)}"
assert len(ref) == 100, f"Reference 數量錯誤: {len(ref)}"


required = {
    "image_id",
    "image_path",
    "quality_score",
}


print("\n===== Checking columns =====")

for name, df in [
    ("train", train),
    ("val", val),
    ("reference", ref),
]:
    missing = required - set(df.columns)

    assert not missing, (
        f"{name} 缺少欄位: {sorted(missing)}"
    )

    assert df["image_id"].is_unique, (
        f"{name} 有重複 image_id"
    )

    assert df["quality_score"].notna().all(), (
        f"{name} 有 NaN quality_score"
    )

    print(f"{name}: columns OK")


print("\n===== Checking split overlap =====")

train_ids = set(train["image_id"].astype(str))
val_ids = set(val["image_id"].astype(str))
ref_ids = set(ref["image_id"].astype(str))

assert not train_ids & val_ids, (
    "Train 和 Val 有重複 image_id"
)

assert not train_ids & ref_ids, (
    "Train 和 Reference 有重複 image_id"
)

assert not val_ids & ref_ids, (
    "Val 和 Reference 有重複 image_id"
)

print("Split overlap: OK")


print("\n===== Checking image files =====")

missing_files = []

for df in [train, val, ref]:
    for p in df["image_path"]:
        path = Path(str(p))

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            missing_files.append(str(path))


print("Missing image files:", len(missing_files))

if missing_files:
    print("\n前 20 個找不到的圖片：")

    for p in missing_files[:20]:
        print(p)

    raise SystemExit("IMAGE CHECK FAILED")


print("\n==============================")
print("DATA CHECK: PASS")
print("==============================")
