from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw_plus")
OUT_DIR = Path("data/processed_plus")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# We normalize spaces to underscores, so "Solar Zenith Angle" -> "Solar_Zenith_Angle"
KEEP_COLS = [
    "timestamp",
    "DNI", "GHI", "DHI",
    "Temperature", "Wind_Speed", "Relative_Humidity", "Cloud_Type",
    "Solar_Zenith_Angle", "Clearsky_DNI",
]

def read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    ts = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute"]]
        .rename(columns={"Year":"year","Month":"month","Day":"day","Hour":"hour","Minute":"minute"})
    )
    df.insert(0, "timestamp", ts)

    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep].copy()

    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("timestamp").reset_index(drop=True)

def main() -> None:
    files = sorted(RAW_DIR.glob("nsrdb_plus_*.csv"))
    if not files:
        raise SystemExit(f"No files found in {RAW_DIR}. Did you run the downloader?")

    parts = [read_one(f) for f in files]
    data = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    year = data["timestamp"].dt.year
    train = data[year <= 2022].copy()      # 1998–2022
    val = data[year == 2023].copy()        # 2023
    test = data[year == 2024].copy()       # 2024

    train_path = OUT_DIR / "train.parquet"
    val_path = OUT_DIR / "val.parquet"
    test_path = OUT_DIR / "test.parquet"

    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)

    print("Saved:", train_path, "rows:", len(train), "range:", train["timestamp"].min(), "→", train["timestamp"].max())
    print("Saved:", val_path, "rows:", len(val), "range:", val["timestamp"].min(), "→", val["timestamp"].max())
    print("Saved:", test_path, "rows:", len(test), "range:", test["timestamp"].min(), "→", test["timestamp"].max())

if __name__ == "__main__":
    main()