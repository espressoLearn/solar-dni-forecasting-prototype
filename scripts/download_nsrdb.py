import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("NREL_API_KEY")
EMAIL = os.getenv("NREL_EMAIL")
LAT = os.getenv("NSRDB_LAT", "40.7982")
LON = os.getenv("NSRDB_LON", "-77.8599")

YEAR_START = int(os.getenv("NSRDB_YEAR_START", "1998"))
YEAR_END = int(os.getenv("NSRDB_YEAR_END", "2024"))

if not API_KEY or not EMAIL:
    raise SystemExit("Missing NREL_API_KEY or NREL_EMAIL in .env")

# Write to a NEW folder so we don't mix formats/columns with your existing files
OUT_DIR = Path("data/raw_plus")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENDPOINT = "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"

BASE_PARAMS = {
    "api_key": API_KEY,
    "email": EMAIL,
    "wkt": f"POINT({LON} {LAT})",
    "interval": "60",
    "utc": "true",
    "leap_day": "true",
    # Added optional attributes:
    "attributes": (
        "dni,ghi,dhi,"
        "air_temperature,wind_speed,relative_humidity,cloud_type,"
        "solar_zenith_angle,clearsky_dni"
    ),
}


def download_year(year: int) -> Path:
    out_path = OUT_DIR / f"nsrdb_plus_{year}_lat{LAT}_lon{LON}.csv"

    # Skip if already downloaded
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return out_path

    params = BASE_PARAMS.copy()
    params["names"] = str(year)

    r = requests.get(ENDPOINT, params=params, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"{year} failed: {r.status_code}\n{r.text[:400]}")

    out_path.write_bytes(r.content)
    if out_path.stat().st_size < 10_000:
        raise RuntimeError(f"{year} downloaded but file too small")

    return out_path


def main():
    print("Downloading NSRDB (with optional attributes) →", OUT_DIR)
    for year in tqdm(range(YEAR_START, YEAR_END + 1), desc="Downloading"):
        try:
            p = download_year(year)
            # print only occasionally to reduce spam
            if year in (YEAR_START, YEAR_END) or year % 5 == 0:
                print("Saved:", p.name)
        except Exception as e:
            print("ERROR:", year, e)


if __name__ == "__main__":
    main()