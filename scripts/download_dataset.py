"""
Automated script to download and extract the UCI Student Performance dataset.
Dataset source: UCI Machine Learning Repository (Cortez and Silva, 2008).
https://archive.ics.uci.edu/dataset/320/student+performance
"""
import io
import sys
import zipfile
from pathlib import Path
import urllib.request

UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def download_and_extract_dataset(force: bool = False) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_csv = RAW_DATA_DIR / "student-mat.csv"

    if target_csv.exists() and not force:
        print(f"[INFO] Dataset already present at: {target_csv}")
        return target_csv

    print(f"[INFO] Downloading dataset from: {UCI_URL}")
    req = urllib.request.Request(
        UCI_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StudentPerformanceML/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            outer_zip_bytes = response.read()

        print(f"[INFO] Downloaded {len(outer_zip_bytes):,} bytes. Extracting archives...")
        with zipfile.ZipFile(io.BytesIO(outer_zip_bytes)) as outer_zip:
            if "student.zip" in outer_zip.namelist():
                inner_zip_bytes = outer_zip.read("student.zip")
                with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zip:
                    for filename in ["student-mat.csv", "student-por.csv", "student.txt"]:
                        if filename in inner_zip.namelist():
                            content = inner_zip.read(filename)
                            dest_file = RAW_DATA_DIR / filename
                            dest_file.write_bytes(content)
                            print(f"[SUCCESS] Extracted: {dest_file.name} ({len(content):,} bytes)")
            else:
                # If zip structure is flat
                outer_zip.extractall(RAW_DATA_DIR)
                print(f"[SUCCESS] Extracted files directly to {RAW_DATA_DIR}")

        if not target_csv.exists():
            raise FileNotFoundError(f"Expected {target_csv.name} was not found after extraction.")

        print(f"[SUCCESS] Verified primary dataset file: {target_csv}")
        return target_csv

    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    download_and_extract_dataset()
