# Dataset Directory

This directory stores datasets used by the Student Performance Prediction and Academic Risk Analysis system.

## Data Privacy & Licensing
- The raw dataset is sourced from the public [UCI Machine Learning Repository: Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez and Silva, 2008).
- Under academic and ethical guidelines, private student personal records are **never** stored or committed.
- The dataset contains 395 secondary school students in mathematics (`student-mat.csv`), including social, demographic, and academic indicators.

## Subdirectories
- `data/raw/`: Contains original, unmodified CSV files (e.g., `student-mat.csv`).
- `data/processed/`: Contains preprocessed or split datasets if exported.

## How to Acquire Data
Run the automated reproducible acquisition script:
```bash
python scripts/download_dataset.py
```
This automatically fetches, verifies, and extracts `student-mat.csv` into `data/raw/`.
