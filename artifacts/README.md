# Artifacts Directory

This directory stores generated exploratory data analysis visualizations and trained machine learning models.

## Structure
- `artifacts/eda/`: Visualizations generated during exploratory data analysis, including grade distributions, feature correlations, study time analysis, and absence distributions.
- `artifacts/models/`: Serialized scikit-learn pipelines (`.joblib`), evaluation metrics (`metrics.json`), and model metadata (`model_metadata.json`).

## Regeneration
To train candidate models, evaluate them on test sets, generate EDA plots, and produce all artifacts, execute:
```bash
python -m src.ml.train
```
Large binary `.joblib` files are excluded by `.gitignore` to keep git repositories clean and lightweight.
