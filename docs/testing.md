# Automated Testing Strategy & Verification Report

## 1. Test Architecture & Frameworks
The testing suite is constructed with **pytest** and FastAPI's **TestClient** (Starlette test harness). Tests operate strictly on synthetic fixtures and an in-memory SQLite database (`sqlite:///:memory:`) to guarantee complete test isolation without modifying persistent local audit databases.

```
tests/
├── __init__.py
├── conftest.py          # Synthetic fixtures, in-memory DB override, mock data
├── test_ml.py           # ML validation, ColumnTransformer, metrics, inference
├── test_api.py          # Endpoints, upload limits, CSV download, page rendering
└── test_database.py     # SQLAlchemy models, CRUD, pagination, table clearing
```

---

## 2. Test Suites Summary (29 Automated Tests)

### 2.1 Machine Learning Suite (`tests/test_ml.py`)
- `test_categorize_academic_risk`: Validates Portuguese grade threshold mappings (<10 High, 10-13 Moderate, >=14 Low).
- `test_validate_dataset_valid`: Validates clean synthetic DataFrames pass with 0 errors.
- `test_validate_dataset_empty`: Confirms empty datasets are rejected with descriptive messages.
- `test_validate_dataset_missing_columns`: Confirms missing feature columns are caught and itemized.
- `test_validate_dataset_out_of_bounds`: Verifies that out-of-range numbers and unexpected categorical strings are flagged.
- `test_validate_single_record`: Tests dictionary validator on valid and invalid payloads.
- `test_preprocessing_feature_isolation`: Asserts that `G1`, `G2`, and `G3` are strictly absent from the feature matrix $X$.
- `test_preprocessor_column_transformer`: Confirms ColumnTransformer imputation and OneHotEncoder dimension expansion.
- `test_predictor_single_and_batch`: Validates model inference score clamping and row-level batch rejection.
- `test_missing_model_handling`: Verifies informative `RuntimeError` when model artifacts are absent.

### 2.2 API & Frontend Integration Suite (`tests/test_api.py`)
- `test_health_endpoint`: Tests `/health` system status and database ping.
- `test_analytics_overview`: Tests `/api/analytics/overview` metrics payload.
- `test_analytics_distributions`: Tests `/api/analytics/distributions` data structures.
- `test_model_info_and_metrics`: Verifies `/api/model/info` and `/api/metrics`.
- `test_predict_single_valid`: Tests `POST /api/predict` returns score, risk tier, and recommendations.
- `test_predict_single_invalid_bounds`: Tests `POST /api/predict` returns HTTP 422 on schema violation.
- `test_predict_batch_json`: Tests `POST /api/predict/batch` for multiple student records.
- `test_dataset_validate_endpoint_valid_csv`: Tests `POST /api/dataset/validate` on CSV files.
- `test_dataset_validate_endpoint_bad_file_type`: Tests 400 Bad Request on non-CSV uploads.
- `test_dataset_predict_download_csv`: Tests `POST /api/dataset/predict` streaming response.
- `test_predictions_history_and_clear`: Tests audit log creation, retrieval, and deletion.
- `test_frontend_pages_smoke`: Parametrized test asserting 200 OK across all 6 Jinja2 HTML routes (`/`, `/dataset`, `/predict`, `/batch-predict`, `/model-evaluation`, `/history`).

### 2.3 Database Persistence Suite (`tests/test_database.py`)
- `test_prediction_log_crud`: Verifies ORM creation, query, and dictionary serialization.
- `test_prediction_pagination`: Tests pagination offset, page size, total pages, and record deletion.

---

## 3. How to Execute Tests

Run the full automated test suite with verbose output:
```bash
python -m pytest
```

Run a specific test module:
```bash
python -m pytest tests/test_ml.py -v
python -m pytest tests/test_api.py -v
python -m pytest tests/test_database.py -v
```

View test execution with summary statistics:
```bash
python -m pytest --tb=short -q
```
