# REST API Documentation

The backend exposes a comprehensive RESTful API built on FastAPI. Interactive Swagger UI documentation is available at `http://127.0.0.1:8000/docs` and ReDoc at `http://127.0.0.1:8000/redoc`.

---

## 1. System & Health

### `GET /health`
Returns the operational health and readiness of server subsystems.

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classifier_loaded": true,
  "database_connected": true,
  "app_version": "1.0.0",
  "timestamp": "2026-10-25T14:35:23.620000+00:00"
}
```

---

## 2. Model Metadata & Empirical Metrics

### `GET /api/model/info`
Returns active model descriptors, target definition, and feature isolation configuration.

**Response `200 OK`:**
```json
{
  "model_version": "1.0.0",
  "experiment_type": "Early-Warning Academic Performance Prediction",
  "target_variable": "G3",
  "excluded_features": ["G1", "G2", "G3"],
  "selected_regressor": "RandomForestRegressor",
  "selected_classifier": "RandomForestClassifier",
  "metrics": { ... }
}
```

### `GET /api/metrics`
Returns actual empirical test metrics, candidate comparison tables, and top feature importances saved in `metrics.json`.

---

## 3. Analytics & Distributions

### `GET /api/analytics/overview`
Provides high-level dataset statistics, passing rates, and active model error benchmarks.

**Response `200 OK`:**
```json
{
  "dataset_available": true,
  "total_students": 395,
  "average_grade": 10.42,
  "passing_rate": 67.09,
  "failing_rate": 32.91,
  "average_absences": 5.71,
  "average_studytime": 2.04,
  "model_metrics": { ... }
}
```

### `GET /api/analytics/distributions`
Returns aggregated grade histograms (0 to 20), study-time correlation groups, and absence impact intervals for Chart.js rendering.

---

## 4. Inference & Predictions

### `POST /api/predict`
Executes single student grade estimation and risk tier classification.

**Request Body (JSON):**
```json
{
  "school": "GP",
  "sex": "F",
  "age": 17,
  "address": "U",
  "famsize": "GT3",
  "Pstatus": "T",
  "Medu": 3,
  "Fedu": 3,
  "Mjob": "services",
  "Fjob": "other",
  "reason": "course",
  "guardian": "mother",
  "traveltime": 1,
  "studytime": 3,
  "failures": 0,
  "schoolsup": "no",
  "famsup": "yes",
  "paid": "no",
  "activities": "yes",
  "nursery": "yes",
  "higher": "yes",
  "internet": "yes",
  "romantic": "no",
  "famrel": 4,
  "freetime": 3,
  "goout": 2,
  "Dalc": 1,
  "Walc": 1,
  "health": 5,
  "absences": 2
}
```

**Response `200 OK`:**
```json
{
  "predicted_score": 11.85,
  "raw_score": 11.8472,
  "risk_category": "Moderate Academic Risk",
  "risk_probabilities": {
    "High Academic Risk": 0.28,
    "Moderate Academic Risk": 0.54,
    "Low Academic Risk": 0.18
  },
  "scale": "0 to 20 Portuguese secondary mathematics scale",
  "passing_threshold": 10.0,
  "recommendations": [
    "Targeted study schedule with peer tutoring in difficult modules.",
    "Encourage reduction of excessive weekday social and screen time.",
    "Bi-weekly progress checks on homework and quiz scores."
  ],
  "disclaimer": "Academic predictions are statistical estimations intended solely for proactive early counseling support. They must never be used for punitive or high-stakes decisions."
}
```

**Error Codes:**
- `422 Unprocessable Entity`: Input violates type or range constraints.
- `503 Service Unavailable`: Model pipelines not found on disk.

### `POST /api/predict/batch`
Accepts a JSON array of student objects (up to 5,000 items) and returns row-by-row prediction and error diagnostics.

---

## 5. File Uploads & Batch Inference

### `POST /api/dataset/validate`
Accepts an uploaded multipart CSV file, inspects columns and values against schema rules, and returns diagnostics without running prediction.

**Parameters:**
- `file`: Multipart form-data `.csv` file (max 5 MB).

### `POST /api/dataset/predict`
Accepts an uploaded CSV dataset, runs row-level validation and inference, and streams back an enriched CSV file attachment with:
- `predicted_G3`: Estimated grade
- `academic_risk_category`: Risk classification
- `prediction_status`: "success" or "rejected"
- `validation_errors`: Diagnostic explanations for any malformed fields

---

## 6. Audit & History

### `GET /api/predictions`
Returns paginated prediction audit history from the local SQLite database.

**Query Parameters:**
- `page` (default `1`, min `1`)
- `page_size` (default `15`, min `1`, max `100`)

### `DELETE /api/predictions`
Deletes all recorded prediction history from SQLite (requires manual confirmation on frontend).
