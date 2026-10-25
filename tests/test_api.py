"""
Integration and endpoint tests for FastAPI backend and Jinja2 frontend routes.
"""
import io
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Verifies GET /health returns application status and subsystem readiness."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "database_connected" in data
    assert data["database_connected"] is True


def test_analytics_overview(client: TestClient):
    """Verifies GET /api/analytics/overview returns aggregate statistics."""
    res = client.get("/api/analytics/overview")
    assert res.status_code == 200
    data = res.json()
    assert "total_students" in data
    assert data["total_students"] > 0
    assert "average_grade" in data
    assert "passing_rate" in data


def test_analytics_distributions(client: TestClient):
    """Verifies GET /api/analytics/distributions returns real chart payload."""
    res = client.get("/api/analytics/distributions")
    assert res.status_code == 200
    data = res.json()
    assert "grades" in data
    assert len(data["grades"]["labels"]) == 21  # 0 to 20
    assert "study_time" in data
    assert "absences" in data
    assert "risk_breakdown" in data


def test_model_info_and_metrics(client: TestClient):
    """Verifies GET /api/model/info and GET /api/metrics endpoints."""
    res_info = client.get("/api/model/info")
    assert res_info.status_code == 200
    info_data = res_info.json()
    assert info_data["target_variable"] == "G3"
    assert "G1" in info_data["excluded_features"]

    res_metrics = client.get("/api/metrics")
    assert res_metrics.status_code == 200
    metrics_data = res_metrics.json()
    assert "regression" in metrics_data
    assert "classification" in metrics_data


def test_predict_single_valid(client: TestClient, sample_valid_student):
    """Tests POST /api/predict with valid payload returns score and risk."""
    res = client.post("/api/predict", json=sample_valid_student)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_score" in data
    assert 0.0 <= data["predicted_score"] <= 20.0
    assert "risk_category" in data
    assert len(data["recommendations"]) > 0


def test_predict_single_invalid_bounds(client: TestClient, sample_valid_student):
    """Tests POST /api/predict fails with 422 when input violates Pydantic constraints."""
    bad_payload = dict(sample_valid_student)
    bad_payload["age"] = 99  # Allowed max is 22
    res = client.post("/api/predict", json=bad_payload)
    assert res.status_code == 422


def test_predict_batch_json(client: TestClient, sample_valid_student):
    """Tests POST /api/predict/batch with multiple student records."""
    batch_req = {"students": [sample_valid_student, sample_valid_student]}
    res = client.post("/api/predict/batch", json=batch_req)
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["total_records"] == 2
    assert data["summary"]["processed_successfully"] == 2
    assert len(data["results"]) == 2


def test_dataset_validate_endpoint_valid_csv(client: TestClient, sample_synthetic_dataframe):
    """Tests POST /api/dataset/validate with uploaded CSV file."""
    csv_bytes = sample_synthetic_dataframe.to_csv(index=False).encode("utf-8")
    files = {"file": ("students.csv", io.BytesIO(csv_bytes), "text/csv")}
    res = client.post("/api/dataset/validate", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    assert data["total_rows"] == 5


def test_dataset_validate_endpoint_bad_file_type(client: TestClient):
    """Tests that non-CSV files are rejected with 400 Bad Request."""
    files = {"file": ("report.pdf", io.BytesIO(b"%PDF-1.4..."), "application/pdf")}
    res = client.post("/api/dataset/validate", files=files)
    assert res.status_code == 400
    assert "only csv" in res.json()["detail"].lower()


def test_dataset_predict_download_csv(client: TestClient, sample_synthetic_dataframe):
    """Tests POST /api/dataset/predict returns downloadable enriched CSV stream."""
    csv_bytes = sample_synthetic_dataframe.to_csv(index=False).encode("utf-8")
    files = {"file": ("students.csv", io.BytesIO(csv_bytes), "text/csv")}
    res = client.post("/api/dataset/predict", files=files)
    assert res.status_code == 200
    assert "attachment" in res.headers.get("content-disposition", "")
    content = res.text
    assert "predicted_G3" in content
    assert "academic_risk_category" in content


def test_predictions_history_and_clear(client: TestClient, sample_valid_student):
    """Tests GET and DELETE /api/predictions for audit logs."""
    # 1. Trigger single prediction to guarantee at least one audit record
    client.post("/api/predict", json=sample_valid_student)

    # 2. Query history
    res = client.get("/api/predictions?page=1&page_size=10")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 1
    assert len(data["items"]) >= 1

    # 3. Clear history
    del_res = client.delete("/api/predictions")
    assert del_res.status_code == 200
    assert "deleted_count" in del_res.json()

    # 4. Verify empty after deletion
    after_res = client.get("/api/predictions?page=1&page_size=10")
    assert after_res.json()["total"] == 0


@pytest.mark.parametrize("route", [
    "/",
    "/dataset",
    "/predict",
    "/batch-predict",
    "/model-evaluation",
    "/history",
])
def test_frontend_pages_smoke(client: TestClient, route: str):
    """Verifies that all Jinja2 HTML dashboard pages render successfully with 200 OK."""
    res = client.get(route)
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "EduAnalytics" in res.text
