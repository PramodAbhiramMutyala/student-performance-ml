"""
HTML page rendering routes using Jinja2 templates.
"""
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.config import BASE_DIR, CATEGORICAL_VALUE_DOMAINS, NUMERICAL_VALUE_RANGES
from src.ml.data_loader import load_dataset
from src.services.analytics_service import (
    get_analytics_overview,
    get_saved_model_metadata,
    get_saved_model_metrics,
)

router = APIRouter(include_in_schema=False)
templates_dir = BASE_DIR / "src" / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/", response_class=HTMLResponse)
def page_dashboard(request: Request):
    """Renders the executive dashboard page."""
    overview = get_analytics_overview()
    metrics = get_saved_model_metrics()
    meta = get_saved_model_metadata()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "active_page": "dashboard",
            "overview": overview,
            "metrics": metrics,
            "meta": meta,
        },
    )


@router.get("/dataset", response_class=HTMLResponse)
def page_dataset(request: Request):
    """Renders the dataset exploration, schema inspection, and validator page."""
    overview = get_analytics_overview()
    try:
        df = load_dataset()
        dataset_sample = df.head(50).to_dict(orient="records")
        columns = list(df.columns)
    except Exception:
        dataset_sample = []
        columns = []

    return templates.TemplateResponse(
        request=request,
        name="dataset.html",
        context={
            "active_page": "dataset",
            "overview": overview,
            "dataset_sample": dataset_sample,
            "columns": columns,
            "categorical_domains": CATEGORICAL_VALUE_DOMAINS,
            "numerical_ranges": NUMERICAL_VALUE_RANGES,
        },
    )


@router.get("/predict", response_class=HTMLResponse)
def page_predict(request: Request):
    """Renders single student performance prediction form."""
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "active_page": "predict",
            "categorical_domains": CATEGORICAL_VALUE_DOMAINS,
            "numerical_ranges": NUMERICAL_VALUE_RANGES,
        },
    )


@router.get("/batch-predict", response_class=HTMLResponse)
def page_batch_predict(request: Request):
    """Renders batch CSV prediction and download page."""
    return templates.TemplateResponse(
        request=request,
        name="batch_predict.html",
        context={
            "active_page": "batch_predict",
        },
    )


@router.get("/model-evaluation", response_class=HTMLResponse)
def page_model_evaluation(request: Request):
    """Renders model comparison, metrics, confusion matrix, and feature importances."""
    metrics = get_saved_model_metrics()
    meta = get_saved_model_metadata()

    return templates.TemplateResponse(
        request=request,
        name="model_evaluation.html",
        context={
            "active_page": "model_evaluation",
            "metrics": metrics,
            "meta": meta,
        },
    )


@router.get("/history", response_class=HTMLResponse)
def page_history(request: Request):
    """Renders prediction audit history table."""
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "active_page": "history",
        },
    )
