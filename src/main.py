"""
Main FastAPI application entrypoint.
Mounts static assets, initializes the SQLite database, loads ML pipelines,
and connects page and API routers.
"""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.api_routes import router as api_router
from src.api.page_routes import router as page_router
from src.config import BASE_DIR, settings
from src.database.session import init_db
from src.ml.predict import predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database schema and ensures ML predictor is loaded on startup."""
    logger.info("Initializing SQLite database tables...")
    init_db()

    logger.info("Verifying ML model artifact status...")
    predictor.load_models()
    if predictor.is_ready:
        logger.info("Active ML Regressor pipeline loaded successfully.")
    else:
        logger.warning(
            "Model artifacts not found or not yet trained. Run 'python -m src.ml.train' to build them."
        )

    yield
    logger.info("Application shutdown completed.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="An end-to-end Student Performance Prediction and Academic Risk Analysis system using Python, scikit-learn, FastAPI, HTML, CSS, Bootstrap and SQLite.",
    lifespan=lifespan,
)

# Static file mounting
static_dir = BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include Routers
app.include_router(page_router)
app.include_router(api_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Centralized exception handling for clean error messaging without leaks."""
    logger.error("Unhandled exception processing request to %s: %s", request.url.path, exc, exc_info=True)
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred while processing the request."},
        )
    # Let standard HTML rendering proceed for page routes
    raise exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
