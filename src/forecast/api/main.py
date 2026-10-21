"""FastAPI application.

Run locally:   uvicorn forecast.api.main:app --reload
Docs (Swagger): http://127.0.0.1:8000/docs

Endpoints
    GET /health               liveness: the process is up
    GET /ready                readiness: 200 only when a production model is loaded
    GET /model                which production model is serving, with its metrics
    GET /forecast?horizon=7   forecast the next N days after the end of the history
"""
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from forecast.api.schemas import MAX_HORIZON, ForecastResponse, ModelInfo
from forecast.api.service import ModelService
from forecast.config import Config, load_config
from forecast.exception import ForecastError, ModelNotFoundError
from forecast.logger import get_logger

log = get_logger(__name__)


def get_service(request: Request) -> ModelService:
    return request.app.state.service


def create_app(config: Config | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = ModelService(config or load_config())
        yield

    app = FastAPI(title="Demand Forecast API", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:8])
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request_id=%s %s %s -> %d in %.1fms",
            request_id, request.method, request.url.path,
            response.status_code, (time.perf_counter() - start) * 1000,
        )
        return response

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
        log.error("Model not available: %s", exc)
        return JSONResponse(status_code=503, content={"detail": exc.message})

    @app.exception_handler(ForecastError)
    async def forecast_error_handler(request: Request, exc: ForecastError):
        # Full context goes to the log; clients only get the short message.
        log.error("Request failed: %s", exc, exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": exc.message})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(service: ModelService = Depends(get_service)):
        if not service.is_ready():
            return JSONResponse(status_code=503, content={"status": "no production model"})
        return {"status": "ready"}

    @app.get("/model", response_model=ModelInfo)
    def model_info(service: ModelService = Depends(get_service)):
        return service.model_info()

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(
        horizon: int = Query(7, ge=1, le=MAX_HORIZON, description="Days to forecast"),
        service: ModelService = Depends(get_service),
    ):
        return service.forecast(horizon)

    return app


app = create_app()
