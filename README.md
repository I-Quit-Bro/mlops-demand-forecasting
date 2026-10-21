# Demand Forecast: CI/CD with safe promotion and rollback

Daily demand forecasting model (Ridge with lag and calendar features), served with FastAPI,
and built to be trained, gated, promoted and rolled back by a CI/CD pipeline.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```
Run every command from the repo root.

## Data
| File | What it is | In git? |
|------|------------|---------|
| `data/raw/train.csv` | Full Kaggle file (date, store, item, sales). Used locally for training and as the API's history. | No (gitignored) |
| `data/sample/daily_sample.csv` | Last 730 days of daily totals (date, sales), about 15 KB. Generated from the raw file. CI trains on this because CI cannot see your local raw file. | Yes |

Create the sample once (and again if you ever change the raw data):
```bash
forecast-make-sample
```

Fresh machine or teammate? Host the file as `retail_sales.csv.gz` (about 22 MB) in a GitHub Release,
set `data.source_url`, `data.raw_path` (ending `.csv.gz`) and `data.sha256` in `configs/config.yaml`, then:
```bash
forecast-fetch-data            # downloads, verifies the checksum, skips if already present
```

## Train, register, serve
```bash
forecast-train                                   # full data -> artifacts/models/<version>/
forecast-registry list                           # saved versions
forecast-registry set production <version>       # point production at a version
uvicorn forecast.api.main:app --reload           # http://127.0.0.1:8000/docs
curl "http://127.0.0.1:8000/forecast?horizon=7"
```
The API re-reads the `production` pointer on every request, so changing the pointer
(promotion or rollback) takes effect immediately, with no restart.

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | liveness |
| `GET /ready` | 200 only when a production model is loaded |
| `GET /model` | serving version and its hold-out metrics |
| `GET /forecast?horizon=N` | next N days (1 to 30) after the end of the history |

Debugging: `LOG_LEVEL=DEBUG` shows tracebacks; every API request is logged with an `X-Request-ID`.
Training exit codes: `0` ok, `1` expected project error, `2` unexpected error.

## Test and lint
```bash
pytest
ruff check .
```

## Layout
```
configs/config.yaml            paths, hold-out size, model params, serving history
src/forecast/
  config.py  logger.py  exception.py  utils.py
  registry.py                  file-based registry: stage pointers (candidate/production/previous)
  components/                  data_ingestion, data_transformation, model_trainer
  pipeline/                    train_pipeline, predict_pipeline, make_sample
  api/                         service.py (logic), schemas.py, main.py (FastAPI)
tests/                         unit + end-to-end tests on synthetic data
```

## Artifacts
```
artifacts/registry.json                  {"candidate": ..., "production": ..., "previous": ...}
artifacts/models/<IST timestamp>-<git sha>/
  model.joblib   metrics.json   metadata.json
```
Versions are never overwritten. Only load `.joblib` files you produced yourself.
