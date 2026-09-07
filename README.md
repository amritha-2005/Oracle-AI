# Oracle AI Backend

This folder contains the FastAPI backend for Oracle AI.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the app with `uvicorn app.main:app --reload`.
4. Open the docs at `http://localhost:8000/docs`.

## Included modules

- app/main.py: application entry point
- app/config.py: environment and configuration
- app/database.py: database connection setup
- app/models/: SQLAlchemy models
- app/schemas/: Pydantic schemas
- app/routers/: API endpoints
- app/services/: prediction logic
- ml_pipeline/: training and data processing scaffolding

## Notes

The initial scaffold is intentionally lightweight so the project can be expanded as the actual climate datasets and trained ML models are added.
