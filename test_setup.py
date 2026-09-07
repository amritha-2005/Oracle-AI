from __future__ import annotations

import importlib

MODULES = [
    "app",
    "app.config",
    "app.database",
    "app.main",
    "app.schemas",
    "app.services.ml_service",
    "app.routers.prediction",
    "app.routers.districts",
    "app.routers.analytics",
    "app.routers.explainability",
]


def test_imports():
    for module_name in MODULES:
        importlib.import_module(module_name)
    assert True
