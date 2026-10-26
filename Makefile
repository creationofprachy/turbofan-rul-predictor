.PHONY: help setup install test train run clean

PYTHON ?= python3.12
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTHON_BIN := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest

help:
	@echo "Turbofan Engine RUL Predictor"
	@echo ""
	@echo "Targets:"
	@echo "  setup   - Create virtualenv and install dependencies"
	@echo "  install - Install dependencies into existing venv"
	@echo "  test    - Run the test suite"
	@echo "  train   - Retrain models and regenerate plots"
	@echo "  run     - Start the FastAPI dev server on port 8000"
	@echo "  clean   - Remove caches and build artifacts"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) tests/ -v

train:
	$(PYTHON_BIN) -m src.train

run:
	$(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ src/__pycache__ app/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete