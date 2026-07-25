.DEFAULT_GOAL := help
.PHONY: help install install-backend install-frontend dev api web test lint \
        typecheck check eval calibrate research doctor docker clean

VENV := backend/.venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: install-backend install-frontend ## Install everything

install-backend: ## Create the venv and install the backend
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e "backend[dev]"
	@echo "Optional model SDKs:  $(PIP) install -e 'backend[openai,anthropic]'"

install-frontend: ## Install frontend dependencies
	cd frontend && npm install

api: ## Run the API on :8000 with reload
	cd backend && .venv/bin/veritas serve --reload

web: ## Run the frontend on :3000
	cd frontend && npm run dev

dev: ## Run API and frontend together
	@$(MAKE) -j2 api web

test: ## Run the backend test suite
	cd backend && .venv/bin/python -m pytest -q

lint: ## Lint the backend
	cd backend && .venv/bin/ruff check veritas tests

typecheck: ## Type-check both sides
	cd backend && .venv/bin/mypy veritas --ignore-missing-imports || true
	cd frontend && npx tsc --noEmit

check: lint typecheck test ## Lint, type-check and test

doctor: ## Check configuration and connectivity
	cd backend && .venv/bin/veritas doctor

research: ## Run research on TOPIC="..."
	@test -n "$(TOPIC)" || (echo "usage: make research TOPIC=\"your topic\"" && exit 1)
	cd backend && .venv/bin/veritas research "$(TOPIC)"

eval: ## Benchmark against the single-LLM baseline
	cd backend && .venv/bin/veritas eval --limit $(or $(N),24)

calibrate: ## Fit the confidence calibrator from the last eval
	cd backend && .venv/bin/veritas calibrate

docker: ## Build and start the full stack
	docker compose up --build

clean: ## Remove build artefacts and local databases
	rm -rf backend/.venv frontend/node_modules frontend/.next
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name "*.db" -o -name "*.db-wal" -o -name "*.db-shm" | xargs rm -f 2>/dev/null || true
