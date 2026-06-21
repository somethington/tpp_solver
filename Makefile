# TPP Solver -- common developer tasks. Run `make` or `make help` for the list.

PYTHON  ?= python
COMPOSE ?= docker compose
PORT    ?= 8501
APP     := tpp_solver_mt.py

.DEFAULT_GOAL := help

.PHONY: help install run test lint format docker-build docker-run \
        server-compose-build-nocache server-compose-interactive \
        server-compose server-compose-production attach

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

# --- Local development -------------------------------------------------------

install: ## Install the app + dev tools (pytest, ruff) in editable mode
	$(PYTHON) -m pip install -e ".[dev]"

run: ## Run the Streamlit app locally (override with PORT=...)
	$(PYTHON) -m streamlit run $(APP) --server.port $(PORT)

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Lint with ruff
	ruff check .

format: ## Auto-fix lint issues with ruff
	ruff check --fix .

# --- Docker (single container) ----------------------------------------------

docker-build: ## Build the Docker image locally (tppsolver:local)
	docker build -t tppsolver:local .

docker-run: ## Run the Docker image locally (override with PORT=...)
	docker run --rm -p $(PORT):8501 tppsolver:local

# --- Docker Compose (deployment) --------------------------------------------

server-compose-build-nocache: ## Compose build with no cache
	$(COMPOSE) --compatibility build --no-cache

server-compose-interactive: ## Build + up with dev overlay (foreground)
	$(COMPOSE) --compatibility build
	$(COMPOSE) --compatibility -f docker-compose.yml -f docker-compose-dev.yml up

server-compose: ## Build + up with dev overlay (detached)
	$(COMPOSE) --compatibility build
	$(COMPOSE) --compatibility -f docker-compose.yml -f docker-compose-dev.yml up -d

server-compose-production: ## Build + up production config (detached)
	$(COMPOSE) --compatibility build
	$(COMPOSE) --compatibility -f docker-compose.yml up -d

attach: ## Open a shell in the running container
	docker exec -it tppsolver-streamlit /bin/bash
