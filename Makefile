#
# See `make help` for a list of all available commands.
#

APP_NAME := tram
VENV := .venv
BIN := $(VENV)/bin
PY_VERSION := python3
TIMESTAMP := $(shell date -u +"%Y%m%d_%H%M%S")
GIT_HASH := $(shell git rev-parse --short HEAD)

.DEFAULT_GOAL := help

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# the aliasing for the venv target is done primarily for readability
$(VENV): $(BIN)/activate

venv: $(VENV)  ## build python venv

$(BIN)/activate:
	$(PY_VERSION) -m venv --prompt $(APP_NAME) $(VENV)

.PHONY: install
install: venv upgrade-pip ## Install Python dependencies
	./$(BIN)/python -m pip install -r requirements/requirements.txt

.PHONY: install-dev
install-dev: venv upgrade-pip  ## Install Python dependencies and dev dependencies
	./$(BIN)/python -m pip install -r requirements/requirements.txt
	./$(BIN)/python -m pip install -r requirements/test-requirements.txt

.PHONY: upgrade-pip
upgrade-pip: venv  ## Upgrade pip and related
	./$(BIN)/python -m pip install --upgrade pip wheel setuptools pip-tools

.git/hooks/pre-commit: install-dev
	./$(BIN)/pre-commit install

.PHONY: pre-commit-run
pre-commit-run: venv .git/hooks/pre-commit ## Run pre-commit manually on changed files
	./$(BIN)/pre-commit run

.PHONY: pre-commit-run-all
pre-commit-run-all: venv .git/hooks/pre-commit ## Run pre-commit manually on all files
	./$(BIN)/pre-commit run -a

.PHONY: build-container
build-container: venv ## Build container image
	docker build -t $(APP_NAME):dev -t $(APP_NAME):$(TIMESTAMP)_$(GIT_HASH) -f Dockerfile . --label "org.opencontainers.image.revision=$(GIT_HASH)"

.PHONY: start-container
start-container: ## Start container via docker-compose, runs ctidorg/tram:latest image by default
	docker-compose -f docker/docker-compose.yml up -d
	@echo "Waiting for container to start..."
	@echo "To stop the container, run: make stop-container, or, docker-compose -f docker/docker-compose.yml down"
	@echo "To view container logs, run make container-logs, or, docker-compose -f docker/docker-compose.yml logs -f"

.PHONY: stop-container
stop-container: ## Stop container via docker-compose
	docker-compose -f docker/docker-compose.yml down

.PHONY: container-logs
container-logs: ## Print container logs
	docker-compose -f docker/docker-compose.yml logs -f

.PHONY: clean
clean: ## Clean up pycache files
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -delete

.PHONY: clean-all
clean-all: clean ## Clean up venv and tox if necessary, in addition to standard clean
	find . \( -name ".tox" -o -name "$(VENV)" -o -name "*.egg-info"  \) -type d -prune -exec rm -rf {} +
	find ./src -name '*.egg' -delete
	rm -f coverage.xml

.PHONY: venv-activate
venv-activate: venv ## Activate venv
	@echo "Activate your virtualenv by running the following command: "
	@echo "source $(BIN)/activate"

.PHONY: venv-deactivate
venv-deactivate: ## Deactivate venv
	@echo "Activate your virtualenv by running the following command: "
	@echo "source deactivate"

.PHONY: lint
lint: pre-commit-run ## Lint code

.PHONY: test
test: venv install-dev ## Run tests
	./$(BIN)/python -m tox
