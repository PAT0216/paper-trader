# Makefile for AI Paper Trader

# 🐍 Conda Env Name
ENV_NAME = paper-trader

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
NC = \033[0m # No Color

# 🛠 Setup
.PHONY: setup
setup:
	@echo "$(YELLOW)Creating Conda Environment...$(NC)"
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "$(GREEN)✅ Environment ready!$(NC)"
	@echo "$(YELLOW)Run 'make trade' to start trading$(NC)"

# 🧠 Train Model
.PHONY: train
train:
	@echo "$(YELLOW)🧠 Training Model...$(NC)"
	conda run -n $(ENV_NAME) python main.py --mode train
	@echo "$(GREEN)✅ Model training complete$(NC)"

# 🔮 Run Trading Bot
.PHONY: trade
trade:
	@echo "$(YELLOW)🔮 Running Trading Bot...$(NC)"
	conda run -n $(ENV_NAME) python main.py --mode trade
	@echo "$(GREEN)✅ Trading execution complete$(NC)"

# 🧪 Run Test Suite
.PHONY: test
test:
	@echo "$(YELLOW)🧪 Running Test Suite...$(NC)"
	conda run -n $(ENV_NAME) pytest tests/ -v
	@echo "$(GREEN)✅ All tests passed$(NC)"

# 📊 Run Tests with Coverage
.PHONY: test-coverage
test-coverage:
	@echo "$(YELLOW)📊 Running Tests with Coverage Report...$(NC)"
	conda run -n $(ENV_NAME) pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✅ Coverage report generated: htmlcov/index.html$(NC)"

# 🐳 Docker Commands
.PHONY: docker-up
docker-up:
	@echo "$(YELLOW)🐳 Starting Docker containers...$(NC)"
	docker compose up --build

.PHONY: docker-down
docker-down:
	@echo "$(YELLOW)🛑 Stopping Docker containers...$(NC)"
	docker compose down
	@echo "$(GREEN)✅ Containers stopped$(NC)"

# 📈 Run Backtest
.PHONY: backtest
backtest:
	@echo "$(YELLOW)📈 Running Backtest (2017-2024)...$(NC)"
	conda run -n $(ENV_NAME) python run_backtest.py
	@echo "$(GREEN)✅ Backtest complete - see results/$(NC)"

# 📈 Quick Backtest (1 year)
.PHONY: backtest-quick
backtest-quick:
	@echo "$(YELLOW)📈 Running Quick Backtest (2023-2024)...$(NC)"
	conda run -n $(ENV_NAME) python run_backtest.py --start 2023-01-01 --end 2024-12-31
	@echo "$(GREEN)✅ Quick backtest complete$(NC)"

# 🔍 Validate Data Only
.PHONY: validate
validate:
	@echo "$(YELLOW)🔍 Validating market data quality...$(NC)"
	conda run -n $(ENV_NAME) python -c "from src.data import loader; from src.data.validator import DataValidator; from src.utils.config import load_config; config = load_config(); data = loader.fetch_data(config['tickers'][:5], '1y'); v = DataValidator(); r = v.validate_data_dict(data); v.print_validation_summary(r)"
	@echo "$(GREEN)✅ Validation complete$(NC)"

# 📈 View Results
.PHONY: results
results:
	@echo "$(YELLOW)📈 Opening results...$(NC)"
	@if [ -f results/confusion_matrix.png ]; then open results/confusion_matrix.png; fi
	@if [ -f results/metrics.txt ]; then cat results/metrics.txt; fi
	@echo "$(GREEN)✅ Results displayed$(NC)"

# 🧹 Clean Build Artifacts
.PHONY: clean
clean:
	@echo "$(YELLOW)🧹 Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

# 🗑 Clean All (including results and models)
.PHONY: clean-all
clean-all: clean
	@echo "$(YELLOW)🗑 Removing results and models...$(NC)"
	rm -rf results/*
	rm -rf models/*.joblib
	@echo "$(GREEN)✅ Full cleanup complete$(NC)"

# 📦 Install Dependencies (without Conda)
.PHONY: install
install:
	@echo "$(YELLOW)📦 Installing dependencies with pip...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

# 📋 Show Portfolio Status
.PHONY: status
status:
	@echo "$(YELLOW)📋 Portfolio Status:$(NC)"
	@if [ -f ledger.csv ]; then \
		tail -n 10 ledger.csv | column -t -s ','; \
	else \
		echo "No ledger found. Run 'make trade' first."; \
	fi

# ℹ️ Help
.PHONY: help
help:
	@echo "$(GREEN)Paper Trader AI - Makefile Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup & Installation:$(NC)"
	@echo "  make setup          - Create Conda environment and install dependencies"
	@echo "  make install        - Install dependencies with pip (no Conda)"
	@echo ""
	@echo "$(YELLOW)Execution:$(NC)"
	@echo "  make trade          - Run trading bot (fetch data, train, execute trades)"
	@echo "  make train          - Train model only (no trade execution)"
	@echo "  make validate       - Validate data quality for sample tickers"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test           - Run unit test suite (27 tests)"
	@echo "  make test-coverage  - Run tests with HTML coverage report"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  make docker-up      - Start Docker containers"
	@echo "  make docker-down    - Stop Docker containers"
	@echo ""
	@echo "$(YELLOW)Monitoring:$(NC)"
	@echo "  make status         - Show recent portfolio transactions"
	@echo "  make results        - Display model performance metrics"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(NC)"
	@echo "  make clean          - Remove build artifacts and cache"
	@echo "  make clean-all      - Remove all artifacts, results, and models"
	@echo ""

.DEFAULT_GOAL := help
