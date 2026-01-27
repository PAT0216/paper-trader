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

# 📈 Run Trading with Momentum Strategy
.PHONY: trade-momentum
trade-momentum:
	@echo "$(YELLOW)📈 Running Momentum Trading Bot (Fama-French 12-1)...$(NC)"
	conda run -n $(ENV_NAME) python main.py --mode trade --strategy momentum
	@echo "$(GREEN)✅ Momentum trading execution complete$(NC)"

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
	conda run -n $(ENV_NAME) python scripts/backtests/run_backtest.py
	@echo "$(GREEN)✅ Backtest complete - see results/$(NC)"

# 📈 Quick Backtest (1 year)
.PHONY: backtest-quick
backtest-quick:
	@echo "$(YELLOW)📈 Running Quick Backtest (2023-2024)...$(NC)"
	conda run -n $(ENV_NAME) python scripts/backtests/run_backtest.py --start 2023-01-01 --end 2024-12-31
	@echo "$(GREEN)✅ Quick backtest complete$(NC)"

# 🔬 Walk-Forward Backtest
.PHONY: backtest-walkforward
backtest-walkforward:
	@echo "$(YELLOW)🔬 Running Walk-Forward Validation (yearly retraining)...$(NC)"
	conda run -n $(ENV_NAME) python scripts/backtests/run_walkforward.py
	@echo "$(GREEN)✅ Walk-forward backtest complete$(NC)"

# 🎯 Phase 7: Risk Control Testing
.PHONY: test-stoploss
test-stoploss:
	@echo "$(YELLOW)🎯 Running Stop-Loss Threshold Sweep...$(NC)"
	conda run -n $(ENV_NAME) python run_stoploss_test.py
	@echo "$(GREEN)✅ Stop-loss test complete$(NC)"

.PHONY: test-ab
test-ab:
	@echo "$(YELLOW)🎯 Running A/B Test (Fixed vs Z-Score signals)...$(NC)"
	conda run -n $(ENV_NAME) python run_ab_test.py
	@echo "$(GREEN)✅ A/B test complete$(NC)"

.PHONY: test-unbiased
test-unbiased:
	@echo "$(YELLOW)🎯 Running Unbiased A/B Test (walk-forward, 75 random stocks)...$(NC)"
	conda run -n $(ENV_NAME) python run_unbiased_comparison.py
	@echo "$(GREEN)✅ Unbiased comparison complete$(NC)"

.PHONY: test-holdout
test-holdout:
	@echo "$(YELLOW)🎯 Running Double Holdout Test (unseen tickers)...$(NC)"
	@echo "   This tests the model on tickers it has NEVER seen before"
	conda run -n $(ENV_NAME) python run_double_holdout.py
	@echo "$(GREEN)✅ Double holdout test complete$(NC)"

# 🇮🇳 India Market Testing
.PHONY: india-fetch
india-fetch:
	@echo "$(YELLOW)🇮🇳 Fetching Indian market data (NIFTY 50)...$(NC)"
	conda run -n $(ENV_NAME) python india/fetch_india_data.py
	@echo "$(GREEN)✅ India data cached$(NC)"

.PHONY: india-test
india-test:
	@echo "$(YELLOW)🇮🇳 Running India Market Test (cross-market validation)...$(NC)"
	@if [ ! -f india/data/india_cache.db ]; then \
		echo "$(YELLOW)No cache found. Run 'make india-fetch' first or download from GitHub Actions$(NC)"; \
		exit 1; \
	fi
	conda run -n $(ENV_NAME) python india/run_india_test.py
	@echo "$(GREEN)✅ India market test complete$(NC)"

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
	@if [ -d data/ledgers ]; then \
		for ledger in data/ledgers/ledger_*.csv; do \
			echo "\n=== $$ledger ==="; \
			tail -n 5 "$$ledger" | column -t -s ','; \
		done; \
	else \
		echo "No ledgers found. Run 'make trade' first."; \
	fi

# ℹ️ Help
.PHONY: help
help:
	@echo "$(GREEN)Paper Trader AI - Makefile Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup & Installation:$(NC)"
	@echo "  make setup               - Create Conda environment and install dependencies"
	@echo "  make install             - Install dependencies with pip (no Conda)"
	@echo ""
	@echo "$(YELLOW)Execution:$(NC)"
	@echo "  make trade               - Run trading bot (fetch data, train, execute trades)"
	@echo "  make trade-momentum      - Run momentum strategy specifically"
	@echo "  make train               - Train model only (no trade execution)"
	@echo "  make validate            - Validate data quality for sample tickers"
	@echo ""
	@echo "$(YELLOW)Backtesting:$(NC)"
	@echo "  make backtest            - Full backtest (2017-2024)"
	@echo "  make backtest-quick      - Quick backtest (2023-2024)"
	@echo "  make backtest-walkforward - Walk-forward validation (yearly retraining)"
	@echo ""
	@echo "$(YELLOW)Phase 7: Risk Control Validation:$(NC)"
	@echo "  make test-stoploss       - Test different stop-loss thresholds"
	@echo "  make test-ab             - A/B test (Fixed vs Z-Score signals)"
	@echo "  make test-unbiased       - Unbiased A/B test (75 random stocks, walk-forward)"
	@echo "  make test-holdout        - Double holdout test (truly unseen tickers)"
	@echo ""
	@echo "$(YELLOW)India Market Testing:$(NC)"
	@echo "  make india-fetch         - Fetch NIFTY 50 data to local cache"
	@echo "  make india-test          - Test model on Indian market (cross-market validation)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test                - Run unit test suite (75 tests)"
	@echo "  make test-coverage       - Run tests with HTML coverage report"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  make docker-up           - Start Docker containers"
	@echo "  make docker-down         - Stop Docker containers"
	@echo ""
	@echo "$(YELLOW)Monitoring:$(NC)"
	@echo "  make status              - Show recent portfolio transactions"
	@echo "  make results             - Display model performance metrics"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(NC)"
	@echo "  make clean               - Remove build artifacts and cache"
	@echo "  make clean-all           - Remove all artifacts, results, and models"
	@echo ""

.DEFAULT_GOAL := help
