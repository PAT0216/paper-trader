# Makefile for AI Paper Trader

# 🐍 Conda Env Name
ENV_NAME = paper-trader

# 🛠 Setup
.PHONY: setup
setup:
	@echo "Creating Conda Environment..."
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Done!"

# 🧠 Train
.PHONY: train
train:
	@echo "Training Model..."
	conda run -n $(ENV_NAME) python main.py --mode train

# 🔮 Trade
.PHONY: trade
trade:
	@echo "Running Trading Bot..."
	conda run -n $(ENV_NAME) python main.py --mode trade

# 🐳 Docker
.PHONY: docker-up
docker-up:
	docker compose up --build

.PHONY: docker-down
docker-down:
	docker compose down

# 🧹 Clean
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf results/*
	@echo "Cleaned."
