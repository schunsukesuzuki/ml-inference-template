.PHONY: help build up down logs test clean

# デフォルトターゲット
help:
	@echo "ML Inference Template - Available commands:"
	@echo ""
	@echo "  make build          - Build all Docker images"
	@echo "  make build-gpu      - Build Docker images for GPU"
	@echo "  make up             - Start all services (CPU)"
	@echo "  make up-gpu         - Start all services (GPU)"
	@echo "  make down           - Stop all services"
	@echo "  make logs           - Show logs from all services"
	@echo "  make logs-gateway   - Show gateway logs"
	@echo "  make logs-workers   - Show worker logs"
	@echo "  make test           - Test the API"
	@echo "  make scale-workers  - Scale workers (usage: make scale-workers N=5)"
	@echo "  make clean          - Remove all containers and images"
	@echo "  make restart        - Restart all services"
	@echo ""

# Docker Compose ファイルの選択
COMPOSE_FILE = docker-compose.yml
COMPOSE_FILE_GPU = docker-compose.gpu.yml

# イメージのビルド
build:
	docker-compose -f $(COMPOSE_FILE) build

build-gpu:
	docker-compose -f $(COMPOSE_FILE_GPU) build

# サービスの起動
up:
	docker-compose -f $(COMPOSE_FILE) up -d
	@echo "Services started! Gateway available at http://localhost:8080"
	@echo "API docs available at http://localhost:8080/docs"

up-gpu:
	docker-compose -f $(COMPOSE_FILE_GPU) up -d
	@echo "Services started with GPU! Gateway available at http://localhost:8080"
	@echo "API docs available at http://localhost:8080/docs"

# サービスの停止
down:
	docker-compose -f $(COMPOSE_FILE) down
	docker-compose -f $(COMPOSE_FILE_GPU) down

# ログの表示
logs:
	docker-compose -f $(COMPOSE_FILE) logs -f

logs-gateway:
	docker-compose -f $(COMPOSE_FILE) logs -f gateway

logs-workers:
	docker-compose -f $(COMPOSE_FILE) logs -f worker1 worker2 worker3

# APIのテスト
test:
	@echo "Testing Gateway health..."
	@curl -s http://localhost:8080/gateway/health | python -m json.tool
	@echo "\nTesting Gateway info..."
	@curl -s http://localhost:8080/gateway/info | python -m json.tool
	@echo "\nTesting Worker health (via gateway)..."
	@curl -s http://localhost:8080/health | python -m json.tool
	@echo "\nTesting prediction..."
	@curl -s -X POST http://localhost:8080/predict \
		-H "Content-Type: application/json" \
		-d '{"input_data": [1.0, 2.0, 3.0, 4.0]}' | python -m json.tool

# Workerのスケーリング
scale-workers:
	@if [ -z "$(N)" ]; then \
		echo "Usage: make scale-workers N=5"; \
	else \
		docker-compose -f $(COMPOSE_FILE) up -d --scale worker1=$(N); \
	fi

# 再起動
restart: down up

# クリーンアップ
clean:
	docker-compose -f $(COMPOSE_FILE) down -v --rmi all
	docker-compose -f $(COMPOSE_FILE_GPU) down -v --rmi all
	@echo "All containers, volumes, and images removed"

# 開発用: Workerのみ再ビルド
rebuild-worker:
	docker-compose -f $(COMPOSE_FILE) build worker1 worker2 worker3
	docker-compose -f $(COMPOSE_FILE) up -d worker1 worker2 worker3

# 開発用: Gatewayのみ再ビルド
rebuild-gateway:
	docker-compose -f $(COMPOSE_FILE) build gateway
	docker-compose -f $(COMPOSE_FILE) up -d gateway

# システム情報の表示
info:
	@echo "=== Docker Compose Status ==="
	docker-compose -f $(COMPOSE_FILE) ps
	@echo "\n=== Gateway Endpoint ==="
	@echo "http://localhost:8080"
	@echo "\n=== API Documentation ==="
	@echo "http://localhost:8080/docs"
	@echo "\n=== Gateway Health ==="
	@curl -s http://localhost:8080/gateway/health | python -m json.tool || echo "Gateway not running"
