start:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml up --build

start-prod:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.prod.yml up --build

stop:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml down

stop-prod:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.prod.yml down

run-local:
	python3 server.py

test:
	@echo "Testing server with GET request..."
	@curl -s http://localhost:8752 | head -10

logs:
	docker compose -f docker-compose.dev.yml logs -f
