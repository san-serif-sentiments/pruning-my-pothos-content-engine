.PHONY: up pull index run logs stop clean

up:
	docker compose -f infra/docker-compose.yml up -d

pull:
	# Pull LLMs; embeddings are handled by sentence-transformers inside engine
	docker exec -it $$(docker ps -qf name=ollama) ollama pull llama3.1:8b-instruct || true
	docker exec -it $$(docker ps -qf name=ollama) ollama pull phi3:mini || true

index:
	docker compose -f infra/docker-compose.yml exec engine python engine/rag/build_index.py

run:
	# Usage: make run brief=briefs/example.yaml
	docker compose -f infra/docker-compose.yml exec engine python engine/run.py $(brief)

logs:
	docker compose -f infra/docker-compose.yml logs -f

stop:
	docker compose -f infra/docker-compose.yml down

clean:
	rm -rf engine/.chroma artifacts/*
