# Atalhos para as tarefas do dia a dia.
# Rode `make` ou `make help` para ver os comandos.

.PHONY: help install test lint format data baseline bert clean

help:  ## Lista os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Instala as dependências
	pip install -r requirements.txt

test:  ## Roda os testes (dados sintéticos, sem baixar nada)
	pytest

lint:  ## Verifica o estilo do código (não altera arquivos)
	ruff check .
	ruff format --check .

format:  ## Formata o código (altera arquivos)
	ruff format .
	ruff check --fix .

data:  ## Baixa o B2W-Reviews01, limpa e grava o manifesto do split
	python -m src.prepare

baseline:  ## Baseline TF-IDF + regressão logística; grava results/tfidf.json
	python -m src.baseline_tfidf

bert:  ## Fine-tuning do BERTimbau (precisa de GPU e requirements-train.txt; ver notebook)
	python -m src.train_bert

clean:  ## Remove arquivos temporários
	rm -rf .pytest_cache .ruff_cache __pycache__ src/__pycache__ tests/__pycache__
