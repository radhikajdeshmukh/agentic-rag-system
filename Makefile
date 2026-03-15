APP := uv run basic-rag-agent

.PHONY: help sync run lock clean

help:
	@printf "Available targets:\n"
	@printf "  help  Show available commands\n"
	@printf "  sync  Install dependencies with uv\n"
	@printf "  run   Run the CLI app\n"
	@printf "  lock  Refresh uv lockfile\n"
	@printf "  clean Remove local cache files\n"

sync:
	uv sync

run:
	$(APP)

lock:
	uv lock

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
