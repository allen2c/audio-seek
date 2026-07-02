# Development
fmt:
	@isort audio_seek tests
	@black audio_seek tests
	@ruff check --fix audio_seek tests

install:
	poetry install --all-extras --all-groups

update:
	poetry update

# Docs
mkdocs:
	mkdocs serve -a 0.0.0.0:8000

# Tests
pytest:
	python -m pytest
