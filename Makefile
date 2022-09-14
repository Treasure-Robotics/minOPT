# Makefile

all: develop
.PHONY: all

# ------------
# Installation
# ------------

develop:
	pip install -e .[dev]
.PHONY: develop

install:
	pip install .
.PHONY: install

# -------
# Linting
# -------

lint-dirs := minopt tests

lint:
	black --diff --check $(lint-dirs)
	isort --check-only $(lint-dirs)
	mypy $(lint-dirs)
	flake8 --count --show-source --statistics $(lint-dirs)
	pylint $(lint-dirs)
.PHONY: lint

format:
	black $(lint-dirs)
	isort $(lint-dirs)
.PHONY: format

# -------
# Testing
# -------

test:
	pytest .
.PHONY: test
