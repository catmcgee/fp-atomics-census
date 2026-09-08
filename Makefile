PYTHON ?= .venv/bin/python
REPOS_DIR ?= repos
MANIFEST ?= scan-manifest.json

.PHONY: docs check-derived reanalyse dispositions shape-tables attach-runtime summary help venv clone census census-all test validate csv

help:
	@echo "make venv           create .venv with pinned dependencies"
	@echo "make clone          clone every repo in $(MANIFEST) at its pinned sha"
	@echo "make census REPO=x  re-run the scanner on one repo (uses pinned sha; override with SHA=...)"
	@echo "make census-all     re-run the scanner on every repo in the manifest"
	@echo "make validate       validate inventory/*.jsonl against triage/inventory.schema.json"
	@echo "make csv            regenerate inventory.csv from inventory/*.jsonl"
	@echo "make test           run the scanner unit tests"

venv:
	uv sync --frozen --python 3.11 --extra dev --extra probe-test

clone:
	$(PYTHON) -m scan.clone --manifest $(MANIFEST) --dest $(REPOS_DIR)

census:
	@test -n "$(REPO)" || (echo "usage: make census REPO=<name> [SHA=<sha>]"; exit 1)
	$(PYTHON) -m scan.clone --manifest $(MANIFEST) --dest $(REPOS_DIR) --only $(REPO) $(if $(SHA),--sha $(SHA),)
	$(PYTHON) -m scan.run --manifest $(MANIFEST) --repos-dir $(REPOS_DIR) --only $(REPO) $(if $(SHA),--sha $(SHA),) --out candidates

census-all:
	$(PYTHON) -m scan.run --manifest $(MANIFEST) --repos-dir $(REPOS_DIR) --out candidates

validate:
	$(PYTHON) -m triage.validate --schema triage/inventory.schema.json --manifest $(MANIFEST) --repos-dir $(REPOS_DIR) --dispositions-dir triage/dispositions inventory/*.jsonl

attach-runtime:
	$(PYTHON) -m triage.attach_runtime inventory/*.jsonl

reanalyse:
	$(PYTHON) probes/shape/reanalyse.py

shape-tables: reanalyse
	$(PYTHON) probes/shape/tables.py probes/shape/results

summary:
	$(PYTHON) -m triage.summary inventory/*.jsonl

csv:
	$(PYTHON) -m triage.to_csv inventory/*.jsonl > inventory.csv

test:
	$(PYTHON) -m pytest -q tests

dispositions:
	$(PYTHON) -m triage.dispositions

docs: reanalyse attach-runtime
	$(PYTHON) -m triage.to_csv inventory/*.jsonl > inventory.csv
	$(PYTHON) -m triage.render_readme

check-derived:
	$(PYTHON) probes/shape/reanalyse.py --check
	$(PYTHON) -m triage.check_derived
	$(PYTHON) -m triage.render_readme --check
