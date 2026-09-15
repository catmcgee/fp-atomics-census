PYTHON ?= .venv/bin/python
REPOS_DIR ?= repos
MANIFEST ?= scan-manifest.json
# Sorted bytewise by make, matching Python's sorted(); a shell glob follows the
# locale and would reorder inventory.csv on macOS, making check-derived report it stale.
INVENTORY := $(sort $(wildcard inventory/*.jsonl))

.PHONY: docs check-derived reanalyse dispositions shape-tables attach-runtime summary help venv clone census census-all test validate csv repin-plan

help:
	@echo "make venv           create .venv with pinned dependencies"
	@echo "make clone          clone every repo in $(MANIFEST) at its pinned sha"
	@echo "make census REPO=x  re-run the scanner on one repo (uses pinned sha; override with SHA=...)"
	@echo "make census-all     re-run the scanner on every repo in the manifest"
	@echo "make validate       validate inventory/*.jsonl against triage/inventory.schema.json"
	@echo "make csv            regenerate inventory.csv from inventory/*.jsonl"
	@echo "make test           run the scanner unit tests"
	@echo "make repin-plan OUT=dir [REPO=x] [TARGETS='x=tag']  map the inventory to newer upstream commits and write the review worklist"

venv:
	uv sync --frozen --python 3.11 --extra dev --extra probe-test

clone:
	$(PYTHON) -m scan.clone --manifest $(MANIFEST) --dest $(REPOS_DIR)

census:
	@test -n "$(REPO)" || (echo "usage: make census REPO=<name> [SHA=<sha>]"; exit 1)
	$(PYTHON) -m scan.clone --manifest $(MANIFEST) --dest $(REPOS_DIR) --only $(REPO) $(if $(SHA),--sha $(SHA),)
	$(PYTHON) -m scan.run --manifest $(MANIFEST) --repos-dir $(REPOS_DIR) --only $(REPO) $(if $(SHA),--sha $(SHA),) --out candidates

repin-plan:
	@test -n "$(OUT)" || (echo "usage: make repin-plan OUT=<dir> [REPO=<name>] [BEFORE=<date>] [TARGETS='vllm=v0.29.0 ...']"; exit 1)
	$(PYTHON) -m scan.repin plan --out $(OUT) $(if $(REPO),--only $(REPO),) $(if $(BEFORE),--before $(BEFORE),) $(foreach t,$(TARGETS),--target $(t))

census-all:
	$(PYTHON) -m scan.run --manifest $(MANIFEST) --repos-dir $(REPOS_DIR) --out candidates

validate:
	$(PYTHON) -m triage.validate --schema triage/inventory.schema.json --manifest $(MANIFEST) --repos-dir $(REPOS_DIR) --dispositions-dir triage/dispositions $(INVENTORY)

attach-runtime:
	$(PYTHON) -m triage.attach_runtime $(INVENTORY)

reanalyse:
	$(PYTHON) probes/shape/reanalyse.py

shape-tables: reanalyse
	$(PYTHON) probes/shape/tables.py probes/shape/results

summary:
	$(PYTHON) -m triage.summary $(INVENTORY)

csv:
	$(PYTHON) -m triage.to_csv $(INVENTORY) > inventory.csv

test:
	$(PYTHON) -m pytest -q tests

dispositions:
	$(PYTHON) -m triage.dispositions

docs: reanalyse attach-runtime
	$(PYTHON) -m triage.to_csv $(INVENTORY) > inventory.csv
	$(PYTHON) -m triage.render_readme

check-derived:
	$(PYTHON) probes/shape/reanalyse.py --check
	$(PYTHON) -m triage.check_derived
	$(PYTHON) -m triage.render_readme --check
