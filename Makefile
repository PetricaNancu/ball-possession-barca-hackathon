# One command per deliverable. `make submission` must regenerate the exact file we upload.

PY        ?= .venv/bin/python
TEAM      ?= $(shell $(PY) -c "import yaml;print(yaml.safe_load(open('configs/default.yaml'))['team'])" 2>/dev/null || echo our_team)
# Data bundle: env var wins; otherwise the organisers' repo next to this one.
MTAH_DATA_DIR ?= $(abspath ../more-than-a-hack-2027/data)
export MTAH_DATA_DIR

.PHONY: help setup check baseline submission validate evaluate board test clean

help:
	@echo "setup       create .venv and install pinned dependencies"
	@echo "check       verify the data bundle, build the tracking cache, print statistics"
	@echo "baseline    reproduce the organisers' baseline through our loader (loader test)"
	@echo "submission  build evaluation/results/$(TEAM).csv and validate it"
	@echo "validate    run the official validator on the submission"
	@echo "evaluate    score the submission on the public 5-minute window (smoke test only)"
	@echo "board       official ranked board over evaluation/results/*.csv"
	@echo "test        unit tests (+ loader-vs-reference test when data is present)"

setup:
	python3.12 -m venv .venv || python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

check:
	$(PY) scripts/check_data.py

baseline:
	$(PY) scripts/make_submission.py --method baseline --window 50 --team ours_baseline50

submission:
	$(PY) scripts/make_submission.py

validate:
	$(PY) evaluation/validate_submission.py evaluation/results/$(TEAM).csv

evaluate:
	$(PY) scripts/evaluate.py evaluation/results/$(TEAM).csv --bootstrap 1000

board:
	$(PY) evaluation/run_evaluation.py --bootstrap 1000

test:
	$(PY) -m pytest

clean:
	rm -rf cache/*.npz evaluation/reports evaluation/scoreboard.csv
