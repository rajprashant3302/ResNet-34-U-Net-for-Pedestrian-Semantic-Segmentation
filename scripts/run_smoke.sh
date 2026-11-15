#!/usr/bin/env bash
# Pipeline self-test on SYNTHETIC data (a few minutes on CPU). Numbers are meaningless; it only proves the code runs.
set -e
python -m pytest -q tests
python -m src.make_synthetic --n 64
python -m src.eda --config configs/smoke.yaml
python -m src.run_experiments --config configs/smoke.yaml
