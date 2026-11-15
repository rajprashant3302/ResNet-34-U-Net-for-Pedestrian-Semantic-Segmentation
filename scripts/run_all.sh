#!/usr/bin/env bash
# Full real experiment: download data -> EDA -> train/evaluate all experiments x seeds -> tables, figures, README.
set -e
python -m src.download_data
python -m src.eda
python -m src.run_experiments --config configs/default.yaml
