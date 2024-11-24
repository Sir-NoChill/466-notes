#!/bin/bash

python3 -m venv venv
source "venv/bin/activate" || exit

pip install -r requirements.txt
kaggle competitions download -c jane-stree-treal-time-market-data-forecasting
unzip jane-street-real-time-market-data-forecasting
