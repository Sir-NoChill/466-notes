#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate || exit

pip install matplotlib numpy
