#!/bin/bash

# This is just a simple script to get the example script of S4 running on a GPU cluster.
# Let's run the example script for the MNIST dataset. According to the README.md, after
# one epeoch we should get an accuracy of around 90%.
# Each epoch should take 1-3 minutes.

# 1) Example launch:
# bash demorun.sh

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv python pin 3.12
uv sync

uv run python -m example_deepar --d_model 256 --context_length 864 --weight_decay 0.0
