#!/bin/bash

# This is just a simple script to get the example script of S4 running on a GPU cluster.
# Let's run the example script for the MNIST dataset. According to the README.md, after
# one epeoch we should get an accuracy of around 90%.
# Each epoch should take 1-3 minutes.

# 1) Example launch:
# bash demorun_deepar.sh

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv python pin 3.12
uv sync

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash demorun_deepar.sh`
if [ -z "$WANDB_RUN" ]; then
  WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Install S4 kernel to speed up computations
cd extensions/kernels
uv run python setup.py install
cd ../..

# -----------------------------------------------------------------------------
# Start training
uv run python -m example_deepar --d_model 256 --context_length 1440 --batch_size 512 --weight_decay 0.0 --num_workers 0 --epochs 1
