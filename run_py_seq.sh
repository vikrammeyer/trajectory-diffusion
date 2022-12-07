#!/bin/bash
python3 scripts/train_diff.py -n 100000 -t 500 -d data/final-dataset -o results/dec6-diff

python3 scripts/train_diff.py -n 100000 -t 500 -d data/final-dataset -dt channels -o results/dec6-diff-channels

python3 scripts/train_fc.py -n 100000 -d data/final-dataset -o results/dec6-baseline
