#!/bin/bash

# Number of runs
NUM_RUNS=10
SCRIPT="python train.py"

echo "Running training $NUM_RUNS times..."
for i in $(seq 1 $NUM_RUNS); do
    echo "======================"
    echo "Run $i"
    echo "======================"
    $SCRIPT
    echo ""
done

echo "All runs completed."
