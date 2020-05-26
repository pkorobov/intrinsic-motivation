#!/bin/bash

for memory_size in 100000 10000 1000; do
  for method in None Forward Inverse ICM RND; do
    for seed in $(seq 0 4); do
      echo "${method}, memory_size=${memory_size}, seed=${seed}"
      python experiment.py --int_motivation_type=$method --seed=$seed --output_dir=logs/${method}_${memory_size} \
          --memory_size=$memory_size
    done
  done
done