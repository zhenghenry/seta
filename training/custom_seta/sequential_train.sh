#!/bin/bash

for i in {1..18}; do
    python eval.py --config config.yaml --backend vllm --output-dir outputs/vllm_minimax_m2.5_${i} --cleanup-docker-resources
done