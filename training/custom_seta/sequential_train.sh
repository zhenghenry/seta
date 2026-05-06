#!/bin/bash

for i in {20..22}; do
    python eval.py --config config.yaml --backend vllm --output-dir outputs/vllm_minimax_m2.5_${i} --preserve-docker-resources
done

python eval.py --config config.yaml --backend vllm --output-dir outputs/vllm_minimax_m2.5_23 --cleanup-docker-resources