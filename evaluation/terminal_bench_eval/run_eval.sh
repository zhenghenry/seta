#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
echo "Changing to script directory: $script_dir"
cd "$script_dir"

CONDA_ENV_NAME="tagent"

export OPENAI_API_KEY="${OPENAI_API_KEY}"
export CAMEL_LOG_DIR="${script_dir}/evaluation/terminal_bench_eval/logs/camel_logs"
mkdir -p "$CAMEL_LOG_DIR"

    source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$CONDA_ENV_NAME"

export OPENAI_API_KEY="${OPENAI_API_KEY}"

uv run tb run \
    --dataset terminal-bench-core==0.1.1 \
    --agent-import-path evaluation.terminal_bench_eval.tbench_camel_agent:TerminalBenchAgent \
    --n-concurrent-trials 2 
    
    # --dataset terminal-bench-core==0.1.1 \
    # --dataset-path dataset/tbench-tasks \
