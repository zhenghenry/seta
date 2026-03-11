# Custom SETA Evaluation

Run terminal-bench evaluation tasks using a **locally hosted vLLM server** or the **OpenRouter API**, without requiring the AReaL training framework.

This is a standalone evaluation workflow extracted from `tbench_areal_workflow/eval.py`, with all AReaL/training dependencies removed.

## Prerequisites

1. **SETA environment** — follow the main `seta/setup.sh` to install `camel`, `terminal-bench`, and Docker.
2. **Dataset** — a parquet file under `seta/dataset/` (default: `tbench-tasks_convert/train_filtered2.parquet`).
3. **Docker** — required for task environments. Ensure Docker is running and the address pool is configured (see main README).
4. **Model endpoint** — either a running vLLM server or an OpenRouter API key.

## Quick start

### Option A: vLLM (local)

Start a vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8000 \
    --tensor-parallel-size 1
```

Run evaluation:

```bash
cd seta/training/custom_seta
python eval.py --config config.yaml
```

### Option B: OpenRouter (cloud)

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

cd seta/training/custom_seta
python eval.py --config config.yaml --backend openrouter
```

## Configuration

Edit `config.yaml` or override via CLI flags:

| Flag | Description |
|------|-------------|
| `--backend {vllm,openrouter}` | Switch between backends |
| `--model MODEL` | Override model name |
| `--base-url URL` | Override API base URL |
| `--api-key KEY` | Override API key |
| `--tasks TASK [TASK ...]` | Evaluate only specific tasks |
| `--batch-size N` | Number of concurrent tasks |
| `--output-dir DIR` | Custom output directory |
| `--cleanup-docker-resources` | Force full Docker cleanup after each task |
| `--preserve-docker-resources` | Override config to keep task volumes/images |

`config.yaml` also supports `cleanup_docker_resources: true` if you want each eval run to remove task Docker containers, networks, volumes, and images after cleanup. Leave it `false` to preserve the current faster-but-leakier behavior.

### Example: run two specific tasks with OpenRouter

```bash
python eval.py \
    --config config.yaml \
    --backend openrouter \
    --model anthropic/claude-sonnet-4 \
    --tasks distribution-search play-lord \
    --batch-size 2
```

### Example: custom vLLM endpoint

```bash
python eval.py \
    --config config.yaml \
    --base-url http://gpu-server:8000/v1 \
    --model meta-llama/Llama-3-70B-Instruct
```

## Output

Results are saved to `<output_dir>/eval_results.json` with per-task reward scores and timing. Failed runs also include `error_type`, `error`, and usually `traceback` so early startup/model failures are visible without needing the terminal logs.

Detailed test results for each task are saved under:
```
<output_dir>/CamelTerminalAgent_Output/<task_name>/<run_id>/test_results.json
```

## Architecture

```
eval.py
  ├── load_config()          — YAML config with CLI overrides
  ├── build_docker_image()   — pre-build task containers
  ├── create_model()         — CAMEL model via ModelFactory (OpenAI-compatible)
  ├── TerminalAgent          — sets up Docker env, runs ChatAgent, evaluates
  │     ├── _setup_env()     — TrialHandler + Terminal start
  │     ├── _setup_agent()   — TerminalToolkit + ChatAgent creation
  │     ├── _evaluate()      — copy tests, run, parse results
  │     └── _cleanup()       — stop containers
  └── run_evaluation()       — batch orchestration with asyncio.gather
```

Compared to `tbench_areal_workflow/eval.py`, this version:
- Removes all AReaL framework dependencies (config, rollout engine, perf tracing, stats)
- Uses CAMEL's `ModelFactory` with `ModelPlatformType` for model creation
- Uses standard `ChatAgent` instead of `ChatAgentTrace`
- Replaces AReaL YAML config system with plain `pyyaml`
- Supports any OpenAI-compatible endpoint (vLLM, OpenRouter, etc.)
