"""Custom SETA Evaluation — run terminal-bench tasks against a vLLM server
or the OpenRouter API, without requiring the AReaL training framework.

Usage examples:
    # vLLM (default)
    python eval.py --config config.yaml

    # OpenRouter
    python eval.py --config config.yaml --backend openrouter

    # Specific tasks only
    python eval.py --config config.yaml --tasks distribution-search play-lord
"""

import asyncio
import argparse
import datetime
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import yaml
from datasets import load_dataset

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit, FunctionTool
from camel.types import ModelPlatformType

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal

SETA_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SETA_ROOT / "training" / "tbench_areal_workflow"))
from prompts import get_developer_agent_prompt


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class TaskTimeouts:
    reset_env: float = 300.0
    reset_agent: float = 120.0
    agent_step: float = 300.0
    evaluate: float = 600.0
    cleanup: float = 300.0


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_docker_image(task: dict, dataset_base: Path, timeout: float = 1200.0):
    """Pre-build the Docker image for a task so container start is fast."""
    task_path = dataset_base / task.get("task_path")
    trial_handler = TrialHandler(
        trial_name="build_run",
        input_path=task_path,
        output_path=Path("build_outputs"),
    )
    compose_mgr = DockerComposeManager(
        client_container_name=trial_handler.client_container_name,
        client_image_name=trial_handler.client_image_name,
        docker_image_name_prefix=trial_handler.docker_image_name_prefix,
        docker_compose_path=trial_handler.task_paths.docker_compose_path,
        no_rebuild=True,
        cleanup=True,
        sessions_logs_path=trial_handler.trial_paths.sessions_path,
        agent_logs_path=trial_handler.trial_paths.agent_logging_dir,
    )
    compose_mgr.build(timeout=timeout)


def create_model(model_config: dict):
    """Create a CAMEL model backend for an OpenAI-compatible endpoint.

    Works with vLLM servers, OpenRouter, and any provider that exposes
    the ``/v1/chat/completions`` endpoint.
    """
    api_key = model_config.get("api_key") or "EMPTY"
    base_url = model_config["base_url"]
    model_name = model_config["model"]
    temperature = model_config.get("temperature", 0.7)
    max_tokens = model_config.get("max_tokens_per_turn", 10240)

    model_config_dict = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Try CAMEL's model factory with platform types that map to
    # OpenAI-compatible endpoints.
    for platform in _candidate_platforms():
        try:
            return ModelFactory.create(
                model_platform=platform,
                model_type=model_name,
                api_key=api_key,
                url=base_url,
                model_config_dict=model_config_dict,
            )
        except Exception:
            continue

    raise RuntimeError(
        f"Could not create a CAMEL model backend for {base_url}. "
        "Ensure your camel-ai version supports OpenAI-compatible models."
    )


def _candidate_platforms():
    """Yield ModelPlatformType values to try, most specific first."""
    for name in (
        "OPENAI_COMPATIBLE_MODEL",
        "VLLM",
        "OPENAI",
    ):
        val = getattr(ModelPlatformType, name, None)
        if val is not None:
            yield val


# ── Agent ────────────────────────────────────────────────────────────────────


class TerminalAgent:
    """Runs a CAMEL ChatAgent on a single terminal-bench task inside Docker."""

    def __init__(
        self,
        model_config: dict,
        max_tokens_per_turn: int = 10240,
        max_total_tokens: int = 28672,
        output_path: str = "CamelTerminalAgent_Output",
        max_iteration: int = 20,
        executor: ThreadPoolExecutor = None,
        task_timeouts: TaskTimeouts = None,
        non_think_mode: bool = True,
        cleanup_docker_resources: bool = False,
        dataset_base: Path = None,
    ):
        self.model_config = model_config
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_total_tokens = max_total_tokens
        self.output_path = output_path
        self.max_iteration = max_iteration
        self.task_timeouts = task_timeouts or TaskTimeouts()
        self.executor = executor
        self.non_think_mode = non_think_mode
        self.cleanup_docker_resources = cleanup_docker_resources
        self.dataset_base = dataset_base or (SETA_ROOT / "dataset")
        self.terminal = None
        assert self.executor is not None, "ThreadPoolExecutor required"

    # ── public entry point ───────────────────────────────────────────────

    async def run(self, data: dict, uid: str = None, traj_i: int = 0) -> dict:
        task_name = data.get("task_name")
        self.task_name = task_name
        self.uid = uid or uuid.uuid4().hex[:8]
        self.traj_i = traj_i
        reward = None
        result = {
            "task_name": task_name,
            "uid": self.uid,
            "traj_i": traj_i,
        }
        t0 = time.time()
        print(f"[{task_name}] Starting (uid={self.uid}, traj={traj_i})")

        try:
            prompt = await self._exec(
                self._setup_env, data, self.uid,
                timeout=self.task_timeouts.reset_env,
            )
            print(f"[{task_name}] Environment ready")

            await self._exec(
                self._setup_agent,
                timeout=self.task_timeouts.reset_agent,
            )
            print(f"[{task_name}] Agent ready")

            try:
                self.response = await asyncio.wait_for(
                    self.agent.astep(prompt),
                    timeout=self.task_timeouts.agent_step,
                )
            except asyncio.TimeoutError:
                print(f"[{task_name}] Agent step timed out")

            print(f"[{task_name}] Agent finished, saving trajectory…")
            self._save_trajectory()

            print(f"[{task_name}] Evaluating…")
            reward = await self._exec(
                self._evaluate,
                timeout=self.task_timeouts.evaluate,
            )
            print(f"[{task_name}] Reward: {reward}")

        except asyncio.TimeoutError as exc:
            print(f"[{task_name}] Timeout: {exc}")
            result["error_type"] = "timeout"
            result["error"] = str(exc)
        except Exception as exc:
            print(f"[{task_name}] Error: {exc}")
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()
            print(result["traceback"])
        finally:
            await self._cleanup()
            result["reward"] = reward
            result["elapsed_sec"] = round(time.time() - t0, 1)
            return result

    # ── internals ────────────────────────────────────────────────────────

    async def _exec(self, fn, *args, timeout=None, **kwargs):
        loop = asyncio.get_running_loop()
        coro = loop.run_in_executor(self.executor, partial(fn, *args, **kwargs))
        if timeout is not None:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    def _setup_env(self, task: dict, uid: str) -> str:
        out = Path(self.output_path).resolve()
        out.mkdir(parents=True, exist_ok=True)

        task_path = self.dataset_base / task.get("task_path")
        instruction = task.get("instruction")
        task_id = task.get("task_name")

        self.trial_handler = TrialHandler(
            trial_name=f"{task_id}.{uid}.custom-seta",
            input_path=task_path,
            output_path=out,
        )
        task_cfg = self.trial_handler.task
        self.parser = ParserFactory.get_parser(task_cfg.parser_name)

        self.terminal = Terminal(
            client_container_name=self.trial_handler.client_container_name,
            client_image_name=self.trial_handler.client_image_name,
            docker_compose_path=self.trial_handler.task_paths.docker_compose_path,
            docker_image_name_prefix=self.trial_handler.docker_image_name_prefix,
            sessions_logs_path=self.trial_handler.trial_paths.sessions_path,
            agent_logs_path=self.trial_handler.trial_paths.agent_logging_dir,
            no_rebuild=True,
            cleanup=self.cleanup_docker_resources,
        )
        self.terminal.start(timeout=self.task_timeouts.reset_env)
        return f"Task name:{task_id}\nTask instruction: {instruction}"

    def _setup_agent(self):
        logs_dir = (
            self.trial_handler.trial_paths.sessions_path
            / "terminal_toolkit_session_logs"
        )
        toolkit = TerminalToolkit(
            timeout=20.0,
            working_directory=None,
            use_docker_backend=True,
            docker_container_name=self.trial_handler.client_container_name,
            session_logs_dir=logs_dir,
            safe_mode=False,
        )
        tools = [
            FunctionTool(toolkit.shell_exec),
            FunctionTool(toolkit.shell_view),
            FunctionTool(toolkit.shell_write_to_process),
            FunctionTool(toolkit.shell_write_content_to_file),
        ]

        system_message = get_developer_agent_prompt(
            current_date=str(datetime.date.today()),
            system="Linux (in Docker)",
            machine="x86_64",
            is_workforce=False,
            non_think_mode=self.non_think_mode,
        )

        model = create_model(self.model_config)

        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Developer Agent",
                content=system_message,
            ),
            model=model,
            tools=tools,
            token_limit=self.max_total_tokens,
        )
        self.agent.reset()
        self.agent.max_iteration = self.max_iteration

    def _evaluate(self) -> float | None:
        assert self.trial_handler is not None and self.terminal is not None

        paths = [self.trial_handler.task_paths.run_tests_path]
        if self.trial_handler.task_paths.test_dir.exists():
            paths.append(self.trial_handler.task_paths.test_dir)
        self.terminal.copy_to_container(
            paths=paths,
            container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
        )

        test_session = self.terminal.create_session(
            "tests", is_active_stream=False, as_configured_user=False,
        )
        test_script = str(DockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh")

        try:
            test_session.send_keys(
                [f"bash {test_script}", "Enter"],
                block=True,
                max_timeout_sec=min(
                    self.task_timeouts.evaluate,
                    4 * self.trial_handler.task.max_test_timeout_sec,
                ),
            )
            output = test_session.capture_pane(capture_entire=True)
            parsed = self.parser.parse(output)

            all_passed = parsed and all(
                s == UnitTestStatus.PASSED for s in parsed.values()
            )
            pass_ratio = (
                sum(1 for s in parsed.values() if s == UnitTestStatus.PASSED)
                / len(parsed)
                if parsed
                else 0.0
            )

            result_dict = {
                "test_results": {
                    k: (v == UnitTestStatus.PASSED) for k, v in parsed.items()
                },
                "all_passed": all_passed,
                "pass_ratio": pass_ratio,
            }
            try:
                result_dict["iteration"] = len(self.response.info["tool_calls"])
                result_dict.update(self.response.info["usage"])
            except Exception:
                pass

            out_path = (
                self.trial_handler.trial_paths.sessions_path.parent
                / "test_results.json"
            )
            with open(out_path, "w") as f:
                json.dump(result_dict, f, indent=4)

        except Exception:
            all_passed = False
            pass_ratio = None

        return pass_ratio

    def _save_trajectory(self):
        """Extract the full conversation from the agent's memory and save
        a single ``trajectory.json`` alongside the test results."""
        if not hasattr(self, "agent") or self.agent is None:
            return
        trial_dir = self.trial_handler.trial_paths.sessions_path.parent
        try:
            records = sorted(
                self.agent.memory.retrieve(),
                key=lambda record: record.memory_record.timestamp,
            )
            messages = []
            for record in records:
                msg = record.memory_record.to_openai_message()
                reasoning_content = getattr(
                    record.memory_record.message, "reasoning_content", None
                )
                if reasoning_content:
                    msg = dict(msg)
                    msg["reasoning_content"] = reasoning_content
                messages.append(msg)

            trajectory = {
                "task_name": self.task_name,
                "uid": self.uid,
                "traj_i": self.traj_i,
                "model": self.model_config.get("model"),
                "timestamp": datetime.datetime.now().isoformat(),
                "messages": messages,
            }
            traj_path = trial_dir / "trajectory.json"
            with open(traj_path, "w") as f:
                json.dump(trajectory, f, indent=2, default=str)
        except Exception as exc:
            print(f"[{self.task_name}] Failed to save trajectory: {exc}")

    @staticmethod
    def _remove_empty_dirs(trial_dir: Path):
        """Remove empty directories created by terminal-bench that we don't use."""
        for name in ("panes", "agent-logs", "CAMEL_LOG_DIR"):
            d = trial_dir / name
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass

    async def _cleanup(self):
        try:
            if self.terminal is not None:
                await self._exec(
                    self.terminal.stop,
                    timeout=self.task_timeouts.cleanup,
                )
                print(f"[{self.task_name}] Cleaned up")
        except Exception as exc:
            print(f"[{self.task_name}] Cleanup error: {exc}")
        finally:
            if hasattr(self, "trial_handler") and self.trial_handler is not None:
                self._remove_empty_dirs(
                    self.trial_handler.trial_paths.sessions_path.parent
                )


# ── Orchestrator ─────────────────────────────────────────────────────────────


async def run_evaluation(config: dict):
    backend = config.get("backend", "vllm")
    backend_cfg = config.get(backend, {})

    model_config = {
        "base_url": backend_cfg["base_url"],
        "model": backend_cfg["model"],
        "api_key": (
            backend_cfg.get("api_key")
            or os.environ.get(
                "OPENROUTER_API_KEY" if backend == "openrouter" else "VLLM_API_KEY",
                "EMPTY",
            )
        ),
        "temperature": config.get("temperature", 0.7),
        "max_tokens_per_turn": config.get("max_tokens_per_turn", 10240),
    }

    tconf = config.get("timeouts", {})
    timeouts = TaskTimeouts(
        reset_env=tconf.get("reset_env", 300),
        reset_agent=tconf.get("reset_agent", 120),
        agent_step=tconf.get("agent_step", 300),
        evaluate=tconf.get("evaluate", 600),
        cleanup=tconf.get("cleanup", 300),
    )

    dataset_base = SETA_ROOT / "dataset"
    ds_path = str(dataset_base / config["dataset_path"])
    dataset = load_dataset(path="parquet", split="train", data_files=[ds_path])

    task_filter = config.get("task_filter", [])
    if task_filter:
        dataset = dataset.filter(lambda x: x["task_name"] in task_filter)

    output_dir = Path(config.get("output_dir", "outputs/custom_seta_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    n_trajs = config.get("n_trajs", 1)
    max_workers = config.get("max_workers", 16)
    batch_size = config.get("batch_size", 4)
    cleanup_docker_resources = config.get("cleanup_docker_resources", False)
    executor = ThreadPoolExecutor(max_workers=max_workers)

    print(f"Backend:      {backend}")
    print(f"Model:        {model_config['model']}")
    print(f"Base URL:     {model_config['base_url']}")
    print(f"Dataset:      {ds_path} ({len(dataset)} tasks)")
    print(f"Output:       {output_dir}")
    print(f"Trajectories: {n_trajs} per task")
    print(f"Concurrency:  batch_size={batch_size}, workers={max_workers}")
    print(f"Cleanup:      {'full docker cleanup' if cleanup_docker_resources else 'preserve task volumes/images'}")
    print()

    # ── Pre-build Docker images ──────────────────────────────────────────

    print("Pre-building Docker images …")
    loop = asyncio.get_running_loop()
    for idx, data in enumerate(dataset):
        try:
            await loop.run_in_executor(
                executor,
                partial(
                    build_docker_image,
                    task=data,
                    dataset_base=dataset_base,
                    timeout=timeouts.reset_env,
                ),
            )
            print(f"  [{idx + 1}/{len(dataset)}] {data['task_name']} ✓")
        except Exception as exc:
            print(f"  [{idx + 1}/{len(dataset)}] {data['task_name']} FAILED: {exc}")
    print()

    # ── Run evaluation in batches ────────────────────────────────────────

    all_results = []
    dataset_list = list(dataset)

    for batch_start in range(0, len(dataset_list), batch_size):
        batch = dataset_list[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        print(f"{'=' * 60}")
        print(
            f"Batch {batch_num}: tasks "
            f"{batch_start + 1}–{batch_start + len(batch)} / {len(dataset_list)}"
        )
        print(f"{'=' * 60}")

        coros = []
        for data in batch:
            for traj_i in range(n_trajs):
                agent = TerminalAgent(
                    model_config=model_config,
                    max_tokens_per_turn=config.get("max_tokens_per_turn", 10240),
                    max_total_tokens=config.get("max_total_tokens", 28672),
                    max_iteration=config.get("max_iteration", 20),
                    output_path=str(output_dir / "CamelTerminalAgent_Output"),
                    executor=executor,
                    non_think_mode=config.get("non_think_mode", True),
                    cleanup_docker_resources=cleanup_docker_resources,
                    task_timeouts=timeouts,
                    dataset_base=dataset_base,
                )
                coros.append(agent.run(data=dict(data), traj_i=traj_i))

        results = await asyncio.gather(*coros, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                print(f"  Exception: {r}")
                all_results.append({"error": str(r)})
            else:
                all_results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────

    results_file = output_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    ok = [r for r in all_results if isinstance(r, dict) and r.get("reward") is not None]
    failed = len(all_results) - len(ok)

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    if ok:
        rewards = [r["reward"] for r in ok]
        print(f"  Tasks evaluated:  {len(ok)}")
        print(f"  Tasks failed:     {failed}")
        print(f"  Average reward:   {sum(rewards) / len(rewards):.4f}")
        print(f"  Full passes:      {sum(1 for r in rewards if r == 1.0)} / {len(ok)}")
    else:
        print("  No successful evaluations.")
    print(f"  Results: {results_file}")

    executor.shutdown(wait=True)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Custom SETA Evaluation — vLLM / OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to YAML config file",
    )
    parser.add_argument(
        "--backend", choices=["vllm", "openrouter"], help="Override backend",
    )
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--base-url", help="Override base URL")
    parser.add_argument("--api-key", help="Override API key")
    parser.add_argument("--tasks", nargs="+", help="Run only these tasks")
    parser.add_argument("--batch-size", type=int, help="Concurrent task count")
    parser.add_argument("--output-dir", help="Override output directory")
    cleanup_group = parser.add_mutually_exclusive_group()
    cleanup_group.add_argument(
        "--cleanup-docker-resources",
        dest="cleanup_docker_resources",
        action="store_true",
        default=None,
        help="Force removal of task Docker containers, networks, volumes, and images",
    )
    cleanup_group.add_argument(
        "--preserve-docker-resources",
        dest="cleanup_docker_resources",
        action="store_false",
        help="Preserve task Docker volumes/images after each run",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.backend:
        config["backend"] = args.backend
    backend = config.get("backend", "vllm")
    if args.model:
        config.setdefault(backend, {})["model"] = args.model
    if args.base_url:
        config.setdefault(backend, {})["base_url"] = args.base_url
    if args.api_key:
        config.setdefault(backend, {})["api_key"] = args.api_key
    if args.tasks:
        config["task_filter"] = args.tasks
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.cleanup_docker_resources is not None:
        config["cleanup_docker_resources"] = args.cleanup_docker_resources

    asyncio.run(run_evaluation(config))


if __name__ == "__main__":
    main()
