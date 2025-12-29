import asyncio
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from dataclasses import dataclass, field
from datasets import load_dataset

import platform

import torch.distributed as dist
from camel.agents import ChatAgent
from transformers import PreTrainedTokenizerFast

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.camel.openai_model import AReaLOpenAICompatibleModel
from areal.experimental.openai import ArealOpenAI
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import cycle_dataloader
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger


# ---
import uuid
import datetime
from pathlib import Path
from prompts import get_developer_agent_prompt
import time
import json
import csv
from collections import defaultdict
import pandas as pd
import threading
import wandb

# ---
from train import AgentRLConfig, TaskTimeouts
from chat_agent_trace import ChatAgentTrace

# --- added imports for tracing ---
from areal.utils import perf_tracer
from areal.utils.perf_tracer import (
    trace_perf,
    session_context,
    atrace_session_phase,
    trace_session,
    Category,
    atrace_scope,
    trace_scope
)
from concurrent.futures import ThreadPoolExecutor

import subprocess
import asyncio
from functools import partial
import os
import sys
from dataclasses import dataclass, field
from datasets import load_dataset

import platform
import threading
import time
import torch.distributed as dist
from camel.agents import ChatAgent
from transformers import PreTrainedTokenizerFast

# --- CAMEL imports
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelFactory
from camel.toolkits import (
    AgentCommunicationToolkit,
    NoteTakingToolkit,
    TerminalToolkit,
    ToolkitMessageIntegration,
    FunctionTool
)

# --- Terminal Bench import
from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal

# ---
import uuid
import datetime
from pathlib import Path
from prompts import get_developer_agent_prompt
import json
import atexit

# --- relavive import
# from docker_cleanup import start_docker_cleanup, stop_docker_cleanup
from collect_results import collect_test_results, periodic_test_collection
from pre_build_tasks import build_docker_image

class CamelTerminalAgent:
    # Add a class-level executor that persists across instances

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast=None,
        max_tokens_per_turn: int = 1024,
        max_total_tokens: int = 32768,
        output_path: str = "CamelTerminalAgent_Output",
        max_iteration: int = 50,
        executor: ThreadPoolExecutor = None,
        task_timeouts: TaskTimeouts = None,
        non_think_mode: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_total_tokens = max_total_tokens
        self.output_path = output_path
        self.max_iteration = max_iteration
        self.task_timeouts = task_timeouts or TaskTimeouts()
        self.executor = executor
        self.non_think_mode = non_think_mode
        assert self.executor is not None, "Executor must be provided to CamelTerminalAgent"

    # Register a session for this run and trace the overall function
    @session_context()
    @trace_perf("CamelTerminalAgent.run_agent", category=Category.COMPUTE)
    async def run_agent(self, data, client, uid: str = None, traj_i: int = 0) -> float | None:
        """Execute a complete agent workflow: setup environment, run agent, cleanup.
        
        Returns:
            float | None: Reward value if successful, None if failed.
        """
        task_name = data.get('task_name')
        self.task_name = task_name
        self.uid = uid
        self.traj_i = traj_i
        self.meta_info = {}
        reward = None

        print(f"Running task {task_name}")
        
        try:
            # 1. Reset environment with timeout — record as a session phase
            async with atrace_scope(f"reset_env:{task_name}, traj:{traj_i}", args={"uid": uid, "timeout": self.task_timeouts._reset_env}):
                prompt = await self.run_in_executor(
                                            self._reset_env, 
                                            data, 
                                            uid, 
                                            timeout=self.task_timeouts._reset_env
                                            )
            print(f"env started: {task_name}")
            
            # 2. Reset agent with timeout
            async with atrace_scope(f"reset_agent:{task_name}, traj:{traj_i}", args={"uid": uid, "timeout": self.task_timeouts._reset_agent}):
                await self.run_in_executor(
                                        self._reset_agent, 
                                        client,
                                        timeout=self.task_timeouts._reset_agent
                                        )
            
            # 3. Run agent step (wrap as a "astep" session phase)
            try:
                async with atrace_scope(f"agent_astep:{task_name}, traj:{traj_i}", args={"uid": uid, "timeout": self.task_timeouts.agent_astep}):
                    self.response = await self.agent.astep(prompt)
            except asyncio.TimeoutError as e:
                print(f"Agent step timeout for task {task_name}: {e}")
            print(f"Task {task_name}: agent responded")
            
            # 4. Evaluate and return reward — this will be a session-level trace via @trace_session on the method
            async with atrace_session_phase("reward",
                    start_payload={"task_name": task_name, "traj_i": traj_i, "uid": uid, "timeout": self.task_timeouts._evaluate_completion_sync}):
                async with atrace_scope(f"evaluate_completion_sync:{task_name}, traj:{traj_i}", args={"uid": uid, "timeout": self.task_timeouts._evaluate_completion_sync}):
                    reward = await self.run_in_executor(
                                                self._evaluate_completion_sync,
                                                timeout=self.task_timeouts._evaluate_completion_sync
                                                )
            client.set_final_reward(reward)

        except asyncio.TimeoutError as e:
            print(f"Timeout for task {task_name}: {e}")
        except Exception as e:
            print(f"Error in task {task_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always cleanup
            try:
                if hasattr(self, 'terminal') and self.terminal is not None:
                    async with atrace_scope(f"cleanup_env:{task_name}, traj:{traj_i}", args={"uid": uid, "timeout": self.task_timeouts._cleanup}):
                        await self.run_in_executor(
                                                self._close_env,
                                                timeout=self.task_timeouts._cleanup
                                                )
                    print(f"Task {task_name}: cleaned up")
            except Exception as e:
                print(f"Cleanup error for task {task_name}: {e}")
            finally:
                return reward

    def _close_env(self):
        """Stop/cleanup terminal containers if present."""
        if self.terminal:
            self.terminal.stop(timeout=self.task_timeouts._cleanup)

    async def run_in_executor(self, fn, *args, timeout: float = None, **kwargs):
        """Run a function in separate thread pool executor with detailed logging."""
        loop = asyncio.get_running_loop()
        
        fn_name = fn.__name__
        task_name = getattr(self, 'task_name', 'unknown')
        
        # Log BEFORE requesting thread
        queue_size = self.executor._work_queue.qsize()
        worker_count = len(self.executor._threads) if hasattr(self.executor, '_threads') else 0
        start_time = time.time()
        
        # This line will BLOCK if no threads available
        executor_task = loop.run_in_executor(
            self.executor, 
            partial(fn, *args, **kwargs)
        )
        try:
            if timeout is not None:
                result = await asyncio.wait_for(executor_task, timeout=timeout)
            else:
                result = await executor_task
        finally:
            elapsed = time.time() - start_time

        return result

    def _reset_env(self, task: dict, uid: str):
        """Create trial, start containers and session, and build initial prompt."""
        output_path = Path(self.output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        task_path = Path(__file__).parent.parent.parent / "dataset" / task.get("task_path")
        print(f"Task path: {task_path}")
        instruction = task.get("instruction")
        task_id = task.get("task_name")

        self.trial_handler = TrialHandler(
            trial_name=f"{task_id}.{uid}.areal-run",
            input_path=task_path,
            output_path=output_path,
        )

        task_config = self.trial_handler.task
        self.parser = ParserFactory.get_parser(task_config.parser_name)

        self.client_container_name = f"{self.trial_handler.client_container_name}"
        self.terminal = Terminal(
            client_container_name=self.trial_handler.client_container_name,
            client_image_name=self.trial_handler.client_image_name,
            docker_compose_path=self.trial_handler.task_paths.docker_compose_path,
            docker_image_name_prefix=self.trial_handler.docker_image_name_prefix,
            sessions_logs_path=self.trial_handler.trial_paths.sessions_path,
            agent_logs_path=self.trial_handler.trial_paths.agent_logging_dir,
            no_rebuild=True,
            cleanup=False,
        )
        with trace_scope(f"reset_env.start_terminal:{task_id}, traj:{self.traj_i}", args={"uid": uid}):
            self.terminal.start(timeout=self.task_timeouts._reset_env)

        usr_msg = (
            f"Task name:{self.task_name}\nTask instruction: {instruction}"
        )
        return usr_msg
    
    def _reset_agent(self, client):
        """Reset the agent with a new client."""
        session_logs_dir = self.trial_handler.trial_paths.sessions_path / "terminal_toolkit_session_logs"
        working_dir = self.trial_handler.trial_paths.agent_logging_dir / "CAMEL_WORKDIR"
        terminal_toolkit_kwargs = {
                                    'timeout': 20.0,
                                    'working_directory': None,
                                    'use_docker_backend': True,
                                    'docker_container_name': self.trial_handler.client_container_name,
                                    'session_logs_dir': session_logs_dir,
                                    'safe_mode': False,
                                    }


        terminal_toolkit = TerminalToolkit(**terminal_toolkit_kwargs)
        # Get enhanced tools
        tools = [
            FunctionTool(terminal_toolkit.shell_exec),
            FunctionTool(terminal_toolkit.shell_view),
            FunctionTool(terminal_toolkit.shell_write_to_process),
            FunctionTool(terminal_toolkit.shell_write_content_to_file),    
        ]

        system_message = get_developer_agent_prompt(
                                                    current_date = str(datetime.date.today()), 
                                                    system = "Linux (in Docker)", 
                                                    machine = "x86_64", 
                                                    is_workforce = False,
                                                    non_think_mode = self.non_think_mode
                                                    )
        print(f"starting chat agent")
        os.environ['CAMEL_MODEL_LOG_ENABLED'] = "True"
        os.environ['CAMEL_LOG_DIR'] = str(self.trial_handler.trial_paths.sessions_path.parent / "CAMEL_LOG_DIR")
        model = AReaLOpenAICompatibleModel(
                openai_client=client, 
                tokenizer=self.tokenizer, 
                model_type="areal", 
                model_config_dict={
                    "max_tokens": self.max_total_tokens,
                    "max_completion_tokens": self.max_tokens_per_turn,
                    },
            )
        self.agent = ChatAgentTrace(
                            system_message=BaseMessage.make_assistant_message(
                                role_name="Developer Agent",
                                content=system_message,
                            ),
                            model=model,
                            tools=tools,
                            token_limit=self.max_total_tokens,
                            step_timeout=self.task_timeouts.agent_astep,
                            )
        self.agent.reset()
        self.agent.max_iteration = self.max_iteration
        print(f"{self.task_name}: agent started")

    def _evaluate_completion_sync(self) -> float:
        """Copy tests, run them, parse output, and return a binary reward."""
        assert self.trial_handler is not None and self.terminal is not None

        # Copy tests into the container
        paths = [self.trial_handler.task_paths.run_tests_path]
        if self.trial_handler.task_paths.test_dir.exists():
            paths.append(self.trial_handler.task_paths.test_dir)
        with trace_scope(f"evaluate_completion_sync.copy_tests:{self.task_name}, traj:{self.traj_i}"):
            self.terminal.copy_to_container(
                paths=paths,
                container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
            )

        # Choose session per config
        print("running tests in a new shell")
        with trace_scope(f"evaluate_completion_sync.create_test_session:{self.task_name}, traj:{self.traj_i}"):
            test_session = self.terminal.create_session("tests", is_active_stream=False, as_configured_user=False)

        # Execute tests
        test_script_path = str(DockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh")
        try:
            with trace_scope(f"evaluate_completion_sync.run_tests:{self.task_name}, traj:{self.traj_i}"):
                test_session.send_keys(
                    [f"bash {test_script_path}", "Enter"],
                    block=True,
                    max_timeout_sec=min(self.task_timeouts._evaluate_completion_sync, 4*self.trial_handler.task.max_test_timeout_sec),
                )
            test_output = test_session.capture_pane(capture_entire=True)
            parser_results = self.parser.parse(test_output)

            all_passed = parser_results and all(status == UnitTestStatus.PASSED for status in parser_results.values())
            pass_ratio = (
                sum(1 for status in parser_results.values() if status == UnitTestStatus.PASSED) / len(parser_results)
                if parser_results else 0.0
            )
            # save the test results to output path as a json file
            results_path = str(self.trial_handler.trial_paths.sessions_path.parent / "test_results.json")
            result_dict = {
                "test_results": {k: (v==UnitTestStatus.PASSED) for k, v in parser_results.items()},
                "all_passed": all_passed,
                "pass_ratio": pass_ratio,
            }
            try:
                result_dict["iteration"] = len(self.response.info['tool_calls'])
                result_dict.update(self.response.info['usage'])
            except Exception:
                pass
            with open(results_path, "w") as f:
                json.dump(result_dict, f, indent=4)

        except Exception:
            all_passed = False
            pass_ratio = None

        return pass_ratio



class CamelRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        n_trajs: int = 1,
        max_tokens: int = 32768,
        max_iteration: int = 50,
        max_workers: int = 25,
        non_think_mode: bool = True,
        task_timeouts: TaskTimeouts = None,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.max_iteration = max_iteration
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        self.n_trajs = n_trajs
        self.non_think_mode = non_think_mode
        self.task_timeouts = task_timeouts or TaskTimeouts()

        # Create a shared ThreadPoolExecutor for all Docker operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def arun_episode(self, engine, data):
        clients = [
            ArealOpenAI(engine=engine, tokenizer=self.tokenizer, tool_call_parser="qwen25")
            for _ in range(self.n_trajs)
        ]
        uids = [uuid.uuid4().hex[:8] for _ in range(self.n_trajs)]
        
        # build images first if not exists
        loop = asyncio.get_running_loop()
        try:
            async with atrace_scope(f"build_docker_image:{data.get('task_name')}", args={"timeout": self.task_timeouts._reset_env}):
                await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor, 
                        partial(build_docker_image, task=data, timeout=self.task_timeouts._reset_env)
                    ),
                    timeout=self.task_timeouts._reset_env+60.0  # extra buffer
                )
        except asyncio.TimeoutError:
            print(f"Timeout while building docker image for task {data.get('task_name')}")
            return None

        print(f"\n{'='*70}")
        print(f"[EPISODE START] Task {data.get('task_name')}")
        print(f"{'='*70}\n")
        
        # Collect trajectories
        rewards = await asyncio.gather(
            *[
                CamelTerminalAgent(
                                    max_tokens_per_turn=self.gconfig.max_new_tokens,
                                    max_total_tokens=self.max_tokens,
                                    max_iteration=self.max_iteration,
                                    output_path=f"{self.dump_dir}/CamelTerminalAgent_Output",
                                    executor=self.executor,
                                    non_think_mode=self.non_think_mode,
                                    task_timeouts=self.task_timeouts,
                                ).run_agent(
                                    data=data,
                                    client=clients[i],
                                    uid=uids[i],
                                    traj_i=i,
                                    )
                for i in range(self.n_trajs)
            ]
        )
        
        # Verify workers released
        print(f"\n{'='*70}")
        print(f"[EPISODE END] Task {data.get('task_name')}")
        print(f"{'='*70}\n")
        
        completions_with_reward = {}
        for i, (reward, client) in enumerate(zip(rewards, clients)):
            if reward is None:
                print(f"Rank {os.getenv('RANK')} - Task {data.get('task_name')}, Trajectory {i} failed.")
                os.makedirs(f"{self.dump_dir}/failed_tasks", exist_ok=True)
                with open(f"{self.dump_dir}/failed_tasks/{data.get('task_name')}_traj_{i}.txt", "w") as f:
                    f.write(f"Task {data.get('task_name')} trajectory {i} failed.\n")
                continue
            print(f"Rank {os.getenv('RANK')} - Task {data.get('task_name')}, Trajectory {i} reward: {reward}")
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
            client.apply_reward_discount(turn_discount=0.9)
            completions = client.export_interactions(style="individual")
            completions_with_reward.update(completions)
        
        # record number of full passes and number of trajectories failed for this task
        stats_tracker.get(self.rollout_stat_scope).scalar(num_full_passes=sum(1 for r in rewards if r == 1.0))
        stats_tracker.get(self.rollout_stat_scope).scalar(num_trajectories_failed=sum(1 for r in rewards if r is None))
        if len(completions_with_reward) == 0:
            print(f"All trajectories failed for task {data.get('task_name')}.")
            completions_with_reward = None

        print(f"Rank {os.getenv('RANK')} - Task {data.get('task_name')} completed.")

        return completions_with_reward

def main(args):
    # python -u test_tasks_trace.py --config configs/config_test_docker_tbench_ds.yaml &> test_task.log
    # python -u test_tasks_trace.py --config configs/config_test_tasks_qwen3-8b.yaml &> test_task.log
    config, _ = load_expr_config(args, AgentRLConfig)
    config: AgentRLConfig
    os.environ["AREAL_LLM_SERVER_ADDRS"] = "127.0.1.1:21343"

    rank = int(os.getenv("RANK") or "0")    
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    dump_dir = os.path.abspath(f"{config.stats_logger.fileroot}/{config.stats_logger.experiment_name}/{config.stats_logger.trial_name}/logs/generated")
    print(f"Dump directory: {dump_dir}")
    seeding.set_random_seed(config.seed, key=f"trainer")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    
    # Initialize wandb (if not already initialized)
    use_wandb = not (config.stats_logger.wandb.mode is 'disabled')
    print(f"Using wandb: {use_wandb}")
    if use_wandb and wandb.run is None:
        wandb.init(
            project=config.stats_logger.experiment_name,
            name=config.stats_logger.trial_name,
            config=vars(config),
        )
    
    if config.perf_tracer is not None:
        perf_tracer.configure(config.perf_tracer, rank=rank)

    # Create dataset and dataloaders
    dataset = load_dataset(
        path="parquet",
        split="train",
        data_files=[str(Path(__file__).resolve().parent.parent.parent / "dataset" / config.train_dataset.path)],
    )

    # randomly shuffle the dataset
    # dataset = dataset.shuffle(seed=config.seed)
    # only select dataset with task_name in task_name_list
    # task_name_list = ['distribution-search',"hf-train-lora-adapter", "leelachess0-pytorch-conversion", "play-lord", "install-windows-3-11"
    #                   "path-tracing-reverse", "feal-linear-cryptanalysis","ode-solver-rk4", "vul-flask", "model-extraction-relu-logits"]
    # TODO: schedule-vacation, train-bpe always failes under 8 x 8 setting! why test out!
    # with open("/root/terminal_agent/src/tbench_areal_workflow/outputs/areal/experiments/camel-terminal_agent-grpo/trial0-config_test_tasks_qwen3-8b_docker_test/logs/generated4/all_failed_tasks.json","r") as f:
    #     failed_tasks = json.load(f)
    # task_name_list = [k for k,v in failed_tasks.items()] + ['schedule-vacation', 'train-bpe-tokenizer']
    # dataset = dataset.filter(lambda example: example['task_name'] in task_name_list)
    train_dataloader = create_dataloader(
        dataset,
        rank=rank,
        world_size=1,
        dataset_config=config.train_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )
    stats_logger = StatsLogger(config, ft_spec)

    # Initialize inference engine
    # config.max_head_offpolicyness = int(1e12)
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize()

    workflow = CamelRLVRWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        n_trajs=config.n_trajs,
        max_tokens=config.max_tokens_per_trajectory,
        dump_dir=dump_dir,
        max_iteration=config.max_iteration,
        max_workers=config.max_workers,
        non_think_mode=config.non_think_mode,
        task_timeouts=config.task_timeouts,
    )

    start_step = 0

    total_epochs = config.total_train_epochs   # number of tests for each task
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    print(f"batch size: {config.train_dataset.batch_size}, steps per epoch: {steps_per_epoch}, total steps: {max_steps}")

    data_generator = cycle_dataloader(train_dataloader)

    # Start periodic test result collection thread
    stop_collection_event = threading.Event()
    collection_thread = threading.Thread(
        target=periodic_test_collection,
        args=(dump_dir, stop_collection_event, 300, use_wandb),  # Pass use_wandb flag
        daemon=True
    )
    collection_thread.start()
    print("Started periodic test collection thread (every 5 minutes)")

    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        start_time = time.time()
        print("\n\n=================================")
        print(f"Rank {os.getenv('RANK')} starting epoch {epoch} step {step}.")
        print("=================================\n\n")
        print("Preparing batch asynchronously.")
        with stats_tracker.record_timing("rollout"):
            batch = rollout.prepare_batch(
                train_dataloader,
                workflow=workflow,
            )
        rollout.set_version(global_step + 1)
        perf_tracer.save(step=global_step)
        end_time = time.time()
        print("\n\n=================================")
        print(f"Rank {os.getenv('RANK')} finished epoch {epoch} step {step} in {(end_time - start_time)/60:.2f} minutes.")
        print("=================================\n\n")

        # Upload statistics to the logger (e.g., wandb)
        stats = stats_tracker.export_all()
        stats_logger.commit(epoch, step, global_step, stats)

    stats_logger.close()
    rollout.destroy()
    perf_tracer.save(force=True)

    # Stop the periodic collection thread
    print("Stopping periodic test collection thread...")
    stop_collection_event.set()
    collection_thread.join(timeout=10)
    
    # Perform final collection
    print("Performing final test results collection...")
    try:
        collect_test_results(dump_dir, log_to_wandb=use_wandb)
    except Exception as e:
        print(f"Error in final test collection: {e}")
    
    # Finish wandb run
    if use_wandb and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main(sys.argv[1:])