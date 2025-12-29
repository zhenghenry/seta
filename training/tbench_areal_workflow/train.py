import asyncio
from functools import partial
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from dataclasses import dataclass, field
from datasets import load_dataset

import platform
import threading
import time
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
from docker_cleanup import start_docker_cleanup, stop_docker_cleanup
from chat_agent_trace import ChatAgentTrace
from pre_build_tasks_utils import build_docker_image

@dataclass
class TaskTimeouts:
    _reset_env: float = 300.0  # seconds
    _reset_agent: float = 120.0  # seconds
    agent_astep: float = 300.0  # seconds
    _evaluate_completion_sync: float = 600.0  # seconds
    _cleanup: float | None = None  # seconds, None means no timeout

@dataclass
class AgentRLConfig(GRPOConfig):
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        },
    )
    max_tokens_per_trajectory: int = field(
        default=32768,
        metadata={
            "help": "Maximum number of tokens per trajectory. By default max_tokens_per_trajectory=32768."
        },
    )
    max_iteration: int = field(
        default=3,
        metadata={
            "help": "Maximum number of iterations for the Camel Terminal Agent. By default max_iteration=50."
        },
    )
    max_workers: int = field(
        default=25,
        metadata={
            "help": "Maximum number of workers for ThreadPoolExecutor. By default max_workers=25."
        },
    )
    non_think_mode: bool = field(
        default=True,
        metadata={
            "help": "Whether to use non-think mode in the developer agent prompt. By default non_think_mode=True."
        },
    )
    async_training: bool = field(
        default=False,
        metadata={
            "help": "Whether to use asynchronous training. By default async_training=False."
        },
    )
    task_timeouts: TaskTimeouts = field(
        default_factory=TaskTimeouts,
        metadata={
            "help": "Timeout settings for various stages of the agent workflow."
        },
    )
    filter_uniform_reward: bool = field(
        default=False,
        metadata={
            "help": "Whether to filter out tasks with uniform rewards across trajectories. By default filter_uniform_reward=False."
        },
    )
    encourage_completion_reward: bool = field(
        default=False,
        metadata={
            "help": "Whether to encourage completion reward in the Camel Terminal Agent. By default encourage_completion_reward=False."
        },
    )


def send_message_to_user(
    message_title: str,
    message_description: str,
    message_attachment: str = "",
) -> str:
    r"""Use this tool to send a tidy message to the user, including a
    short title, a one-sentence description, and an optional attachment.

    This one-way tool keeps the user informed about your progress,
    decisions, or actions. It does not require a response.
    You should use it to:
    - Announce what you are about to do.
      For example:
      message_title="Starting Task"
      message_description="Searching for papers on GUI Agents."
    - Report the result of an action.
      For example:
      message_title="Search Complete"
      message_description="Found 15 relevant papers."
    - Report a created file.
      For example:
      message_title="File Ready"
      message_description="The report is ready for your review."
      message_attachment="report.pdf"
    - State a decision.
      For example:
      message_title="Next Step"
      message_description="Analyzing the top 10 papers."
    - Give a status update during a long-running task.

    Args:
        message_title (str): The title of the message.
        message_description (str): The short description.
        message_attachment (str): The attachment of the message,
            which can be a file path or a URL.

    Returns:
        str: Confirmation that the message was successfully sent.
    """
    print(f"\nAgent Message:\n{message_title} " f"\n{message_description}\n")
    if message_attachment:
        print(message_attachment)

    return (
        f"Message successfully sent to user: '{message_title} "
        f"{message_description} {message_attachment}'"
    )


from concurrent.futures import ThreadPoolExecutor

import subprocess

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
        encourage_completion_reward: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_total_tokens = max_total_tokens
        self.output_path = output_path
        self.max_iteration = max_iteration
        self.task_timeouts = task_timeouts or TaskTimeouts()
        self.executor = executor
        self.non_think_mode = non_think_mode
        self.encourage_completion_reward = encourage_completion_reward
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

        if self.encourage_completion_reward:
            if pass_ratio == 1.0:
                pass_ratio += 1.0  # bonus for full completion 

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
        filter_uniform_reward: bool = False,
        encourage_completion_reward: bool = False,
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
        self.filter_uniform_reward = filter_uniform_reward
        self.encourage_completion_reward = encourage_completion_reward

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
                                    encourage_completion_reward=self.encourage_completion_reward,
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
        if self.filter_uniform_reward:
            valid_rewards = [r for r in rewards if r is not None]
            if valid_rewards and all(r == valid_rewards[0] for r in valid_rewards):
                print(f"Rank {os.getenv('RANK')} - Task {data.get('task_name')} has uniform reward across trajectories. Discarding all.")
                return completions_with_reward
            elif not valid_rewards:
                print(f"Rank {os.getenv('RANK')} - Task {data.get('task_name')} all trajectories failed.")
                return completions_with_reward
            
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
        
        if len(completions_with_reward) == 0:
            print(f"All trajectories failed for task {data.get('task_name')}.")
            completions_with_reward = None

        # record number of full passes and number of trajectories failed for this task
        stats_tracker.get(self.rollout_stat_scope).scalar(num_full_passes=sum(1 for r in rewards if r == 1.0))
        stats_tracker.get(self.rollout_stat_scope).scalar(num_trajectories_failed=sum(1 for r in rewards if r is None))
        
        print(f"Rank {os.getenv('RANK')} - Task {data.get('task_name')} completed.")

        return completions_with_reward

def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)
    config: AgentRLConfig

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    perf_tracer.configure(config.perf_tracer, rank=rank)
    # Create dataset and dataloaders
    dataset = load_dataset(
        path="parquet",
        split="train",
        data_files=[str(Path(__file__).parent.parent.parent / "dataset" / config.train_dataset.path)],
    )
    train_dataloader = create_dataloader(
        dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = CamelRLVRWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        n_trajs=config.n_trajs,
        max_tokens=config.max_tokens_per_trajectory,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        max_iteration=config.max_iteration,
        max_workers=config.max_workers,
        non_think_mode=config.non_think_mode,
        task_timeouts=config.task_timeouts,
        filter_uniform_reward=config.filter_uniform_reward,
        encourage_completion_reward=config.encourage_completion_reward,
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch
    data_generator = cycle_dataloader(train_dataloader)    
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                print("Preparing batch asynchronously.")
                batch = actor.prepare_batch(
                    train_dataloader,
                    workflow=workflow,
                    should_accept_fn = lambda x: (x is not None) and (len(x) > 0),
                    #should_accept=lambda x: (x is not None) and (len(x) > 0),
                )
                print(f"Rank {os.getenv('RANK')} batch prepared.")
            else:
                batch = actor.rollout_batch(
                    next(data_generator),
                    workflow=workflow,
                    should_accept_fn = lambda x: (x is not None) and (len(x) > 0),
                    #should_accept=lambda x: (x is not None) and (len(x) > 0),
                )
            perf_tracer.save(step=global_step, force=False)

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with stats_tracker.record_timing("train_step"):
            actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        perf_tracer.save(force=True)
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # Upload statistics to the logger (e.g., wandb)
        stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_logger.commit(epoch, step, global_step, stats)

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    rollout.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
