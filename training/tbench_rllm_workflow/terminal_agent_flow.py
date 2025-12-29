import logging
from pathlib import Path

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

from rollout_engine_model import rLLMEngineModel
from eigent_simple import developer_agent_factory

logger = logging.getLogger(__name__)

# TODO: 
# 1. adapt the verl rollout engine or other rollout engine to work with camel chatagent
# 2. extract trajectory and termination_reason from agent step
# 3. extract complete test results of each sub-test for metrics

class TerminalAgentWorkflow(Workflow):
    def __init__(
        self,
        rollout_engine,
        executor,
        **kwargs,
    ):
        # rollout_engine: the rollout engine, will be provided VerlEngine if using AgentTrainer, specifically AgentWorkflowPPOTrainer
        #               otherwise need to be provided by user
        # executor: thread/process pool executor for sync calls, will be provided if using AgentWorkflowEngine

        super().__init__(rollout_engine=rollout_engine, **kwargs)
        
        self.session = None
        self.agent   = None

        self.rollout_engine = rollout_engine

        self.output_path = kwargs.get("output_path", Path("/tmp/rllm_terminal_bench_output"))
        self.max_steps = kwargs.get("max_steps", 50)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Reset, run TerminalAgent to completion, evaluate, and package an Episode.
        
        task: dict
            "task_name": task.task_name,
            "task_path": str(task.task_path),
            "instruction": task.instruction,
            "test_weights": task.test_weights,
            "dockerfile_contents": task.dockerfile_contents,
            "py_test_file_contents": task.py_test_file_contents,
            "max_test_timeout_sec": task.max_test_timeout_sec,
        """
        task_id = task.get("task_id", "unknown")
        logger.info("Starting terminal agent run task_id=%s uid=%s", task_id, uid)
        cleanup_required = False

        # 1. Reset docker environment, create session, and reset chat agent
        try:
            observation, info = await self.run_in_executor(self._reset_env, task=task, uid=uid)
            cleanup_required = True
        except Exception:
            logger.exception("Failed to reset environment task_id=%s uid=%s", task_id, uid)
            raise

        prompt = observation["prompt"]
        assert self.session is not None and self.agent is not None

        # 2. Run agent step and return with trajectory, and termination_reason: whether it hit max steps or complete running
        try:
            # response = self.agent.astep(prompt)
            response = await asyncio.wait_for(
                self.agent.astep(prompt),
                timeout=self.global_agent_timeout_sec,
            )

        except Exception:
            logger.exception("Agent step failed task_id=%s uid=%s", task_id, uid)
            if cleanup_required:
                await self.run_in_executor(self._close_env)
            raise
        termination_reason = TerminationReason.ENV_DONE
        trajectory = self.get_trajectory()

        # 3. Evaluate completion and cleanup
        try:
            reward = await self.run_in_executor(self._evaluate_completion_sync)
        except Exception:
            logger.exception("Evaluation failed task_id=%s uid=%s", task_id, uid)
            raise
        finally:
            await self.run_in_executor(self._close_env)

        logger.info(
            "Terminal agent run completed task_id=%s uid=%s reward=%s",
            task_id,
            uid,
            reward,
        )

        episode = Episode(
                        id=uid, 
                        task=task, 
                        is_correct=bool(reward > 0), 
                        trajectories=[trajectory],
                        termination_reason=termination_reason
                        )
        return episode

    async def _eval_and_terminate(self) -> None:
        try:
            await self.run_in_executor(self._evaluate_completion_sync)
        finally:
            await self.run_in_executor(self._close_env)
        raise TerminationEvent(TerminationReason.ENV_DONE)

    # ------------------------------ Sync helpers ------------------------------
    def _reset_env(self, task: dict, uid: str):
        """Create trial, start containers and session, and build initial prompt."""
        output_path = self.output_path
        output_path.mkdir(parents=True, exist_ok=True)

        task_path = Path(task.get("task_path"))
        instruction = task.get("instruction")
        task_id = task.get("task_id", "unknown")

        self.trial_handler = TrialHandler(
            trial_name=f"{task_id}.{uid}.rllm-run",
            input_path=task_path,
            output_path=output_path,
        )

        task_config = self.trial_handler.task
        self.parser = ParserFactory.get_parser(task_config.parser_name)

        self.terminal = Terminal(
            client_container_name=self.trial_handler.client_container_name,
            client_image_name=self.trial_handler.client_image_name,
            docker_compose_path=self.trial_handler.task_paths.docker_compose_path,
            docker_image_name_prefix=self.trial_handler.docker_image_name_prefix,
            sessions_logs_path=self.trial_handler.trial_paths.sessions_path,
            agent_logs_path=self.trial_handler.trial_paths.agent_logging_dir,
            no_rebuild=self.env_args.get("no_rebuild", False),
            cleanup=self.env_args.get("cleanup", True),
        )
        self.terminal.start()
        self.session = self.terminal.create_session("agent", is_active_stream=False, as_configured_user=True)

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

        self.model_backend_reason = rLLMEngineModel(
            engine=self.rollout_engine,
            model_type="VerlEngine",
        )

        self.agent = developer_agent_factory(
            self.model_backend_reason,       # TODO: need to use rollout engine instead of model_backend_reason
            terminal_toolkit_kwargs,
            system="Linux (in Docker)",
            machine="x86_64",
            is_workforce=False,
            working_directory=working_dir,
        )
        self.agent.reset()
        self.agent.max_iteration = self.max_iteration

        usr_msg = (
            f"Task instruction: {instruction}"
        )
        observation = {"prompt": usr_msg, "type": "initial"}
        info = {
            "task_id": task_id,
            "episode": 0,
            "max_iteration": self.max_iteration,
            "instruction": instruction,
        }
        return observation, info

    def _evaluate_completion_sync(self) -> float:
        """Copy tests, run them, parse output, and return a binary reward."""
        assert self.trial_handler is not None and self.terminal is not None

        # Copy tests into the container
        paths = [self.trial_handler.task_paths.run_tests_path]
        if self.trial_handler.task_paths.test_dir.exists():
            paths.append(self.trial_handler.task_paths.test_dir)
        self.terminal.copy_to_container(
            paths=paths,
            container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
        )

        # Choose session per config
        if self.trial_handler.task.run_tests_in_same_shell:
            print(1)
            test_session = self.session
        else:
            print(2)
            test_session = self.terminal.create_session("tests", is_active_stream=False, as_configured_user=False)

        # Execute tests
        test_script_path = str(DockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh")
        try:
            test_session.send_keys(
                [f"bash {test_script_path}", "Enter"],
                block=True,
                max_timeout_sec=self.trial_handler.task.max_test_timeout_sec,
            )
            test_output = test_session.capture_pane(capture_entire=True)
            parser_results = self.parser.parse(test_output)

            all_passed = parser_results and all(status == UnitTestStatus.PASSED for status in parser_results.values())
        except Exception:
            all_passed = False

        return 1.0 if all_passed else 0.0

    def _close_env(self):
        """Stop/cleanup terminal containers if present."""
        if self.terminal:
            self.terminal.stop()

    def get_trajectory(self) -> Trajectory:
        """Return the trajectory extracted from the agent's memory."""
        # TODO: improve the trajectory extraction later, currently just put it here as a placeholder
        #  it currently has no effects on the main workflow

        trajectory = Trajectory()
        trajectory = [
            Step(
                model_output=m
            )
            for m in self.agent._trajectory
        ]
