import os
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import contextmanager

from harbor.models.trial.result import AgentInfo, ModelInfo
from terminal_bench.agents.base_agent import BaseAgent, AgentResult
from terminal_bench.harness.models import FailureMode
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from evaluation.terminal_bench_run.eigent_simple import developer_agent_factory

logger = logging.getLogger(__name__)

DEFAULT_COMMAND_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 64000
DEFAULT_WORKING_DIR = "/app"
MAX_TIMESTAMPED_MARKERS = 1000


@contextmanager
def timed_operation(operation_name: str):
    """Context manager to time operations."""
    start = time.time()
    logger.info(f"Starting: {operation_name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed: {operation_name} in {elapsed:.2f}s")


class TerminalBenchAgent(BaseAgent):
    """Terminal Bench agent using CAMEL framework with Harbor Docker execution."""

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name")
        env_log_dir = os.getenv("CAMEL_LOG_DIR")
        if env_log_dir and env_log_dir.strip():
            self.logging_dir = env_log_dir
        else:
            default_logs = Path(__file__).resolve().parent / "logs" / "camel_logs"
            self.logging_dir = str(default_logs)
        os.makedirs(self.logging_dir, exist_ok=True)
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "TerminalBenchAgent"

    def version(self) -> str | None:
        return "1.0"

    async def setup(self, environment) -> None:
        """Setup method required by Harbor."""
        pass

    async def run(self, instruction: str, environment, context):
        """Run method required by Harbor."""
        logger.info(f"Starting task with instruction: {instruction[:100]}...")
        
        project_root = Path(os.getcwd())
        jobs_dir = project_root / "jobs"
        trial_root = self._find_trial_root(jobs_dir, project_root)
        
        logger.info(f"Final trial_root: {trial_root}")
        
        # Create directory structure
        camel_workdir = trial_root / "CAMEL_WORKDIR"
        camel_workdir.mkdir(parents=True, exist_ok=True)
        
        session_logs_dir = trial_root / "sessions" / "session_logs"
        session_logs_dir.mkdir(parents=True, exist_ok=True)
        (session_logs_dir / "blocking_commands.log").touch(exist_ok=True)
        
        # Set CAMEL directories per-task
        os.environ["CAMEL_LOG_DIR"] = str(camel_workdir)
        os.environ["CAMEL_WORKDIR"] = str(camel_workdir)
        
        logger.info(f"CAMEL_WORKDIR (notes): {camel_workdir}")
        logger.info(f"Session logs: {session_logs_dir}")
        
        result = await self.perform_task(
            instruction=instruction,
            environment=environment,
            session_logs_dir=str(session_logs_dir),
            camel_workdir=str(camel_workdir),
            trial_root=trial_root,
        )
        
        # Update context with token counts
        if hasattr(context, 'n_input_tokens'):
            context.n_input_tokens = result.total_input_tokens
        if hasattr(context, 'n_output_tokens'):
            context.n_output_tokens = result.total_output_tokens
        if hasattr(context, 'n_cache_tokens') and hasattr(result, 'total_cache_tokens'):
            context.n_cache_tokens = result.total_cache_tokens
        
        return result

    def _find_trial_root(self, jobs_dir: Path, project_root: Path) -> Path:
        """Find the trial root directory for this task."""
        if not jobs_dir.exists():
            logger.warning(f"Jobs directory not found, using project root: {project_root}")
            return project_root
        
        job_dirs = sorted(
            [d for d in jobs_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True
        )
        
        if not job_dirs:
            logger.warning(f"No job directories found, using project root: {project_root}")
            return project_root
        
        current_job_dir = job_dirs[0]
        logger.info(f"Found current job directory: {current_job_dir}")
        
        # Find unprocessed task directory (one without chatagent.log yet)
        existing_tasks = [
            d for d in current_job_dir.iterdir()
            if d.is_dir() and (d / "agent").exists()
        ]
        
        for task_dir in existing_tasks:
            chat_log = task_dir / "chatagent.log"
            if not chat_log.exists():
                logger.info(f"Found unprocessed task directory: {task_dir}")
                return task_dir
        
        if existing_tasks:
            logger.warning(f"All tasks processed, using first: {existing_tasks[0]}")
            return existing_tasks[0]
        
        logger.warning(f"No task directories found, using project root: {project_root}")
        return project_root

    def to_agent_info(self) -> AgentInfo:
        model_name = getattr(self, "model_name", None)
        model_info = None

        if isinstance(model_name, str) and model_name:
            provider, name = (model_name.split("/", 1) if "/" in model_name 
                            else (None, model_name))
            model_info = ModelInfo(name=name, provider=provider or "unknown")

        return AgentInfo(
            name=self.name(),
            version=self.version() or "latest",
            model_info=model_info,
        )

    async def perform_task(
        self,
        instruction: str,
        environment,
        session_logs_dir: str,
        camel_workdir: str,
        trial_root: Path,
    ) -> AgentResult:
        """Execute a task using CAMEL agent with Harbor environment."""
        task_start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        
        chat_log_path = trial_root / "chatagent.log"
        chat_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        def log(msg: str = "", level: int = logging.INFO) -> None:
            """Log to stdout, logger, and chatagent.log."""
            elapsed = time.time() - task_start_time
            timestamped_msg = f"[T+{elapsed:06.1f}s] {msg}"
            print(timestamped_msg, flush=True)
            logger.log(level, timestamped_msg)
            try:
                with chat_log_path.open("a", encoding="utf-8") as f:
                    f.write(timestamped_msg + "\n")
            except Exception:
                pass
        
        # Environment setup
        with timed_operation("environment_setup"):
            os.environ["CAMEL_MODEL_LOG_ENABLED"] = "True"
        
        with timed_operation("model_creation"):
            
            model_backend = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4_1,
                model_config_dict={
                    "stream": False,
                },
            )
        
        log(f"CAMEL_WORKDIR (notes): {camel_workdir}")
        
        # Resolve Docker container
        with timed_operation("container_resolution"):
            container_name = self._resolve_container_name(environment)
            if not container_name:
                container_name = os.getenv("HARBOR_CONTAINER_NAME") or os.getenv("DOCKER_CONTAINER_NAME")
                if container_name:
                    log(f"Using container from env var: {container_name}")
            
            if not container_name:
                msg = "Could not determine Docker container name; refusing to run on host."
                log(msg, level=logging.ERROR)
                return AgentResult(
                    total_input_tokens=0,
                    total_output_tokens=0,
                    failure_mode=FailureMode.AGENT_ERROR,
                    timestamped_markers=[(time.time(), f"Container resolution failed: {msg}")]
                )
        
        # Configure terminal toolkit
        command_timeout = float(os.getenv("TERMINAL_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT)))
        terminal_toolkit_kwargs = {
            'timeout': command_timeout,
            'working_directory': DEFAULT_WORKING_DIR,
            'use_docker_backend': True,
            'docker_container_name': container_name,
            'session_logs_dir': session_logs_dir,
            'safe_mode': False,
        }
        
        log(f"Terminal toolkit: container={container_name}, timeout={command_timeout}s")
        
        camel_agent = None
        try:
            with timed_operation("agent_factory"):
                camel_agent = developer_agent_factory(
                    model_backend,
                    'workforce_task',
                    terminal_toolkit_kwargs,
                    system="Linux (in Docker)",
                    machine="x86_64",
                    is_workforce=False,
                    working_directory=camel_workdir,
                )
                camel_agent.reset()
            
            usr_msg = f"Task instruction: {instruction}"
            log(f"User message: {usr_msg}")
            
            # Execute agent step
            try:
                log("Starting agent step...")
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, camel_agent.step, usr_msg)
                log("Agent step completed successfully")
                
            except KeyboardInterrupt:
                log("Agent task interrupted by user")
                raise
            except Exception as e:
                import traceback
                log(f"Error during agent step: {type(e).__name__}: {e}", level=logging.ERROR)
                log(f"Traceback: {traceback.format_exc()}", level=logging.ERROR)
                
                # Try to extract tokens even on failure
                try:
                    if hasattr(camel_agent, '_memory') and camel_agent._memory is not None:
                        memory_list = camel_agent._memory._chat_history_block.storage.memory_list
                        log(f"Partial execution: {len(memory_list)} memory records before failure")
                except Exception:
                    pass
                
                return AgentResult(
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    failure_mode=FailureMode.AGENT_ERROR,
                    timestamped_markers=[(time.time(), f"Agent error: {str(e)[:200]}")]
                )
            
            # Extract results
            with timed_operation("extract_results"):
                log(str(response.info.get('tool_calls', []))[:1000])
                
                total_input_tokens = response.info.get('usage', {}).get('prompt_tokens', 0)
                total_output_tokens = response.info.get('usage', {}).get('completion_tokens', 0)
                
                memory_list = camel_agent._memory._chat_history_block.storage.memory_list
                
                def create_timestamped_marker_from_memory(records: List[dict]) -> List[Tuple[float, str]]:
                    """Create timestamped markers from memory records."""
                    results = []
                    log(f"Total records: {len(records)}")
                    for record in records:
                        message = record.get('message', {})
                        if 'func_name' in message:
                            timestamp = record.get('timestamp', time.time())
                            func_name = message['func_name']
                            args = message.get('args', {})
                            command = args.get('command', '') if args else ''
                            results.append((timestamp, f"Called tool: {func_name} with args: {command}"))
                    return results
                
                timestamped_markers = create_timestamped_marker_from_memory(memory_list)
                
                # Truncate markers if too many to prevent memory issues
                if len(timestamped_markers) > MAX_TIMESTAMPED_MARKERS:
                    log(f"Warning: Truncating markers from {len(timestamped_markers)} to {MAX_TIMESTAMPED_MARKERS}")
                    timestamped_markers = timestamped_markers[:MAX_TIMESTAMPED_MARKERS]
                
                log(f"Total input tokens: {total_input_tokens}")
                log(f"Total output tokens: {total_output_tokens}")
                log(f"Markers: {len(timestamped_markers)} tool calls")
            
            return AgentResult(
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                failure_mode=FailureMode.NONE,
                timestamped_markers=timestamped_markers,
            )
            
        finally:
            # Clean up the agent instance
            with timed_operation("cleanup"):
                if camel_agent is not None:
                    try:
                        del camel_agent
                        log("Cleaned up camel_agent instance")
                    except Exception as e:
                        logger.error(f"Error cleaning up camel_agent: {e}")
            
            total_time = time.time() - task_start_time
            log(f"Task completed in {total_time:.1f}s")

    def _resolve_container_name(self, environment) -> Optional[str]:
        """Resolve and validate the Harbor Docker container name."""
        session_id = getattr(environment, "session_id", None)
        environment_name = getattr(environment, "environment_name", None)
        candidates = []

        # Check container object attributes
        if hasattr(environment, "container"):
            for attr in ("name", "id", "short_id"):
                val = getattr(environment.container, attr, None)
                if val:
                    candidates.append(val)
            if hasattr(environment.container, "attrs"):
                cid = environment.container.attrs.get("Id")
                if cid:
                    candidates.append(cid)

        # Check environment attributes
        for attr in ("name", "id", "container_name", "container_id"):
            val = getattr(environment, attr, None)
            if val:
                candidates.append(val)

        try:
            env_vars = getattr(environment, "_env_vars", None)
            if env_vars:
                for attr in ("main_image_name", "prebuilt_image_name"):
                    val = getattr(env_vars, attr, None)
                    if val:
                        candidates.append(val)
        except Exception:
            pass

        # Harbor naming patterns: {session_id.lower()}-main-1
        if session_id:
            candidates.append(f"{session_id.lower()}-main-1")
        if environment_name and session_id:
            candidates.append(f"{environment_name}__{session_id.lower()}-main-1")

        seen = set()
        deduped = [c for c in candidates if c and c not in seen and not seen.add(c)]
        
        logger.info(f"Container candidates: {deduped}")

        try:
            import docker
            client = docker.from_env()
            running = [c.name for c in client.containers.list()]
            logger.info(f"Running containers: {running}")
            
            for cand in deduped:
                try:
                    client.containers.get(cand)
                    logger.info(f"Validated container: {cand}")
                    return cand
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Docker validation unavailable: {e}")
        
        return None
