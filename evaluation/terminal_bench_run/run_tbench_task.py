# Run terminal bench task using a single agent

# 1. select a task from the terminal bench tasks
from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.terminal import Terminal, spin_up_terminal
from terminal_bench.harness.models import FailureMode
from terminal_bench.parsers.parser_factory import ParserFactory, ParserName
from terminal_bench.parsers.base_parser import UnitTestStatus
from pathlib import Path
from strip_ansi import clean_text_file, clean_and_display_log
from typing import Tuple
from utils import (
    MetadataCollector, 
    create_timestamped_marker_from_memory, 
    create_test_results_dict, 
    process_test_parser_results, 
    get_image_size
)

# ==========================================================================================
#
#               Set up the task 
# ==========================================================================================
import argparse
import time
import json
import docker
parser = argparse.ArgumentParser(description="Run a TerminalBench task with a single agent.")
parser.add_argument("-t", "--task", default="play-zork", help="Task name from tbench-tasks")
parser.add_argument("-p", "--path_to_task", default=None, help="Path to the task directory (not used in this script)")
parser.add_argument("-a", "--attempt", type=int, default=1, help="Attempt number")
parser.add_argument("-n", "--n_attempts", type=int, default=1, help="Total number of attempts")
parser.add_argument("-r", "--run_id", default="test_run", help="Run identifier")
parser.add_argument("-w", "--workforce", action="store_true", help="Use workforce agent (not used in this script)")
parser.add_argument("-m", "--max_iteration", type=int, default=50, help="Maximum steps for the agent")
parser.add_argument("-b", "--backend", default="gpt", help="Model backend to use (not used in this script)")
args = parser.parse_args()

task_name = args.task; attempt = args.attempt; _n_attempts = args.n_attempts; _run_id = args.run_id

# Construct output paths
if args.path_to_task:
    task_path = Path(args.path_to_task).resolve()
    output_path = Path(__file__).parent.resolve() / Path() / Path("output") / f"{_run_id}"
    trial_name = f"{task_path.name}.{attempt}-of-{_n_attempts}.{_run_id}"
else:
    output_path = Path(__file__).parent.resolve() / Path() / Path("output") / f"{_run_id}"
    task_path = Path(__file__).parent.parent.parent / "dataset" / "tbench-tasks" / task_name
    trial_name = f"{task_path.name}.{attempt}-of-{_n_attempts}.{_run_id}"

print(f"Using task path: {task_path}")
print(f"Output path: {output_path}")
# Check if the task path exists
if not task_path.exists():
    print(f"Task path {task_path} does not exist. Please check the path.")


trial_handler = TrialHandler(
    trial_name=trial_name,
    input_path=task_path,
    output_path=output_path,
)

task_instruction = trial_handler.instruction
print(f"Task instruction: {task_instruction}")
working_dir = trial_handler.trial_paths.sessions_path.parent / "CAMEL_WORKDIR"
import os
os.environ["CAMEL_WORKDIR"] = str(working_dir)
print(f"Set CAMEL_WORKDIR to: {os.environ['CAMEL_WORKDIR']}")

os.environ['CAMEL_MODEL_LOG_ENABLED'] = "True"
os.environ['CAMEL_LOG_DIR'] = str(trial_handler.trial_paths.sessions_path.parent / "CAMEL_LOG_DIR")

# Initialize metadata tracking
metadata_collector = MetadataCollector(
    task_name=task_name,
    attempt=attempt,
    run_id=_run_id,
    backend=args.backend,
    max_iteration=args.max_iteration,
    workforce=args.workforce
)
# ==========================================================================================
#
#               create chat agent with terminal toolkit docker
# ==========================================================================================
# 4. Run the agent
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit
from camel.types import ModelPlatformType, ModelType


with spin_up_terminal(
        client_container_name=trial_handler.client_container_name,
        client_image_name=trial_handler.client_image_name,
        docker_image_name_prefix=trial_handler.docker_image_name_prefix,
        docker_compose_path=trial_handler.task_paths.docker_compose_path,
        sessions_logs_path=trial_handler.trial_paths.sessions_path,
        agent_logs_path=trial_handler.trial_paths.agent_logging_dir,
        commands_path=trial_handler.trial_paths.commands_path,
        no_rebuild=True,
        cleanup=False,
        livestream=False,
        disable_recording=False,
        ) as terminal:
    # 3. Create chat agent and run the task, pass in the terminal_toolkit_docker, and container name
    session = terminal.create_session(
        "agent", is_active_stream=False, as_configured_user=True
    )


    # ==========================================================================================
    #
    #               create terminal toolkit instance and check basic functionality
    # ==========================================================================================


    terminal_toolkit_kwargs = {
        'timeout': 20.0,
        'working_directory': "/app",
        'use_docker_backend': True,
        'docker_container_name': trial_handler.client_container_name,
        'session_logs_dir': trial_handler.trial_paths.sessions_path / 'session_logs',
        'safe_mode': False,
    }

    # terminal_toolkit = TerminalToolkit(**terminal_toolkit_kwargs)
    # output = terminal_toolkit.shell_exec(id = "agent", command="echo 'Hello, World!'")
    # print(f"Output of echo command: {output}")

    # del terminal_toolkit


    # ==========================================================================================
    #
    #               Run single chat agent single step
    # ==========================================================================================
    from eigent_simple import developer_agent_factory
    from eigent_simple import main as eigent_main

    if not args.workforce:  # Run single agent
        # Create a single model backend for all agents

        if args.backend == "gpt":
            model_backend_reason = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4_1,
                model_config_dict={
                    "stream": False,
                },
            )
        elif args.backend == "qwen":
            model_backend_reason = ModelFactory.create(
                model_platform=ModelPlatformType.LMSTUDIO,
                model_type="qwen3-8b",
                model_config_dict={
                    "stream": False,
                },
                url = "http://localhost:8000/v1",
            )
        task_id = 'workforce_task'
        camel_agent = developer_agent_factory(
            model_backend_reason,
            task_id,
            terminal_toolkit_kwargs,
            system="Linux (in Docker)",
            machine="x86_64",
            is_workforce=False,
            working_directory=working_dir,
        )
        camel_agent.reset()
        camel_agent.max_iteration = args.max_iteration

        # Define a user message for creating logs directory
        usr_msg = (
            f"Task instruction: {task_instruction}"
        )
        print(f"User message: {usr_msg}")
        # Get response information
        response = camel_agent.step(usr_msg)
        print(str(response.info['tool_calls'])[:1000])

        usage = response.info['usage']
        print(f"Prompt tokens: {usage['prompt_tokens']}, completion tokens: {usage['completion_tokens']}")
        
        # Update metadata with token usage
        metadata_collector.update_token_usage(usage)

        memory_list = camel_agent._memory._chat_history_block.storage.memory_list
        timestamped_markers = create_timestamped_marker_from_memory(memory_list)
        print("\n".join(f"{ts}: {msg}" for ts, msg in timestamped_markers))
    else:   # Run workforce agent
        import asyncio

        asyncio.run(eigent_main(
            task_instruction=task_instruction,
            terminal_toolkit_kwargs=terminal_toolkit_kwargs,
            logdir=f"{str(output_path)}/{trial_name}/",
            system="Linux (in Docker)",
            machine="x86_64",
            working_directory=working_dir,
        ))
        
        # For workforce mode, we'll need to extract metrics differently
        # This is a placeholder - you may need to modify eigent_main to return metrics
        metadata_collector.metadata["turns_taken"] = 1  # Placeholder for workforce mode

    # ==========================================================================================
    #
    #               Test the results in the container
    # ==========================================================================================

    print("Run test script inside the container...")
    # ==========================================================================================
    # 5. Copy the test script to the container and run it

    # Start timing the test execution
    metadata_collector.start_test_timing()

    session = terminal.create_session(
                        "tests", is_active_stream=False, as_configured_user=False
                )
    def _setup_test_env(terminal: Terminal, trial_handler: TrialHandler) -> None:
        paths = [
            trial_handler.task_paths.run_tests_path,
        ]

        if trial_handler.task_paths.test_dir.exists():
            paths.append(trial_handler.task_paths.test_dir)

        terminal.copy_to_container(
            paths=paths,
            container_dir=str(terminal._compose_manager.CONTAINER_TEST_DIR),
        )

    def _run_tests(
        terminal: Terminal,
        session,
        trial_handler: TrialHandler,
    ) -> Tuple[FailureMode, dict]:
        _setup_test_env(terminal, trial_handler)

        _global_test_timeout_sec = None
        if _global_test_timeout_sec:
            test_timeout_sec = _global_test_timeout_sec
        else:
            test_timeout_sec = (
                trial_handler.task.max_test_timeout_sec
                * 1.0
            )

        test_results = create_test_results_dict()

        try:
            session.send_keys(
                [
                    "bash ",
                    str(
                        terminal._compose_manager.CONTAINER_TEST_DIR
                        / trial_handler.task_paths.run_tests_path.name
                    ),
                    "Enter",
                ],
                block=True,
                max_timeout_sec=test_timeout_sec,
            )
            test_output = session.capture_pane(capture_entire=True)
            parser = ParserFactory.get_parser(
                parser_name=ParserName.PYTEST,
            )
            parser_results = parser.parse(test_output)
            test_results = process_test_parser_results(parser_results)

        except TimeoutError:
            print(
                "Test command timed out after "
                f"{test_timeout_sec}s for task "
                f"{trial_handler.task_id}."
            )

            return FailureMode.TEST_TIMEOUT, test_results

        return FailureMode.NONE, test_results


    test_failure_mode, test_results = _run_tests(
        terminal=terminal,
        session=session,
        trial_handler=trial_handler,
    )
    
    # Update metadata with test results
    metadata_collector.update_test_results(test_failure_mode, test_results)

    # ==========================================================================================
    # TODO: Copy out the test results/ camel logs from container to local machine

    # ==========================================================================================
    #
    #               Strip ansi from the test log and display test session results
    # ==========================================================================================

    testlog_path = trial_handler.trial_paths.sessions_path / "tests.log"
    testlog_strip_path = trial_handler.trial_paths.sessions_path / "tests.log.strip"
    clean_text_file(str(testlog_path), str(testlog_strip_path))
    print(f"Cleaned test log saved to: {testlog_strip_path}")
    clean_and_display_log(str(testlog_strip_path))

    # Finalize metadata collection
    metadata = metadata_collector.finalize(trial_handler.client_container_name, task_name)
    
    # Save metadata to JSON file
    metadata_path = trial_handler.trial_paths.sessions_path / "metadata.json"
    metadata_collector.save_to_file(metadata_path)
    
    print(f"Metadata saved to: {metadata_path}")
    metadata_collector.print_summary()
    
# Final metadata collection outside the terminal context
print(f"\nRun completed. Metadata saved for task: {task_name}")