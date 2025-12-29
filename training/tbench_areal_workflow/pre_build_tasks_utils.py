from pathlib import Path

# --- Terminal Bench import
from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager



def build_docker_image(task: dict, timeout=1200.0):

    task_path = Path(__file__).parent.parent.parent / "dataset" / task.get("task_path")
    trial_handler = TrialHandler(
        trial_name=f"build_run",
        input_path=task_path,
        output_path=Path("build_outputs"),
    )
    print(f"Task path: {task_path}")

    compose_manager = DockerComposeManager(
            client_container_name=trial_handler.client_container_name,
            client_image_name=trial_handler.client_image_name,
            docker_image_name_prefix=trial_handler.docker_image_name_prefix,
            docker_compose_path=trial_handler.task_paths.docker_compose_path,
            no_rebuild=True,
            cleanup=True,
            sessions_logs_path=trial_handler.trial_paths.sessions_path,
            agent_logs_path=trial_handler.trial_paths.agent_logging_dir,
        )
    compose_manager.build(timeout=timeout)
    