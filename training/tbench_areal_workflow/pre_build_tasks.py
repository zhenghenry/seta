import sys
from pathlib import Path
from datasets import load_dataset
from areal.api.cli_args import load_expr_config

# --- Terminal Bench import
from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager



from train import AgentRLConfig
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pre_build_tasks_utils import build_docker_image
    

def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)
    config: AgentRLConfig

    # load dataset 
    dataset = load_dataset(
        path="parquet",
        split="train",
        data_files=[str(Path(__file__).resolve().parent.parent.parent / "dataset" / config.train_dataset.path)],
    )

    # loop over dataset with parallelism
    max_workers = min(16, len(dataset), cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(build_docker_image, data) for data in dataset]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task failed with error: {e}")



if __name__ == "__main__":
    main(sys.argv[1:])
