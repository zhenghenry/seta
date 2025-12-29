import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import wandb


def collect_test_results(dump_dir, log_to_wandb=False):
    # recursively find results in dump_dir/CamelTerminalAgent_Output/<task_name>/<task_run_id>/test_results.json
    dump_path = Path(dump_dir)
    output_dir = dump_path / "CamelTerminalAgent_Output"
    
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return
    
    # Dictionary to store all results: {task_name: [{run_id, data}, ...]}
    all_results = defaultdict(list)
    
    # Recursively find all test_results.json files
    for test_result_file in output_dir.rglob("test_results.json"):
        try:
            # Parse the path to extract task_name and run_id
            # Expected: CamelTerminalAgent_Output/{task_name}/{task_run_id}/test_results.json
            relative_path = test_result_file.relative_to(output_dir)
            path_parts = relative_path.parts
            
            if len(path_parts) >= 3:
                task_name = path_parts[0]
                task_run_id = path_parts[1]
                
                # Load the JSON data
                with open(test_result_file, 'r') as f:
                    data = json.load(f)
                
                all_results[task_name].append({
                    "run_id": task_run_id,
                    "pass_ratio": data.get("pass_ratio", 0.0),
                    "all_passed": data.get("all_passed", False),
                    "test_results": data.get("test_results", {}),
                })
        except Exception as e:
            print(f"Error processing {test_result_file}: {e}")
            continue
    
    # Collect failed tasks from failed_tasks directory
    failed_tasks_dir = dump_path / "failed_tasks"
    failed_tasks_count = defaultdict(int)
    
    if failed_tasks_dir.exists():
        # Pattern: {task_name}_traj_{i}.txt
        for failed_file in failed_tasks_dir.glob("*_traj_*.txt"):
            try:
                # Parse filename to extract task_name and trajectory index
                filename = failed_file.stem  # Remove .txt extension
                # Split by '_traj_' to separate task_name and trajectory index
                if "_traj_" in filename:
                    task_name = filename.rsplit("_traj_", 1)[0]
                    failed_tasks_count[task_name] += 1
            except Exception as e:
                print(f"Error processing failed task file {failed_file}: {e}")
                continue
    
    # Save failed tasks to JSON
    failed_tasks_json_path = dump_path / "all_failed_tasks.json"
    with open(failed_tasks_json_path, 'w') as f:
        json.dump(dict(failed_tasks_count), f, indent=2)
    print(f"Saved failed tasks to: {failed_tasks_json_path}")
    
    # Save all results to a combined JSON file
    combined_json_path = dump_path / "all_test_results.json"
    with open(combined_json_path, 'w') as f:
        json.dump(dict(all_results), f, indent=2)
    print(f"Saved combined results to: {combined_json_path}")
    
    # Create CSV summary using pandas
    csv_path = dump_path / "test_results_summary.csv"
    
    # Prepare data for DataFrame
    rows = []
    for task_name in sorted(all_results.keys()):
        runs = all_results[task_name]
        # Sort runs by run_id for consistency
        runs_sorted = sorted(runs, key=lambda x: x["run_id"])
        
        row_data = {"task_name": task_name}
        
        # Add pass_ratio for each run
        for i, run in enumerate(runs_sorted):
            row_data[f"run_{i+1}_pass_ratio"] = run["pass_ratio"]
        
        # Count number of perfect runs (pass_ratio == 1.0)
        num_perfect = sum(1 for run in runs_sorted if run["pass_ratio"] == 1.0)
        row_data["num_perfect_runs"] = num_perfect
        
        # Add number of failed runs if any
        if task_name in failed_tasks_count:
            row_data["num_failed_runs"] = failed_tasks_count[task_name]
        
        rows.append(row_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    print(f"Saved CSV summary to: {csv_path}")
    print(f"\nSummary: Found {len(all_results)} tasks with a total of {sum(len(runs) for runs in all_results.values())} test runs")
    if failed_tasks_count:
        print(f"Failed tasks: {sum(failed_tasks_count.values())} failures across {len(failed_tasks_count)} tasks")
    
    # Upload to wandb if enabled
    if log_to_wandb and wandb.run is not None:
        try:
            # Log files as artifacts
            artifact = wandb.Artifact(
                name=f"test_results_{wandb.run.id}",
                type="test_results",
                description="Test results summary"
            )
            artifact.add_file(str(combined_json_path), name="all_test_results.json")
            artifact.add_file(str(csv_path), name="test_results_summary.csv")
            artifact.add_file(str(failed_tasks_json_path), name="all_failed_tasks.json")
            wandb.log_artifact(artifact)
            
            # Log summary metrics
            total_tasks = len(all_results)
            total_runs = sum(len(runs) for runs in all_results.values())
            total_perfect_runs = sum(
                sum(1 for run in runs if run["pass_ratio"] == 1.0)
                for runs in all_results.values()
            )
            avg_pass_ratio = sum(
                sum(run["pass_ratio"] for run in runs)
                for runs in all_results.values()
            ) / total_runs if total_runs > 0 else 0.0
            total_failed_runs = sum(failed_tasks_count.values())
            
            wandb.log({
                "test_results/total_tasks": total_tasks,
                "test_results/total_runs": total_runs,
                "test_results/total_perfect_runs": total_perfect_runs,
                "test_results/avg_pass_ratio": avg_pass_ratio,
                "test_results/total_failed_runs": total_failed_runs,
            })
            
            # Log per-task metrics
            for task_name, runs in all_results.items():
                task_avg_pass_ratio = sum(run["pass_ratio"] for run in runs) / len(runs)
                metrics = {
                    f"test_results/task_{task_name}/avg_pass_ratio": task_avg_pass_ratio,
                    f"test_results/task_{task_name}/num_runs": len(runs),
                }
                if task_name in failed_tasks_count:
                    metrics[f"test_results/task_{task_name}/num_failed_runs"] = failed_tasks_count[task_name]
                wandb.log(metrics)
            
            print("Successfully logged test results to wandb")
        except Exception as e:
            print(f"Error logging to wandb: {e}")


def periodic_test_collection(dump_dir, stop_event, interval_seconds=300, log_to_wandb=False):
    """
    Periodically collect test results in a background thread.
    
    Args:
        dump_dir: Directory to collect results from
        stop_event: threading.Event to signal when to stop
        interval_seconds: Time between collections (default: 300 = 5 minutes)
        log_to_wandb: Whether to upload results to wandb
    """
    while not stop_event.is_set():
        try:
            collect_test_results(dump_dir, log_to_wandb=log_to_wandb)
        except Exception as e:
            print(f"Error in periodic test collection: {e}")
        
        # Wait for interval or until stop_event is set
        stop_event.wait(timeout=interval_seconds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect test results from terminal benchmark outputs.")
    parser.add_argument("--dump_dir", type=str, required=True, help="Directory containing CamelTerminalAgent_Output")
    parser.add_argument("--log_to_wandb", action="store_true", help="Whether to log results to wandb")
    
    args = parser.parse_args()
    
    collect_test_results(args.dump_dir, log_to_wandb=args.log_to_wandb)