"""
Utility functions for metadata collection and container management in terminal bench tasks.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from terminal_bench.harness.models import FailureMode
from terminal_bench.parsers.base_parser import UnitTestStatus


class MetadataCollector:
    """Handles metadata collection and management for terminal bench runs."""
    
    def __init__(self, task_name: str, attempt: int, run_id: str, backend: str, 
                 max_iteration: int, workforce: bool):
        """Initialize metadata collector with basic run information."""
        self.metadata = {
            "task_name": task_name,
            "attempt": attempt,
            "run_id": run_id,
            "backend": backend,
            "max_iteration": max_iteration,
            "workforce": workforce,
            "start_time": time.time(),
            "total_tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "turns_taken": 0,
            "test_failure_mode": None,
            "execution_success": False,
            "test_results": None,
            "container_size_mb": 0,
            "image_size_mb": 0,
            "test_time_seconds": 0,
        }
        self.test_start_time = None
    
    def update_token_usage(self, usage: Dict[str, int]) -> None:
        """Update token usage metadata."""
        self.metadata["total_tokens"]["prompt_tokens"] += usage.get('prompt_tokens', 0)
        self.metadata["total_tokens"]["completion_tokens"] += usage.get('completion_tokens', 0)
        self.metadata["total_tokens"]["total_tokens"] = (
            self.metadata["total_tokens"]["prompt_tokens"] + 
            self.metadata["total_tokens"]["completion_tokens"]
        )
        self.metadata["turns_taken"] += 1
    
    def start_test_timing(self) -> None:
        """Start timing the test execution."""
        self.test_start_time = time.time()
        print(f"Started test timing at: {self.test_start_time}")
    
    def update_test_results(self, test_failure_mode: FailureMode, test_results: Dict[str, Any]) -> None:
        """Update test results metadata and calculate test duration."""
        # Calculate test duration if timing was started
        if self.test_start_time is not None:
            test_end_time = time.time()
            self.metadata["test_time_seconds"] = test_end_time - self.test_start_time
            print(f"Test duration: {self.metadata['test_time_seconds']:.2f} seconds")
        
        self.metadata["test_failure_mode"] = (
            test_failure_mode.value if hasattr(test_failure_mode, 'value') else str(test_failure_mode)
        )
        self.metadata["execution_success"] = test_failure_mode == FailureMode.NONE
        self.metadata["test_results"] = test_results
    
    def finalize(self, container_name: str, image_name: str) -> Dict[str, Any]:
        """Finalize metadata collection and return the complete metadata."""
        self.metadata["end_time"] = time.time()
        self.metadata["total_time_seconds"] = self.metadata["end_time"] - self.metadata["start_time"]
        self.metadata["container_size_mb"] = get_container_size(container_name)
        self.metadata["image_size_mb"] = get_image_size(image_name)
        
        # Remove start_time and end_time from final metadata (keep only duration)
        del self.metadata["start_time"]
        del self.metadata["end_time"]
        
        return self.metadata
    
    def save_to_file(self, file_path: Path) -> None:
        """Save metadata to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def print_summary(self) -> None:
        """Print a summary of the metadata."""
        print("Metadata summary:")
        print(f"  - Total time: {self.metadata.get('total_time_seconds', 0):.2f} seconds")
        print(f"  - Test time: {self.metadata.get('test_time_seconds', 0):.2f} seconds")
        print(f"  - Total tokens: {self.metadata['total_tokens']['total_tokens']}")
        print(f"  - Turns taken: {self.metadata['turns_taken']}")
        print(f"  - Container size: {self.metadata.get('container_size_mb', 0):.2f} MB")
        print(f"  - Image size: {self.metadata.get('image_size_mb', 0):.2f} MB")
        print(f"  - Execution success: {self.metadata.get('execution_success', False)}")
        print(f"  - Test failure mode: {self.metadata.get('test_failure_mode', 'Unknown')}")
        if self.metadata.get('test_results'):
            test_res = self.metadata['test_results']
            print(f"  - Test pass ratio: {test_res.get('pass_ratio', 0):.2%}")
            print(f"  - Tests passed/total: {test_res.get('passed_tests', 0)}/{test_res.get('total_tests', 0)}")


def get_container_size(container_name: str) -> float:
    """Get the size of a Docker container in MB."""
    try:
        print(f"Getting size for container: {container_name}")
        output = subprocess.check_output(
            ["docker", "system", "df", "-v", "--format", "json"]
        )
        data = json.loads(output.decode())

        for item in data.get("Containers", []):
            if item.get("Names") == container_name:
                size_str = item.get("Size", "0B")
                return parse_size(size_str)

        return 0
    except Exception as e:
        print(f"Error getting container size: {e}")
        return 0
      
def get_image_size(image_name: str) -> float:
    """Get the size of a Docker image in MB."""
    try:
        print(f"Getting size for image: {image_name}")
        output = subprocess.check_output(
            ["docker", "system", "df", "-v", "--format", "json"]
        )
        data = json.loads(output.decode())

        for item in data.get("Images", []):
            if item.get("Repository") == ("tb__" + image_name + "__client"):
                size_str = item.get("VirtualSize", "0B")
                return parse_size(size_str)

        return 0
    except Exception as e:
        print(f"Error getting image size: {e}")
        return 0


def parse_size(size_str: str) -> float:
    """Parse a size string (e.g., '1.2GB', '504MB') and return size in MB using decimal units (1000-based)."""
    if not size_str or size_str == "0" or size_str == "0B":
        return 0.0
    
    # Handle different unit formats
    size_str = size_str.strip()
    
    # Extract number and unit
    import re
    match = re.match(r'([\d.]+)\s*([A-Za-z]*)', size_str)
    if not match:
        return 0.0
    
    number_str, unit = match.groups()
    try:
        number = float(number_str)
    except ValueError:
        return 0.0
    
    # Convert to MB using decimal units (1000-based) - consistent with Docker's reporting
    unit = unit.upper()
    if unit in ['B', 'BYTE', 'BYTES']:
        return number / (1000 * 1000)  # bytes to MB
    elif unit in ['KB', 'K']:
        return number / 1000  # KB to MB
    elif unit in ['MB', 'M']:
        return number  # already in MB
    elif unit in ['GB', 'G']:
        return number * 1000  # GB to MB
    elif unit in ['TB', 'T']:
        return number * 1000 * 1000  # TB to MB
    else:
        # Default to bytes if no unit specified
        return number / (1000 * 1000)


def create_timestamped_marker_from_memory(records: List[dict]) -> List[Tuple[float, str]]:
    """Create timestamped markers from memory records."""
    results = []
    print(f"Total records: {len(records)}")
    for record in records:
        if 'func_name' in record['message'].keys():
            timestamp = record['timestamp']
            func_name = record['message']['func_name']
            args = record['message'].get('args', {})
            if args:
                command = args.get('command', '')
            else:
                command = ''
            results.append((timestamp, f"Called tool: {func_name} with args: {command}"))
    return results


def create_test_results_dict() -> Dict[str, Any]:
    """Create an empty test results dictionary with default structure."""
    return {
        "parser_results": {},
        "pass_ratio": 0.0,
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0
    }


def process_test_parser_results(parser_results: Dict[str, Any]) -> Dict[str, Any]:
    """Process test parser results and return structured test results."""
    test_results = create_test_results_dict()
    
    print("Test Results:")
    
    # Convert parser results to serializable format
    serializable_results = {}
    for test_name, status in parser_results.items():
        print(f"  {test_name}: {status.name}")
        serializable_results[test_name] = status.name
    
    pass_ratio = (
        sum(1 for status in parser_results.values() if status == UnitTestStatus.PASSED) / len(parser_results)
        if parser_results else 0.0
    )
    print(f"Overall Pass Ratio: {pass_ratio:.2%}")
    
    # Update test results
    test_results["parser_results"] = serializable_results
    test_results["pass_ratio"] = pass_ratio
    test_results["total_tests"] = len(parser_results)
    test_results["passed_tests"] = sum(1 for status in parser_results.values() if status == UnitTestStatus.PASSED)
    test_results["failed_tests"] = test_results["total_tests"] - test_results["passed_tests"]
    
    return test_results