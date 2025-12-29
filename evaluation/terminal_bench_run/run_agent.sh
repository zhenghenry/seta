#!/bin/bash

# This script runs a single agent on a TerminalBench task.
# It allows the user to select a task from a list, activates the
# necessary conda environment, and logs the output of the agent run.

# Exit immediately if a command exits with a non-zero status.
set -e



# --- Configuration ---
RUN_ID="test_run"
ATTEMPT=1
N_ATTEMPTS=2
SCRIPT_TO_RUN="run_tbench_task.py"
TASK_LIST_FILE="task_name_list"
CONDA_ENV_NAME="tagent"
TASK_NAME="hello-world" # Will be set by argument or interactive selection
WORKFORCE_ARG="" # Argument to pass to the python script

# --- Argument Parsing ---
# This loop processes command-line arguments and overrides the default configurations.
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--run-id) RUN_ID="$2"; shift ;;
        -a|--attempt) ATTEMPT="$2"; shift ;;
        -n|--n-attempts) N_ATTEMPTS="$2"; shift ;;
        -s|--script) SCRIPT_TO_RUN="$2"; shift ;;
        -l|--task-list) TASK_LIST_FILE="$2"; shift ;;
        -e|--env-name) CONDA_ENV_NAME="$2"; shift ;;
        -t|--task) TASK_NAME="$2"; shift ;;
        -w|--workforce) WORKFORCE_ARG="--workforce" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# if workforce arg is set, change the run_id to test_workforce
if [[ "$WORKFORCE_ARG" == "--workforce" ]]; then
    RUN_ID="test_workforce"
fi

# --- Task Selection ---
echo "Reading tasks from ${TASK_LIST_FILE}..."
# Read tasks into an array. Using a while loop for portability as mapfile may not be available.
tasks=()
while IFS= read -r line; do
    tasks+=("$line")
done < "$TASK_LIST_FILE"
echo "Please select a task to run:"
# Use the 'select' command to create an interactive menu.
select TASK_NAME in "${tasks[@]}"; do
    if [[ -n "$TASK_NAME" ]]; then
        echo "You selected task: $TASK_NAME"
        break
    else
        echo "Invalid selection. Please try again."
    fi
done

# --- Environment Setup ---
echo "Setting up environment for $CONDA_ENV_NAME..."

# Clean PATH to remove duplicates before conda activation
export PATH=$(echo "$PATH" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//')

# Source conda.sh to make the 'conda' command available to the script.
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found. Please ensure Conda is installed correctly."
    exit 1
fi

# Activate conda environment (suppress error output, it's a conda bug but works anyway)
conda activate "$CONDA_ENV_NAME" 2>&1 | grep -v "ERROR REPORT" | grep -v "TypeError" | grep -v "conda.activate" | grep -v "overwriting" || true
echo "✅ Conda environment activated: $CONDA_ENV_NAME"

# Set CUDA 12.1 paths for SGLang (after conda activation)
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}
echo "✅ CUDA 12.1 paths set"

# --- Log Path and Directory Setup ---
# Construct the output path based on the logic in the Python script.
TRIAL_NAME="${TASK_NAME}.${ATTEMPT}-of-${N_ATTEMPTS}.${RUN_ID}"
OUTPUT_DIR="output/${RUN_ID}/${TASK_NAME}/${TRIAL_NAME}"
LOG_FILE="${OUTPUT_DIR}/chatagent.log"

# Create the output directory if it doesn't exist.
mkdir -p "$OUTPUT_DIR"
echo "Output will be saved to: $LOG_FILE"

# --- Run the Agent ---
echo "Running agent on task '$TASK_NAME'. This may take a while..."
# Execute the python script with the selected task and other parameters.
# Redirect both standard output and standard error to the log file.
# Change to the script's directory to ensure relative paths work correctly.
run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$run_dir" || { echo "Failed to change directory to $run_dir"; exit 1; }

python -u "$script_dir/$SCRIPT_TO_RUN" \
    --task "$TASK_NAME" \
    --run_id "$RUN_ID" \
    --attempt "$ATTEMPT" \
    --n_attempts "$N_ATTEMPTS" \
    $WORKFORCE_ARG > "$script_dir/$LOG_FILE" 2>&1


echo "Agent run finished. Log file created at $script_dir/$LOG_FILE"