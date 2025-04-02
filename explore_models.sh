#!/bin/bash

# Script to run all Python files in the model_exploration directory
# Place this script in the root of your Machine-Learning-Challenge/ directory

# --- Configuration ---
PROJECT_ROOT=$(pwd) # Assumes script is run from the project root
TARGET_DIR="model_exploration"
LOG_FILE="model_exploration_output.log"
PYTHON_CMD="python3" # Use python3; ensure it points to your venv python if needed

# --- Safety Check ---
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory '$TARGET_DIR' not found in current directory ($PROJECT_ROOT)." >&2
    echo "Please run this script from the root of the Machine-Learning-Challenge project." >&2
    exit 1
fi

# --- Initialization ---
# Create/clear the log file and add header
echo "========================================" > "$LOG_FILE"
echo "    MODEL EXPLORATION EXECUTION LOG     " >> "$LOG_FILE"
echo "    Timestamp: $(date '+%a %b %d %H:%M:%S %Z %Y')" >> "$LOG_FILE" # Using format from example output
echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Also print header info to console
echo "========================================"
echo "    MODEL EXPLORATION EXECUTION LOG     "
echo "    Timestamp: $(date '+%a %b %d %H:%M:%S %Z %Y')"
echo "========================================"
echo ""
echo "Starting script execution."
echo "Target Directory: $PROJECT_ROOT/$TARGET_DIR" | tee -a "$LOG_FILE"
echo "Python Command: $($PYTHON_CMD --version 2>&1)" | tee -a "$LOG_FILE" # Show python version being used
echo "Logging output to: $LOG_FILE"
echo "" | tee -a "$LOG_FILE"


# --- Find and Execute Scripts ---
# Use find to locate .py files within the target directory, excluding __init__.py
# Use -print0 and read -d '' for safe handling of filenames with spaces/special chars
find "$TARGET_DIR" -type f -name "*.py" -not -name "__init__.py" -print0 | while IFS= read -r -d $'\0' script_path; do

    script_dir=$(dirname "$script_path")
    script_name=$(basename "$script_path")

    # Log the script being run to console and file
    echo "--- Running: $script_path ---" | tee -a "$LOG_FILE"
    echo "Executing $PYTHON_CMD in directory '$PROJECT_ROOT' for script '$script_path'..." >> "$LOG_FILE"

    # Execute the python script
    # Redirect both stdout and stderr to the log file (appending)
    # Capture the exit code ($?) immediately after execution
    # Run the script from the project root directory ($PROJECT_ROOT)
    "$PYTHON_CMD" "$script_path" >> "$LOG_FILE" 2>&1
    exit_code=$?

    # Log the completion status and exit code to console and file
    echo "--- Finished: $script_path (Exit Code: $exit_code) ---" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE" # Add a blank line for readability

done

echo ""
echo "Execution finished. Full log saved to: $LOG_FILE"

exit 0
