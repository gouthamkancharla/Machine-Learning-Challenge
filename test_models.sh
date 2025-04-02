#!/bin/bash

# Define the root directory to search within
SEARCH_DIR="Machine-Learning-Challenge"

# Define the output file
OUTPUT_FILE="all_scripts_output.txt"

# --- Helper Function to Execute a Single Python Script ---
# This function takes the python script path as its argument ($1)
# It handles changing directory, executing, and logging output/errors.
execute_python_script() {
    local script_path="$1"
    local script_dir
    local script_name
    local exit_code

    # Check if the input is a file
    if [ ! -f "$script_path" ]; then
        echo "--- Error: Not a file: $script_path ---" >> "$OUTPUT_FILE"
        return 1 # Indicate failure
    fi

    script_dir=$(dirname "$script_path")
    script_name=$(basename "$script_path")

    echo "--- Running: $script_path ---" >> "$OUTPUT_FILE"
    echo "Executing python3 in directory '$script_dir' for script '$script_name'..." >> "$OUTPUT_FILE"

    # Execute python script from its own directory in a subshell
    # Redirect stdout and stderr of the python command *within* the subshell
    (cd "$script_dir" && python3 "$script_name") >> "$OUTPUT_FILE" 2>&1
    exit_code=$? # Capture exit code of the subshell/python command

    echo "--- Finished: $script_path (Exit Code: $exit_code) ---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE" # Add a blank line for separation

    # Return the exit code of the python script
    return $exit_code
}

# --- Main Script Logic ---

# Check if the search directory exists
if [ ! -d "$SEARCH_DIR" ]; then
  echo "Error: Directory '$SEARCH_DIR' not found."
  echo "Please run this script from the directory containing '$SEARCH_DIR'."
  exit 1
fi

# Check if python3 command exists
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 command could not be found. Please ensure Python 3 is installed and in your PATH."
    exit 1
fi

# Clear the output file or create it if it doesn't exist
# Also export OUTPUT_FILE so the function can access it in subshells
> "$OUTPUT_FILE"
export OUTPUT_FILE

echo "Starting script execution..."
echo "Output will be saved to: $OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "          SCRIPT EXECUTION LOG          " >> "$OUTPUT_FILE"
echo "      Timestamp: $(date)               " >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Export the function so it's available to the subshells created by find -exec
export -f execute_python_script

# Find all .py files and execute the helper function for each
# Uses 'bash -c' because function exporting is a bash feature.
# The '_' is a placeholder for $0 in the bash -c context.
# "$1" inside the bash -c command refers to the script path passed by find ({}).
find "$SEARCH_DIR" -type f -name "*.py" -exec bash -c 'execute_python_script "$1"' _ {} \;

# Note: The redirection >> "$OUTPUT_FILE" 2>&1 is now handled *inside* the function

echo "========================================" >> "$OUTPUT_FILE"
echo "           ALL SCRIPTS FINISHED         " >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"

echo "Script execution finished."
echo "Output saved to $OUTPUT_FILE"

# Unset the exported variables/functions if desired (optional cleanup)
unset OUTPUT_FILE
unset -f execute_python_script

exit 0
