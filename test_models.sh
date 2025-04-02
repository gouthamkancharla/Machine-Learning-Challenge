#!/bin/bash

# Define the root directory to search within
# CHANGE: Search from the current directory (.)
SEARCH_DIR="."

# Define the output file (will be created in the current directory)
OUTPUT_FILE="all_models_output.txt"

# --- Helper Function to Execute a Single Python Script ---
# (This function remains the same as before)
execute_python_script() {
    local script_path="$1"
    local script_dir
    local script_name
    local exit_code

    # Clean up path for display if it starts with ./
    local display_path="${script_path#./}"

    # Check if the input is a file
    if [ ! -f "$script_path" ]; then
        echo "--- Error: Not a file: $display_path ---" >> "$OUTPUT_FILE"
        return 1 # Indicate failure
    fi

    script_dir=$(dirname "$script_path")
    script_name=$(basename "$script_path")

    echo "--- Running: $display_path ---" >> "$OUTPUT_FILE"
    echo "Executing python3 in directory '$script_dir' for script '$script_name'..." >> "$OUTPUT_FILE"

    # Execute python script from its own directory in a subshell
    # Redirect stdout and stderr of the python command *within* the subshell
    (cd "$script_dir" && python3 "$script_name") >> "$OUTPUT_FILE" 2>&1
    exit_code=$? # Capture exit code of the subshell/python command

    echo "--- Finished: $display_path (Exit Code: $exit_code) ---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE" # Add a blank line for separation

    # Return the exit code of the python script
    return $exit_code
}

# --- Main Script Logic ---

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
echo "Output will be saved to: $OUTPUT_FILE (in the current directory)"
echo "========================================" >> "$OUTPUT_FILE"
echo "          SCRIPT EXECUTION LOG          " >> "$OUTPUT_FILE"
echo "      Timestamp: $(date)               " >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Export the function so it's available to the subshells created by find -exec
export -f execute_python_script

# Find all .py files and execute the helper function for each
# CHANGE: No need to check for SEARCH_DIR existence since it's "."
# The find command now starts searching from the current directory.
# We add -path './test_models.sh' -prune -o to prevent the script from finding itself.
find "$SEARCH_DIR" \( -name 'test_models.sh' -o -name "${OUTPUT_FILE}" \) -prune -o -type f -name "*.py" -exec bash -c 'execute_python_script "$1"' _ {} \;


echo "========================================" >> "$OUTPUT_FILE"
echo "           ALL SCRIPTS FINISHED         " >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"

echo "Script execution finished."
echo "Output saved to $OUTPUT_FILE"

# Unset the exported variables/functions if desired (optional cleanup)
unset OUTPUT_FILE
unset -f execute_python_script

exit 0
