import pandas as pd
import sys
from sklearn.metrics import accuracy_score, classification_report
# Optional: Use train_test_split for random splitting if desired later
# from sklearn.model_selection import train_test_split

# --- Configuration ---
# Path to the prediction script
PRED_SCRIPT_PATH = '.' # Assuming pred.py is in the same directory
# Path to the dataset containing features AND true labels
LABELED_DATA_CSV = './dataset/cleaned_data_combined.csv'
# Name of the column containing the true labels in the CSV
LABEL_COLUMN_NAME = 'Label'
# Train/Test split ratio (e.g., 0.8 means 80% train, 20% test)
TRAIN_SPLIT_RATIO = 0.8
# ---------------------

# Add prediction script directory to path if necessary
sys.path.insert(0, PRED_SCRIPT_PATH)

try:
    # Import the necessary functions from pred.py
    from pred import predict_all
    print("Successfully imported functions from pred.py")
except ImportError as e:
    print(f"Error importing from pred.py: {e}")
    print(f"Please ensure pred.py is in the directory: {PRED_SCRIPT_PATH} or in the Python path.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

def run_accuracy_test(csv_path, label_column, train_ratio):
    """
    Loads data, splits into train/test (chronologically), runs predictions
    on the full dataset, and calculates accuracy on the test set portion.

    Args:
        csv_path (str): Path to the CSV file with features and true labels.
        label_column (str): Name of the column containing the true labels.
        train_ratio (float): Proportion of data to use for the 'train' part
                               (e.g., 0.8 for 80%). Test set is (1 - train_ratio).

    Returns:
        None: Prints the accuracy results for the test set.
    """
    print(f"\n--- Starting Accuracy Test on Test Split ---")
    print(f"Loading data from: {csv_path}")

    try:
        # Load the entire dataset with true labels
        df_labeled = pd.read_csv(csv_path, keep_default_na=False)
    except FileNotFoundError:
        print(f"Error: Labeled dataset file not found at '{csv_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        sys.exit(1)

    # Ensure the label column exists
    if label_column not in df_labeled.columns:
        print(f"Error: Label column '{label_column}' not found in '{csv_path}'.")
        print(f"Available columns: {list(df_labeled.columns)}")
        sys.exit(1)

    # --- Data Splitting (Chronological) ---
    n_total = len(df_labeled)
    if n_total == 0:
        print("Error: Dataset is empty.")
        sys.exit(1)

    n_train = int(train_ratio * n_total)
    n_test = n_total - n_train

    if n_train == 0 or n_test == 0:
        print(f"Error: With train_ratio={train_ratio} and total samples={n_total},")
        print(f"the resulting train ({n_train}) or test ({n_test}) set is empty.")
        print("Adjust train_ratio or increase dataset size.")
        sys.exit(1)

    print(f"Splitting data chronologically: {n_train} training samples, {n_test} testing samples.")

    # Extract ALL true labels first
    true_labels_all = df_labeled[label_column].tolist()
    print(f"Found {len(true_labels_all)} total true labels.")

    # Get the true labels *only for the test set*
    true_labels_test = true_labels_all[n_train:]
    # Get the portion of the dataframe corresponding to the test set (for mismatch reporting)
    df_test = df_labeled.iloc[n_train:].reset_index(drop=True) # Reset index for easier lookup later

    print("\nGenerating predictions using predict_all function...")
    print("NOTE: predict_all is assumed to process the *entire* CSV and return predictions in order.")
    try:
        # Generate predictions using the function from pred.py on the *entire* dataset
        predicted_labels_all = predict_all(csv_path)
        print(f"Generated {len(predicted_labels_all)} total predictions.")
    except Exception as e:
        print(f"Error occurred during prediction using predict_all: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

    # --- Validation and Accuracy Calculation (on Test Set) ---

    # First, validate the output of predict_all against the full dataset size
    if len(true_labels_all) != len(predicted_labels_all):
        print("\nError: Mismatch between total number of true labels and total predicted labels from predict_all!")
        print(f"Number of true labels (total): {len(true_labels_all)}")
        print(f"Number of predicted labels (total): {len(predicted_labels_all)}")
        print("Cannot proceed with test set evaluation.")
        sys.exit(1)

    # Extract the predictions corresponding to the test set indices
    predicted_labels_test = predicted_labels_all[n_train:]
    print(f"Extracted {len(predicted_labels_test)} predictions for the test set.")

    # Final check for consistency within the test set (should always pass if above checks passed)
    if len(true_labels_test) != len(predicted_labels_test):
        print("\nError: Internal inconsistency in splitting test labels/predictions. This should not happen.")
        sys.exit(1)

    if not true_labels_test: # Should be caught by n_test==0 check earlier
        print("\nError: Test set is effectively empty. Cannot calculate accuracy.")
        sys.exit(1)

    print("\n--- Test Set Results ---")
    # Calculate accuracy using only the test set labels and predictions
    accuracy = accuracy_score(true_labels_test, predicted_labels_test)
    print(f"Test Set Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Print a more detailed classification report for the test set
    print("\nTest Set Classification Report:")
    # Get unique labels present in the test set results
    labels_present_test = sorted(list(set(true_labels_test) | set(predicted_labels_test)))
    try:
        print(classification_report(true_labels_test, predicted_labels_test, labels=labels_present_test, zero_division=0))
    except ValueError as e:
        print(f"Could not generate classification report for test set: {e}")
        print("This might happen if there are inconsistencies in label types or formats.")

    # Optional: Show some mismatches from the test set
    print("\n--- Test Set Mismatches (Predicted vs True) ---")
    mismatches = 0
    limit_mismatches_shown = 10
    for i in range(len(true_labels_test)): # Iterate through the test set indices
        if true_labels_test[i] != predicted_labels_test[i]:
            mismatches += 1
            if mismatches <= limit_mismatches_shown:
                # Get the ID from df_test if available, otherwise use index relative to test set
                # df_test has a reset index (0 to n_test-1)
                row_id_info = df_test.iloc[i].get('ID', f'Test Index {i} (Orig Index {n_train + i})')
                print(f"Row {row_id_info}: Predicted='{predicted_labels_test[i]}', True='{true_labels_test[i]}'")

    if mismatches == 0:
        print("No mismatches found in the test set.")
    elif mismatches > limit_mismatches_shown:
        print(f"... and {mismatches - limit_mismatches_shown} more mismatches in the test set.")
    print(f"Total test set mismatches: {mismatches} out of {len(true_labels_test)}")


if __name__ == "__main__":
    run_accuracy_test(LABELED_DATA_CSV, LABEL_COLUMN_NAME, TRAIN_SPLIT_RATIO)
