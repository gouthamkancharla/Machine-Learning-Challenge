# --- START OF FILE centroid.py ---

"""
This Python file provides code for reading the training file
"cleaned_data_combined.csv" and training a Nearest Centroid classifier.
It adapts the original kNN baseline code. Note that basic feature
transformations are used, and further feature engineering might be beneficial.
"""

import pandas as pd
from sklearn.neighbors import NearestCentroid # Changed import
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # Using sklearn's split for clarity

file_name = "../../dataset/cleaned_data_combined.csv"
random_state = 42

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """
    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

if __name__ == "__main__":

    df = pd.read_csv(file_name)

    # Select a subset of features for the baseline model
    # Consider adding/removing/transforming features for better performance
    selected_features = ["Q2: How many ingredients would you expect this food item to contain?",
                         "Q3: In what setting would you expect this food to be served? Please check all that apply",
                         "Q4: How much would you expect to pay for one serving of this food item?",
                         "Q6: What drink would you pair with this food item?"]

    # Prepare the data for training
    df_processed = df[selected_features + ["Label"]].copy() # Work on a copy

    # Handle missing values - Simple fill with 0, consider other strategies
    # Apply to_numeric specifically to Q4 before filling NaNs
    # Ensure Q4 is numeric before potential string operations in LabelEncoder
    if "Q4: How much would you expect to pay for one serving of this food item?" in df_processed.columns:
        df_processed["Q4: How much would you expect to pay for one serving of this food item?"] = df_processed["Q4: How much would you expect to pay for one serving of this food item?"].apply(to_numeric)

    df_processed = df_processed.fillna(0)

    # Encode categorical features using LabelEncoder
    feature_encoders = {}
    for col in selected_features:
        if df_processed[col].dtype == 'object':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            feature_encoders[col] = le # Store encoders if needed later

    # Encode the target variable 'Label' into numerical format (0, 1, 2...)
    # NearestCentroid expects a 1D array of labels, not one-hot encoded
    label_encoder = LabelEncoder()
    df_processed["Label_encoded"] = label_encoder.fit_transform(df_processed["Label"])

    # Separate features (X) and target (y)
    X = df_processed[selected_features].values
    y = df_processed["Label_encoded"].values # Use the single encoded column

    # Train-test split using sklearn function
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y # Stratify helps maintain class proportion
    )

    # Train and evaluate a Nearest Centroid classifier
    # No 'n_neighbors' parameter for NearestCentroid
    clf = NearestCentroid()
    clf.fit(X_train, y_train)

    # Evaluate the model
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"{type(clf).__name__} train acc: {train_acc:.4f}") # Added formatting
    print(f"{type(clf).__name__} test acc: {test_acc:.4f}") # Added formatting

    # Optional: Print class labels if needed
    # print("Class labels mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# --- END OF FILE centroid.py ---
