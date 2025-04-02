import pandas as pd
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
# Consider adding StandardScaler for feature scaling, especially for distance-based models
# from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
import numpy as np

file_name = "../../dataset/cleaned_data_combined.csv"
random_state = 42

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    (Note: This function is not actively used in the current feature selection below,
     but might be useful if processing other numerical columns).
    """
    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

if __name__ == "__main__":

    df = pd.read_csv(file_name)

    # Select a subset of features for the baseline model
    # WARNING: This is a limited subset. Consider using more features for better performance.
    selected_features = ["Q2: How many ingredients would you expect this food item to contain?",
                         "Q3: In what setting would you expect this food to be served? Please check all that apply",
                         "Q4: How much would you expect to pay for one serving of this food item?",
                         "Q6: What drink would you pair with this food item?"]

    # Prepare the data for training
    df = df[selected_features + ["Label"]]

    # Handle missing values (using simple fillna(0) as in the original baseline)
    # More sophisticated imputation might be better.
    df = df.fillna(0)

    # Encode categorical features using simple integer encoding
    # Consider OneHotEncoder for nominal features if appropriate,
    # although LabelEncoder is often sufficient for tree-based/neighbor-based models sometimes.
    label_encoders = {} # Store encoders if needed later
    for col in selected_features:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # --- Target Variable Handling ---
    # Keep the target 'Label' as a single column with string labels first
    # Then encode it into integer labels (0, 1, 2, ...)
    target_col = "Label"
    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col])
    # We can retrieve the original class names later using target_encoder.classes_
    # --------------------------------

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state)

    # *** FIX: Separate features (X) and the *single-column integer-encoded* target (y) ***
    x = df[selected_features].values
    y = df[target_col].values # Use the integer encoded target directly
    # ************************************************************************************

    # === Feature Scaling (Recommended but optional step) ===
    # Distance-based algorithms like RadiusNeighbors often benefit from scaling.
    # Uncomment the following lines to add StandardScaler:
    # from sklearn.preprocessing import StandardScaler # Make sure it's imported
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x) # Scale all features
    # ========================================================

    # Train-test split
    n_train = int(0.8 * len(df))
    x_train = x[:n_train]
    y_train = y[:n_train] # y_train is now 1D array of integers

    x_test = x[n_train:]
    y_test = y[n_train:]   # y_test is now 1D array of integers

    # --- Model Change: Use RadiusNeighborsClassifier ---
    # The 'radius' parameter is crucial and data-dependent.
    # It defines the maximum distance for a point to be considered a neighbor.
    # This value likely needs tuning based on the feature space distribution.
    # If no neighbors are found within the radius, it can lead to errors or
    # require handling (e.g., setting outlier_label).

    # *** FIX: Use the special string "most_frequent" for outlier_label ***
    # This tells scikit-learn to automatically determine the most frequent class
    # in y_train during fitting and use that as the label for outliers.
    # This avoids the need for manual calculation and potential type/shape mismatches.
    # Remove the manual mode calculation block.

    clf = RadiusNeighborsClassifier(radius=10.0, # <<< Key hyperparameter - NEEDS TUNING!
                                    weights='uniform', # can be 'uniform' or 'distance'
                                    outlier_label="most_frequent" # Use built-in strategy
                                    # You might need to adjust n_jobs depending on your system
                                    # n_jobs=-1 # Use all available CPU cores
                                    )
    # **********************************************************************
    # -------------------------------------------------

    # Train and evaluate the classifier
    clf.fit(x_train, y_train) # Fit with the 1D integer-encoded y_train

    # *** FIX: Evaluate score correctly with 1D integer labels ***
    # The .score() method expects the same format for y as was used in fit()
    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    # ************************************************************

    print(f"{type(clf).__name__} (radius={clf.radius}) train acc: {train_acc:.4f}")
    print(f"{type(clf).__name__} (radius={clf.radius}) test acc: {test_acc:.4f}")

    # You can access the automatically determined outlier label if needed:
    # print(f"Outlier label used: {clf.outlier_label_}")
    # To map back to original label name:
    # if isinstance(clf.outlier_label_, (int, np.integer)):
    #    print(f"Original outlier label name: {target_encoder.inverse_transform([clf.outlier_label_])[0]}")
