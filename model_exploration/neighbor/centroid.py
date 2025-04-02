"""
This Python file provides code for reading the training file
"cleaned_data_combined.csv" and training an instance-based model.
It adapts the original kNN baseline code to use RadiusNeighborsClassifier.

Keep in mind that the code provided does only basic feature transformations
to build a rudimentary model. Not all features are considered
in this code, and you should consider those features! Data scaling and
hyperparameter tuning (like the 'radius') are also important steps
for optimization. Use this code where appropriate, but don't stop here!

Another alternative instance-based model in scikit-learn is NearestCentroid.
"""

import pandas as pd
from sklearn.neighbors import RadiusNeighborsClassifier # Changed import
from sklearn.preprocessing import LabelEncoder
# Consider adding StandardScaler for feature scaling, especially for distance-based models
# from sklearn.preprocessing import StandardScaler

file_name = "../cleaned_data_combined.csv"
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
    for col in selected_features:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Convert categorical labels to numerical values (one-hot encoding)
    df = pd.get_dummies(df, columns=["Label"], prefix="Label")

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state)

    x = df.drop(columns=[col for col in df.columns if col.startswith("Label_")]).values
    y = df[[col for col in df.columns if col.startswith("Label_")]].values

    # === Feature Scaling (Recommended but optional step) ===
    # Distance-based algorithms like RadiusNeighbors often benefit from scaling.
    # Uncomment the following lines to add StandardScaler:
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x) # Scale all features
    # ========================================================

    # Train-test split
    n_train = int(0.8 * len(df))
    x_train = x[:n_train]
    y_train = y[:n_train]

    x_test = x[n_train:]
    y_test = y[n_train:]

    # --- Model Change: Use RadiusNeighborsClassifier ---
    # The 'radius' parameter is crucial and data-dependent.
    # It defines the maximum distance for a point to be considered a neighbor.
    # This value likely needs tuning based on the feature space distribution.
    # If no neighbors are found within the radius, it can lead to errors or
    # require handling (e.g., setting outlier_label).
    # We also specify `outlier_label` for points with no neighbors in radius.
    # Let's find the majority class in y_train to use as the outlier label.
    # Note: y_train is one-hot encoded, so we need to convert back or find the mode column index
    y_train_single_label = y_train.argmax(axis=1)
    from scipy.stats import mode
    import numpy as np

    mode_result = mode(y_train_single_label, keepdims=False)
    modes_array: np.ndarray = mode_result[0]

    # Use .item() IF you are sure there's only one mode element in modes_array
    try:
        majority_class_index: int = int(modes_array.item())
    except ValueError:
        # Handle the case where .item() fails (e.g., multiple modes found)
        print("Warning: Multiple modes found or mode array not scalar. Using the first mode.")
        majority_class_index: int = int(modes_array[0]) # Fallback to first mode

    # Create the outlier label in the same one-hot format as y_train
    num_classes = y_train.shape[1]
    outlier_label_one_hot = [0] * num_classes
    outlier_label_one_hot[majority_class_index] = 1


    clf = RadiusNeighborsClassifier(radius=10.0, # <<< Key hyperparameter - NEEDS TUNING!
                                    weights='uniform', # can be 'uniform' or 'distance'
                                    outlier_label=outlier_label_one_hot # Predict majority class for outliers
                                    # You might need to adjust n_jobs depending on your system
                                    # n_jobs=-1 # Use all available CPU cores
                                    )
    # -------------------------------------------------

    # Train and evaluate the classifier
    clf.fit(x_train, y_train)
    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    print(f"{type(clf).__name__} (radius={clf.radius}) train acc: {train_acc:.4f}")
    print(f"{type(clf).__name__} (radius={clf.radius}) test acc: {test_acc:.4f}")
