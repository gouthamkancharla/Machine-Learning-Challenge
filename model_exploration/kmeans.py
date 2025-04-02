# --- START OF FILE kmeans_clustering.py ---

"""
This Python file demonstrates k-means clustering on the dataset from
"cleaned_data_combined.csv". It preprocesses features, applies k-means,
and evaluates the results using common clustering metrics.

Note: This script adapts the preprocessing steps from the original kNN example
but modifies them for unsupervised clustering (k-means). Feature scaling is added,
and one-hot encoding is used for categorical features.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import numpy as np # Import numpy for isnan check

file_name = "../../dataset/cleaned_data_combined.csv" # Make sure this path is correct
random_state = 42

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    Handles potential comma separators in numbers.
    """
    if isinstance(s, str):
        try:
            s = s.replace(",", '')
            return float(s)
        except ValueError:
            return float('nan')
    # Check for numpy NaN specifically as pandas might return numpy types
    elif isinstance(s, (int, float, np.number)) and not np.isnan(s):
        return float(s)
    else:
        # Handles None, np.nan, pd.NA etc.
        return float('nan')

if __name__ == "__main__":

    df = pd.read_csv(file_name)

    # --- Feature Selection ---
    # Consider using more features or different ones for potentially better clustering
    selected_features = ["Q2: How many ingredients would you expect this food item to contain?",
                         "Q3: In what setting would you expect this food to be served? Please check all that apply",
                         "Q4: How much would you expect to pay for one serving of this food item?",
                         "Q6: What drink would you pair with this food item?"]

    # Keep the original 'Label' column separately for evaluation
    if "Label" not in df.columns:
        raise ValueError("The required 'Label' column is missing from the CSV.")
    true_labels_raw = df["Label"].astype(str) # Ensure labels are strings before encoding

    # Select only the features for clustering
    df_features = df[selected_features].copy() # Use .copy() to avoid SettingWithCopyWarning

    # --- Preprocessing ---

    # 1. Convert potentially numeric string columns to actual numeric types
    # Apply 'to_numeric' carefully, especially to columns intended to be numeric
    # Example: Apply to Q4 if it contains strings like "1,000" or "$5.00"
    # Note: The original example didn't explicitly use to_numeric on selected cols,
    # but Q4 might benefit. Assuming Q2 and Q4 should be numeric:
    if "Q2: How many ingredients would you expect this food item to contain?" in df_features.columns:
        df_features["Q2: How many ingredients would you expect this food item to contain?"] = df_features["Q2: How many ingredients would you expect this food item to contain?"].apply(to_numeric)
    if "Q4: How much would you expect to pay for one serving of this food item?" in df_features.columns:
        df_features["Q4: How much would you expect to pay for one serving of this food item?"] = df_features["Q4: How much would you expect to pay for one serving of this food item?"].apply(to_numeric)


    # 2. Identify categorical and numerical features *after* initial conversion
    categorical_cols = [col for col in selected_features if df_features[col].dtype == 'object']
    numerical_cols = [col for col in selected_features if col not in categorical_cols]

    # 3. Handle Missing Values
    # For numerical: Impute with median (often more robust to outliers than mean)
    for col in numerical_cols:
        if df_features[col].isnull().any():
            median_val = df_features[col].median()
            df_features[col] = df_features[col].fillna(median_val)

    # For categorical: Impute with mode or a constant placeholder like 'Missing'
    for col in categorical_cols:
        if df_features[col].isnull().any():
            mode_val = df_features[col].mode()[0] # mode() returns a Series
            df_features[col] = df_features[col].fillna(mode_val)

    # 4. Encode Categorical Features using One-Hot Encoding
    # pd.get_dummies is suitable here
    df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=False) # Keep all dummies

    # Ensure all feature columns are numeric after get_dummies
    # This loop helps catch any unexpected non-numeric types left.
    for col in df_features.columns:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        if df_features[col].isnull().any():
            # If NaNs appear after coercion (shouldn't happen with get_dummies), fill them
            print(f"Warning: NaNs appeared in column {col} after get_dummies/coercion. Filling with 0.")
            df_features[col] = df_features[col].fillna(0)


    # --- Feature Scaling ---
    # Scale features to have zero mean and unit variance
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_features)

    # --- K-Means Clustering ---

    # Determine the number of clusters (k)
    # Using the number of unique original labels as a common heuristic
    n_clusters = true_labels_raw.nunique()
    print(f"Number of unique labels (using as k for k-means): {n_clusters}")

    # Initialize and fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10) # n_init=10 is recommended
    kmeans.fit(x_scaled)

    # Get cluster assignments for each data point
    cluster_labels = kmeans.labels_

    # --- Evaluation ---
    # Encode the true labels numerically for evaluation metrics
    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(true_labels_raw)

    # Calculate clustering metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(x_scaled, cluster_labels)
    ari = adjusted_rand_score(true_labels_encoded, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels_encoded, cluster_labels)

    print(f"\n--- K-Means Clustering Results ---")
    print(f"Number of clusters (k): {n_clusters}")
    print(f"Inertia (Within-cluster sum of squares): {inertia:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

    # Optional: You can inspect the cluster centers
    # print("\nCluster Centers (in scaled feature space):")
    # print(kmeans.cluster_centers_)

    # Optional: Analyze cluster composition relative to true labels
    # results_df = pd.DataFrame({'TrueLabel': true_labels_raw, 'Cluster': cluster_labels})
    # print("\nCluster composition by true label:")
    # print(pd.crosstab(results_df['TrueLabel'], results_df['Cluster']))

# --- END OF FILE kmeans_clustering.py ---
