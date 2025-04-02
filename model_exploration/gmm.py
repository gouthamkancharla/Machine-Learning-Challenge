# --- START OF FILE gmm_clustering.py ---

"""
This Python file demonstrates Gaussian Mixture Model (GMM) clustering on the
dataset from "cleaned_data_combined.csv". It preprocesses features,
applies GMM, and evaluates the results using common clustering metrics
and model selection criteria (BIC/AIC).

Note: This script adapts preprocessing steps similar to the k-means example.
Feature scaling and one-hot encoding are used.
"""

import pandas as pd
from sklearn.mixture import GaussianMixture
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
    if "Q2: How many ingredients would you expect this food item to contain?" in df_features.columns:
        df_features["Q2: How many ingredients would you expect this food item to contain?"] = df_features["Q2: How many ingredients would you expect this food item to contain?"].apply(to_numeric)
    if "Q4: How much would you expect to pay for one serving of this food item?" in df_features.columns:
        df_features["Q4: How much would you expect to pay for one serving of this food item?"] = df_features["Q4: How much would you expect to pay for one serving of this food item?"].apply(to_numeric)


    # 2. Identify categorical and numerical features *after* initial conversion
    categorical_cols = [col for col in selected_features if df_features[col].dtype == 'object']
    numerical_cols = [col for col in selected_features if col not in categorical_cols]

    # 3. Handle Missing Values
    # For numerical: Impute with median
    for col in numerical_cols:
        if df_features[col].isnull().any():
            median_val = df_features[col].median()
            df_features[col] = df_features[col].fillna(median_val)
            if df_features[col].isnull().any(): # Handle cases where median might be NaN (e.g., all NaNs)
                df_features[col] = df_features[col].fillna(0)


    # For categorical: Impute with mode
    for col in categorical_cols:
        if df_features[col].isnull().any():
            # Calculate mode, handle potential multiple modes by taking the first
            modes = df_features[col].mode()
            mode_val = modes[0] if not modes.empty else 'Missing' # Use placeholder if mode is empty
            df_features[col] = df_features[col].fillna(mode_val)

    # 4. Encode Categorical Features using One-Hot Encoding
    df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=False, dummy_na=False) # Don't create NaN category if we filled them

    # Ensure all feature columns are numeric after get_dummies
    for col in df_features.columns:
        # Convert to numeric; coerce errors will turn problematic values into NaN
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        # Check if NaNs were introduced or still exist, and fill them (e.g., with 0 or median/mean of col)
        if df_features[col].isnull().any():
            print(f"Warning: NaNs found in column {col} after get_dummies/coercion. Filling with 0.")
            df_features[col] = df_features[col].fillna(0) # Or use median/mean if appropriate


    # --- Feature Scaling ---
    # Scale features to have zero mean and unit variance
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_features)

    # --- Gaussian Mixture Model Clustering ---

    # Determine the number of components (clusters)
    # Using the number of unique original labels as a starting point
    n_components = true_labels_raw.nunique()
    print(f"Number of unique labels (using as n_components for GMM): {n_components}")

    # Initialize and fit Gaussian Mixture Model
    # covariance_type options: 'full', 'tied', 'diag', 'spherical'
    covariance_type = 'full'
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type=covariance_type,
                          random_state=random_state,
                          n_init=5, # Number of initializations to perform
                          max_iter=100) # Max iterations for EM
    gmm.fit(x_scaled)

    # Get cluster assignments (hard assignments)
    cluster_labels = gmm.predict(x_scaled)
    # You can also get probabilities: cluster_probs = gmm.predict_proba(x_scaled)

    # --- Evaluation ---
    # Encode the true labels numerically for evaluation metrics
    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(true_labels_raw)

    # Calculate clustering metrics
    # Note: Silhouette score can be computationally expensive for large datasets
    try:
        silhouette = silhouette_score(x_scaled, cluster_labels)
    except ValueError as e:
        print(f"Could not calculate Silhouette Score: {e}")
        silhouette = float('nan') # Assign NaN if calculation fails (e.g., only 1 cluster found)


    ari = adjusted_rand_score(true_labels_encoded, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels_encoded, cluster_labels)

    # Calculate BIC and AIC (lower is generally better)
    bic = gmm.bic(x_scaled)
    aic = gmm.aic(x_scaled)

    print(f"\n--- Gaussian Mixture Model Clustering Results ---")
    print(f"Number of components: {n_components}")
    print(f"Covariance type: '{covariance_type}'")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Bayesian Information Criterion (BIC): {bic:.4f}")
    print(f"Akaike Information Criterion (AIC): {aic:.4f}")

    # Optional: Analyze cluster composition relative to true labels
    # results_df = pd.DataFrame({'TrueLabel': true_labels_raw, 'Cluster': cluster_labels})
    # print("\nCluster composition by true label:")
    # print(pd.crosstab(results_df['TrueLabel'], results_df['Cluster']))

# --- END OF FILE gmm_clustering.py ---
