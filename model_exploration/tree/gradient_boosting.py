import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

file_name = "../../cleaned_data_combined.csv"
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
    selected_features = ["Q2: How many ingredients would you expect this food item to contain?",
                         "Q3: In what setting would you expect this food to be served? Please check all that apply",
                         "Q4: How much would you expect to pay for one serving of this food item?",
                         "Q6: What drink would you pair with this food item?"]


    # Prepare the data for training
    df = df[selected_features + ["Label"]]

    # Handle missing values
    df = df.fillna(0)

    # Encode categorical features (if necessary)
    for col in selected_features:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Convert categorical labels to numerical values
    # df = pd.get_dummies(df, columns=["Label"], prefix="Label") # This line is incorrect for multi-class GradientBoostingClassifier
    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"]) # Convert labels to numeric representation

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state)

    x = df[selected_features].values  # Use selected_features to select features
    y = df["Label"].values  # Select the label column

    # Train-test split
    n_train = int(0.8 * len(df))
    x_train = x[:n_train]
    y_train = y[:n_train]

    x_test = x[n_train:]
    y_test = y[n_train:]

    # Train and evaluate a Gradient Boosting classifier
    clf = GradientBoostingClassifier(random_state=random_state)
    clf.fit(x_train, y_train)
    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    print(f"{type(clf).__name__} train acc: {train_acc}")
    print(f"{type(clf).__name__} test acc: {test_acc}")
