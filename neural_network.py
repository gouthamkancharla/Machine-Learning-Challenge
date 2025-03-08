import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

file_name = "cleaned_data_combined.csv"
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
    # Load the dataset
    df = pd.read_csv(file_name)

    # Select a subset of features for the baseline model
    selected_features = [
        "Q2: How many ingredients would you expect this food item to contain?",
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q4: How much would you expect to pay for one serving of this food item?",
        "Q6: What drink would you pair with this food item?"
    ]

    # Prepare the data for training
    df = df[selected_features + ["Label"]]

    # Handle missing values
    df = df.fillna(0)

    # Encode categorical features (if necessary)
    for col in selected_features:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Convert categorical labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Label"])
    y = to_categorical(y)  # One-hot encode labels for neural network

    # Drop the original "Label" column and prepare features
    X = df.drop(columns=["Label"]).values

    # Standardize the features (important for neural networks)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Define the neural network architecture
    model = Sequential([
        Input(shape=(X_train.shape[1],)),  # Explicitly define the input shape
        Dense(128, activation='relu'),  # First hidden layer
        Dropout(0.3),  # Dropout for regularization
        Dense(64, activation='relu'),  # Hidden layer
        Dropout(0.3),  # Dropout for regularization
        Dense(32, activation='relu'),  # Hidden layer
        Dense(y.shape[1], activation='softmax')  # Output layer (softmax for multi-class classification)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',  # Loss function for multi-class classification
                  metrics=['accuracy'])

    # Train the model
    batch_size = 32
    epochs = 50
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1)

    # Evaluate the model
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Neural Network train accuracy: {train_acc:.4f}")
    print(f"Neural Network test accuracy: {test_acc:.4f}")
