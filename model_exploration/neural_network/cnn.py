# --- START OF FILE cnn.py ---

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

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
    num_classes = len(np.unique(y)) # Get the number of classes
    y = to_categorical(y, num_classes=num_classes)  # One-hot encode labels for neural network

    # Drop the original "Label" column and prepare features
    X = df.drop(columns=["Label"]).values

    # Standardize the features (important for neural networks)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # --- CNN Specific Step: Reshape data ---
    # CNNs (especially Conv1D) expect input in the format (batch_size, steps, channels)
    # We treat the features as 'steps' and add a single 'channel' dimension.
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # --- Define the CNN architecture ---
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),  # Input shape: (num_features, 1)
        Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'), # 1D Convolutional layer
        # Using 'same' padding to maintain sequence length initially if needed
        # MaxPooling1D(pool_size=2), # Optional: Pooling layer to reduce dimensionality
        Dropout(0.3),  # Dropout for regularization
        Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'), # Another 1D Conv layer
        # MaxPooling1D(pool_size=2), # Optional pooling
        Flatten(),  # Flatten the output of Conv layers before passing to Dense layers
        Dropout(0.3),  # Dropout for regularization
        Dense(64, activation='relu'), # Dense layer like in MLP, but after feature extraction by CNN
        Dense(num_classes, activation='softmax')  # Output layer (softmax for multi-class classification)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',  # Loss function for multi-class classification
                  metrics=['accuracy'])

    # Print model summary to understand the architecture
    print("Model Summary:")
    model.summary()

    # Train the model
    batch_size = 32
    epochs = 50
    print("\nStarting Training...")
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1) # Set verbose=1 or 2 to see progress per epoch

    # Evaluate the model
    print("\nEvaluating Model...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nCNN train accuracy: {train_acc:.4f}")
    print(f"CNN test accuracy: {test_acc:.4f}")

# --- END OF FILE cnn.py ---
