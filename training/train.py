from pathlib import Path
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras import layers


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_DIR = BASE_DIR / "model"


# Load dataset
print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(data.head())


# Input and output
X = data.drop("final_score", axis=1)
y = data["final_score"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build ANN
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),

    layers.Dense(32, activation="relu"),

    layers.Dense(16, activation="relu"),

    layers.Dense(8, activation="relu"),

    layers.Dense(1)
])


# Compile
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# Train
print("\nTraining ANN...\n")

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_split=0.2,
    verbose=1
)


# Evaluate
loss, mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")
print(f"Test MAE: {mae:.2f}")


# Create model directory
MODEL_DIR.mkdir(exist_ok=True)


# Save ANN
model.save(
    MODEL_DIR / "student_model.keras"
)


# Save scaler
with open(MODEL_DIR / "scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


print("\n==============================")
print("SUCCESS")
print("==============================")
print("Model saved:")
print(MODEL_DIR / "student_model.keras")

print("Scaler saved:")
print(MODEL_DIR / "scaler.pkl")