import pandas as pd
import numpy as np
import tensorflow as tf

from datetime import datetime
import os

from pathlib import Path
import json

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt

from src.models.common.data import (
    PROJECT_ROOT,
    TRAIN_FILE,
    WINDOW_SIZE,
    FORECAST_HORIZON,
    load_splits,
    create_sequences
)

# ------------------------------------
# Configuration
# ------------------------------------

BATCH_SIZE = 32
EPOCHS = 25

# ------------------------------------
# Project Paths
# ------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# ------------------------------------
# Output Directories
# ------------------------------------

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

MODEL_NAME = "LSTM"

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / MODEL_NAME
)

EXPERIMENT_DIR = (
    MODEL_ROOT
    / "experiments"
    / timestamp
)

FIGURE_DIR = (
    EXPERIMENT_DIR
    / "figures"
)

CHECKPOINT_DIR = (
    MODEL_ROOT
    / "checkpoints"
)

# Create directories

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Files

MODEL_OUTPUT = (
    EXPERIMENT_DIR
    / "weather_lstm.keras"
)

TRAINING_HISTORY_FILE = (
    EXPERIMENT_DIR
    / "training_history.csv"
)

METRICS_FILE = (
    EXPERIMENT_DIR
    / "metrics.json"
)

PREDICTIONS_FILE = (
    EXPERIMENT_DIR
    / "predictions.csv"
)

MODEL_SUMMARY_FILE = (
    EXPERIMENT_DIR
    / "model_summary.txt"
)

LATEST_DIR = (
    MODEL_ROOT
    / "latest"
)

LATEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint = ModelCheckpoint(
    filepath=CHECKPOINT_DIR / "best.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# ------------------------------------
# Load datasets
# ------------------------------------

print("Loading datasets...")

print("PROJECT_ROOT:", PROJECT_ROOT)
print("TRAIN_FILE:", TRAIN_FILE)
print("Exists:", TRAIN_FILE.exists())

train_df, dev_df, test_df = load_splits()

# ------------------------------------
# Create Sequences
# ------------------------------------

print("Creating sequences...")

X_train, y_train = create_sequences(train_df)

X_dev, y_dev = create_sequences(dev_df)

X_test, y_test = create_sequences(test_df)

print()

print("Training samples :", len(X_train))
print("Development samples :", len(X_dev))
print("Testing samples :", len(X_test))

print()

print("Input shape:", X_train.shape)

# ------------------------------------
# Normalization Layer
# ------------------------------------

normalizer = layers.Normalization()

normalizer.adapt(

    X_train.reshape(

        -1,
        X_train.shape[-1]

    )

)

print("Normalization complete.")

# ------------------------------------
# Build Model
# ------------------------------------

inputs = tf.keras.Input(

    shape=(
        WINDOW_SIZE,
        X_train.shape[-1]
    ),

    name="weather_sequence"

)

# Normalize every timestep

x = layers.TimeDistributed(

    normalizer

)(inputs)

# ------------------------------------
# LSTM Stack
# ------------------------------------

x = layers.LSTM(

    64,
    return_sequences=True

)(x)

x = layers.Dropout(

    0.20

)(x)

x = layers.LSTM(

    32

)(x)

x = layers.Dropout(

    0.20

)(x)

# ------------------------------------
# Dense Layers
# ------------------------------------

x = layers.Dense(

    32,
    activation="relu"

)(x)

x = layers.Dropout(

    0.20

)(x)

x = layers.Dense(

    16,
    activation="relu"

)(x)

outputs = layers.Dense(

    1,

    name="temperature_prediction"

)(x)

model = Model(

    inputs,

    outputs

)

print()

model.summary()

with open(
    MODEL_SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:
    print("Encoding:", f.encoding)

    model.summary(
        print_fn=lambda x: f.write(x + "\n")
    )

# ------------------------------------
# Compile Model
# ------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse")
    ]
)

print("\nModel compiled successfully.\n")

# ------------------------------------
# Callbacks
# ------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

callbacks=[early_stopping, checkpoint]

ModelCheckpoint(
    filepath=CHECKPOINT_DIR / "best.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# ------------------------------------
# Train Model
# ------------------------------------

print("Beginning training...\n")

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_dev, y_dev),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        checkpoint
    ],

    verbose=1

)


history_df = pd.DataFrame(history.history)

history_df.to_csv(
    TRAINING_HISTORY_FILE,
    index=False
)

print("\nTraining Complete!\n")

# ------------------------------------
# Evaluate Model
# ------------------------------------

print("Evaluating on Test Set...\n")

loss, mae, rmse = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Test Loss : {loss:.4f}")
print(f"Test MAE  : {mae:.4f}")
print(f"Test RMSE : {rmse:.4f}")

metrics = {
    "loss": float(loss),
    "mae": float(mae),
    "rmse": float(rmse)
}

with open(
    METRICS_FILE,
    "w"
) as f:
    json.dump(metrics, f, indent=4)

print("Evaluation metrics saved.")

# ------------------------------------
# Make Predictions
# ------------------------------------

predictions = model.predict(
    X_test,
    verbose=0
)

prediction_df = pd.DataFrame({

    "Actual": y_test,

    "Predicted": predictions.flatten(),

    "Error": predictions.flatten() - y_test

})

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

predictions = predictions.flatten()

# ------------------------------------
# Display Predictions
# ------------------------------------

print("\nSample Predictions\n")
print("-" * 45)

for i in range(10):

    print(
        f"Prediction: {predictions[i]:7.2f} °C"
        f" | Actual: {y_test[i]:7.2f} °C"
    )

# ------------------------------------
# Calculate Error Statistics
# ------------------------------------

errors = predictions - y_test

print("\nPrediction Statistics")
print("-" * 45)

print(f"Mean Error: {np.mean(errors):.4f}")
print(f"Std Error : {np.std(errors):.4f}")
print(f"Max Error : {np.max(np.abs(errors)):.4f}")

# ------------------------------------
# Save Final Model
# ------------------------------------

model.save(MODEL_OUTPUT)

print(f"\nModel saved to:\n{MODEL_OUTPUT}")

# ------------------------------------
# Plot Training History
# ------------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    FIGURE_DIR / "loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------
# Plot MAE
# ------------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    history.history["mae"],
    label="Training MAE"
)

plt.plot(
    history.history["val_mae"],
    label="Validation MAE"
)

plt.title("Training vs Validation MAE")

plt.xlabel("Epoch")

plt.ylabel("Mean Absolute Error")

plt.legend()

plt.grid(True)

plt.savefig(
    FIGURE_DIR / "mae_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------
# Predicted vs Actual
# ------------------------------------

plt.figure(figsize=(14,6))

plt.plot(
    y_test[:200],
    label="Actual Temperature"
)

plt.plot(
    predictions[:200],
    label="Predicted Temperature"
)

plt.title("Predicted vs Actual Temperatures")

plt.xlabel("Sample")

plt.ylabel("Temperature (°C)")

plt.legend()

plt.grid(True)

plt.savefig(
    FIGURE_DIR / "prediction_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFinished!")