from pathlib import Path
from datetime import datetime
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model


# ------------------------------------
# Configuration
# ------------------------------------

# Use the previous 72 hours to forecast hours 1 through 24.
INPUT_HOURS = 72
OUTPUT_HOURS = 24

BATCH_SIZE = 32
EPOCHS = 50
RANDOM_SEED = 21

# Predict a correction to the previous day's temperatures rather than
# forcing the network to reconstruct the full daily temperature curve.
USE_PREVIOUS_DAY_BASELINE = True

# Optional column detection. Change these lists only if your CSV uses
# different names.
TIME_COLUMN_CANDIDATES = [
    "time",
    "datetime",
    "timestamp",
    "date",
]

GROUP_COLUMN_CANDIDATES = [
    "city",
    "location",
    "location_name",
]

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ------------------------------------
# Project Paths
# ------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = DATA_DIR / "splits" / "train.csv"
DEV_FILE = DATA_DIR / "splits" / "dev.csv"
TEST_FILE = DATA_DIR / "splits" / "test.csv"


# ------------------------------------
# Output Directories
# ------------------------------------

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "LSTM"

MODEL_ROOT = MODEL_DIR / MODEL_NAME
EXPERIMENT_DIR = MODEL_ROOT / "experiments" / timestamp
FIGURE_DIR = EXPERIMENT_DIR / "figures"
LATEST_DIR = MODEL_ROOT / "latest"

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
LATEST_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT = EXPERIMENT_DIR / "weather_lstm.keras"
LATEST_MODEL_OUTPUT = LATEST_DIR / "weather_lstm.keras"
TRAINING_HISTORY_FILE = EXPERIMENT_DIR / "training_history.csv"
METRICS_FILE = EXPERIMENT_DIR / "metrics.json"
PREDICTIONS_FILE = EXPERIMENT_DIR / "predictions.csv"
MODEL_SUMMARY_FILE = EXPERIMENT_DIR / "model_summary.txt"
CONFIG_FILE = EXPERIMENT_DIR / "model_config.json"
LATEST_CONFIG_FILE = LATEST_DIR / "model_config.json"


# ------------------------------------
# Feature Columns
# ------------------------------------

# Past temperature is intentionally included as an input feature. This is
# historical information available before a forecast begins, not leakage.
FEATURE_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "is_day",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]

TARGET_COLUMN = "temperature_2m"
TEMPERATURE_FEATURE_INDEX = FEATURE_COLUMNS.index(TARGET_COLUMN)


# ------------------------------------
# Helpers
# ------------------------------------

def detect_column(dataframe, candidates):
    """Return the first candidate column that exists, otherwise None."""
    return next((name for name in candidates if name in dataframe.columns), None)


def validate_dataframe(dataframe, split_name):
    """Validate required columns and convert model data to float32."""
    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name} is missing required columns: {missing_columns}"
        )

    dataframe = dataframe.copy()

    for column in FEATURE_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    invalid_counts = dataframe[FEATURE_COLUMNS].isna().sum()
    invalid_counts = invalid_counts[invalid_counts > 0]

    if not invalid_counts.empty:
        raise ValueError(
            f"{split_name} contains missing or non-numeric feature values:\n"
            f"{invalid_counts.to_string()}"
        )

    dataframe[FEATURE_COLUMNS] = dataframe[FEATURE_COLUMNS].astype(
        np.float32
    )

    return dataframe


def split_into_contiguous_groups(dataframe, split_name):
    """
    Yield city/location groups and split them again when hourly timestamps
    are not contiguous. If no city or timestamp column exists, the complete
    dataframe is used as one sequence source.
    """
    time_column = detect_column(dataframe, TIME_COLUMN_CANDIDATES)
    group_column = detect_column(dataframe, GROUP_COLUMN_CANDIDATES)

    working = dataframe.copy()

    if time_column is not None:
        working[time_column] = pd.to_datetime(
            working[time_column],
            errors="coerce",
        )

        if working[time_column].isna().any():
            bad_rows = int(working[time_column].isna().sum())
            raise ValueError(
                f"{split_name} contains {bad_rows} invalid timestamp values "
                f"in column '{time_column}'."
            )

    if group_column is None:
        grouped_data = [("all", working)]
    else:
        grouped_data = working.groupby(
            group_column,
            sort=False,
            dropna=False,
        )

    for group_name, group_df in grouped_data:
        group_df = group_df.copy()

        if time_column is not None:
            group_df = group_df.sort_values(time_column).reset_index(drop=True)

            # A new segment begins whenever adjacent records are not exactly
            # one hour apart. This prevents sequences from crossing data gaps.
            new_segment = (
                group_df[time_column]
                .diff()
                .ne(pd.Timedelta(hours=1))
            )
            group_df["_sequence_segment"] = new_segment.cumsum()
            segment_groups = group_df.groupby(
                "_sequence_segment",
                sort=False,
            )
        else:
            segment_groups = [(0, group_df.reset_index(drop=True))]

        for segment_id, segment_df in segment_groups:
            yield group_name, segment_id, segment_df.reset_index(drop=True)


def create_sequences(dataframe, split_name):
    """Create 72-hour input windows and matching 24-hour targets."""
    X = []
    y = []
    sequence_metadata = []
    minimum_rows = INPUT_HOURS + OUTPUT_HOURS

    for group_name, segment_id, segment_df in split_into_contiguous_groups(
        dataframe,
        split_name,
    ):
        if len(segment_df) < minimum_rows:
            continue

        features = segment_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        target = segment_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        last_start = len(segment_df) - minimum_rows + 1

        for start_index in range(last_start):
            input_end = start_index + INPUT_HOURS
            target_end = input_end + OUTPUT_HOURS

            X.append(features[start_index:input_end])
            y.append(target[input_end:target_end])
            sequence_metadata.append(
                {
                    "group": str(group_name),
                    "segment": int(segment_id),
                    "start_index": int(start_index),
                }
            )

    if not X:
        raise ValueError(
            f"No {split_name} sequences were created. Each continuous "
            f"city/location segment needs at least {minimum_rows} hourly rows."
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        sequence_metadata,
    )


def calculate_forecast_metrics(actual, predicted):
    """Calculate overall and per-horizon forecast errors."""
    errors = predicted - actual

    mae_by_hour = np.mean(np.abs(errors), axis=0)
    rmse_by_hour = np.sqrt(np.mean(np.square(errors), axis=0))
    bias_by_hour = np.mean(errors, axis=0)

    return {
        "overall_mae": float(np.mean(np.abs(errors))),
        "overall_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "overall_bias": float(np.mean(errors)),
        "max_absolute_error": float(np.max(np.abs(errors))),
        "mae_by_hour": mae_by_hour.astype(float).tolist(),
        "rmse_by_hour": rmse_by_hour.astype(float).tolist(),
        "bias_by_hour": bias_by_hour.astype(float).tolist(),
    }


def print_hourly_metrics(metric_values):
    print("\nForecast accuracy by horizon")
    print("-" * 64)

    for hour in range(OUTPUT_HOURS):
        print(
            f"Hour {hour + 1:2d} | "
            f"MAE: {metric_values['mae_by_hour'][hour]:6.3f} °C | "
            f"RMSE: {metric_values['rmse_by_hour'][hour]:6.3f} °C | "
            f"Bias: {metric_values['bias_by_hour'][hour]:+6.3f} °C"
        )


def save_prediction_table(actual, predicted, metadata):
    """Save one row per sample and forecast hour."""
    records = []

    for sample_index in range(len(actual)):
        sample_meta = metadata[sample_index]

        for horizon_index in range(OUTPUT_HOURS):
            actual_value = float(actual[sample_index, horizon_index])
            predicted_value = float(predicted[sample_index, horizon_index])

            records.append(
                {
                    "sample": sample_index,
                    "forecast_hour": horizon_index + 1,
                    "group": sample_meta["group"],
                    "segment": sample_meta["segment"],
                    "actual": actual_value,
                    "predicted": predicted_value,
                    "error": predicted_value - actual_value,
                }
            )

    pd.DataFrame(records).to_csv(PREDICTIONS_FILE, index=False)


# ------------------------------------
# Load and Validate Datasets
# ------------------------------------

print("Loading datasets...")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("TRAIN_FILE:", TRAIN_FILE)
print("Exists:", TRAIN_FILE.exists())

for required_file in [TRAIN_FILE, DEV_FILE, TEST_FILE]:
    if not required_file.exists():
        raise FileNotFoundError(f"Required data file not found: {required_file}")

train_df = validate_dataframe(pd.read_csv(TRAIN_FILE), "training data")
dev_df = validate_dataframe(pd.read_csv(DEV_FILE), "development data")
test_df = validate_dataframe(pd.read_csv(TEST_FILE), "testing data")


# ------------------------------------
# Create Sequences
# ------------------------------------

print("Creating sequences...")

X_train, y_train, train_metadata = create_sequences(
    train_df,
    "training data",
)
X_dev, y_dev, dev_metadata = create_sequences(
    dev_df,
    "development data",
)
X_test, y_test, test_metadata = create_sequences(
    test_df,
    "testing data",
)

print(f"Training samples    : {len(X_train)}")
print(f"Development samples : {len(X_dev)}")
print(f"Testing samples     : {len(X_test)}")
print(f"Input shape         : {X_train.shape}")
print(f"Target shape        : {y_train.shape}")


# ------------------------------------
# Normalization Layer
# ------------------------------------

normalizer = layers.Normalization(axis=-1, name="feature_normalization")
normalizer.adapt(
    X_train.reshape(-1, X_train.shape[-1])
)

print("Normalization complete.")


# ------------------------------------
# Build Model - Revised Existing LSTM
# ------------------------------------

inputs = tf.keras.Input(
    shape=(INPUT_HOURS, X_train.shape[-1]),
    name="weather_sequence",
)

# Keras Normalization supports the final feature axis directly, including
# three-dimensional sequence data.
x = normalizer(inputs)

x = layers.LSTM(
    48,
    return_sequences=True,
    dropout=0.15,
    kernel_regularizer=regularizers.l2(1e-4),
    name="lstm_1",
)(x)

x = layers.LayerNormalization(name="lstm_1_normalization")(x)

x = layers.LSTM(
    24,
    dropout=0.15,
    kernel_regularizer=regularizers.l2(1e-4),
    name="lstm_2",
)(x)

x = layers.LayerNormalization(name="lstm_2_normalization")(x)

x = layers.Dense(
    32,
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4),
    name="dense_features",
)(x)

x = layers.Dropout(0.20, name="dense_dropout")(x)

# The existing one-value output becomes a 24-value adjustment vector.
temperature_adjustment = layers.Dense(
    OUTPUT_HOURS,
    name="temperature_adjustment",
)(x)

if USE_PREVIOUS_DAY_BASELINE:
    # Last 24 observed temperatures represent the same hours from the
    # previous day. The LSTM learns how much the forecast should differ.
    previous_day_temperature = inputs[
        :,
        -OUTPUT_HOURS:,
        TEMPERATURE_FEATURE_INDEX,
    ]

    outputs = layers.Add(name="temperature_prediction")(
        [previous_day_temperature, temperature_adjustment]
    )
else:
    outputs = layers.Activation(
        "linear",
        name="temperature_prediction",
    )(temperature_adjustment)

model = Model(inputs=inputs, outputs=outputs, name="weather_lstm_24h")


# ------------------------------------
# Compile Once
# ------------------------------------

optimizer = tf.keras.optimizers.Adam(
    learning_rate=3e-4,
    clipnorm=1.0,
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.Huber(delta=1.5),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
    ],
)

print()
model.summary()

with open(MODEL_SUMMARY_FILE, "w", encoding="utf-8") as file:
    model.summary(print_fn=lambda line: file.write(line + "\n"))

print("\nModel compiled successfully.\n")


# ------------------------------------
# Callbacks
# ------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    min_delta=0.01,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_delta=0.005,
    min_lr=1e-6,
    verbose=1,
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_OUTPUT,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)


# ------------------------------------
# Train
# ------------------------------------

print("Beginning training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_dev, y_dev),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    shuffle=True,
    verbose=1,
)

history_df = pd.DataFrame(history.history)
history_df.to_csv(TRAINING_HISTORY_FILE, index=False)

print("\nTraining complete!\n")


# ------------------------------------
# Evaluate and Predict
# ------------------------------------

print("Evaluating on test set...\n")

evaluation = model.evaluate(
    X_test,
    y_test,
    verbose=0,
    return_dict=True,
)

predictions = model.predict(X_test, verbose=0)
predictions = np.asarray(predictions, dtype=np.float32)

forecast_metrics = calculate_forecast_metrics(y_test, predictions)

# Simple baselines provide a minimum standard the LSTM should beat.
last_observed_temperature = X_test[:, -1, TEMPERATURE_FEATURE_INDEX]
persistence_predictions = np.repeat(
    last_observed_temperature[:, np.newaxis],
    OUTPUT_HOURS,
    axis=1,
)
previous_day_predictions = X_test[
    :,
    -OUTPUT_HOURS:,
    TEMPERATURE_FEATURE_INDEX,
]

persistence_metrics = calculate_forecast_metrics(
    y_test,
    persistence_predictions,
)
previous_day_metrics = calculate_forecast_metrics(
    y_test,
    previous_day_predictions,
)

metrics = {
    "keras_evaluation": {
        key: float(value)
        for key, value in evaluation.items()
    },
    "lstm_forecast": forecast_metrics,
    "persistence_baseline": persistence_metrics,
    "previous_day_baseline": previous_day_metrics,
}

print(f"Test loss         : {evaluation['loss']:.4f}")
print(f"Test MAE          : {evaluation['mae']:.4f} °C")
print(f"Test RMSE         : {evaluation['rmse']:.4f} °C")
print(f"Overall bias      : {forecast_metrics['overall_bias']:+.4f} °C")
print(
    "Persistence MAE   : "
    f"{persistence_metrics['overall_mae']:.4f} °C"
)
print(
    "Previous-day MAE  : "
    f"{previous_day_metrics['overall_mae']:.4f} °C"
)

print_hourly_metrics(forecast_metrics)

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

save_prediction_table(y_test, predictions, test_metadata)

print("\nSample 24-hour forecast")
print("-" * 50)
for hour in range(OUTPUT_HOURS):
    print(
        f"Hour {hour + 1:2d}: "
        f"Predicted {predictions[0, hour]:7.2f} °C | "
        f"Actual {y_test[0, hour]:7.2f} °C"
    )


# ------------------------------------
# Save Model and Inference Configuration
# ------------------------------------

# EarlyStopping restored the best validation weights, so this final save
# stores the best model rather than the last training epoch.
model.save(MODEL_OUTPUT)
shutil.copy2(MODEL_OUTPUT, LATEST_MODEL_OUTPUT)

model_config = {
    "model_name": MODEL_NAME,
    "input_hours": INPUT_HOURS,
    "output_hours": OUTPUT_HOURS,
    "feature_columns": FEATURE_COLUMNS,
    "target_column": TARGET_COLUMN,
    "temperature_feature_index": TEMPERATURE_FEATURE_INDEX,
    "uses_previous_day_baseline": USE_PREVIOUS_DAY_BASELINE,
    "created_at": timestamp,
}

with open(CONFIG_FILE, "w", encoding="utf-8") as file:
    json.dump(model_config, file, indent=4)

shutil.copy2(CONFIG_FILE, LATEST_CONFIG_FILE)

print(f"\nModel saved to:\n{MODEL_OUTPUT}")
print(f"Latest model copied to:\n{LATEST_MODEL_OUTPUT}")


# ------------------------------------
# Plot Training History
# ------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Huber Loss")
plt.legend()
plt.grid(True)
plt.savefig(
    FIGURE_DIR / "loss_curve.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.title("Training vs Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error (°C)")
plt.legend()
plt.grid(True)
plt.savefig(
    FIGURE_DIR / "mae_curve.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ------------------------------------
# Plot 24-Hour Forecast Example
# ------------------------------------

forecast_hours = np.arange(1, OUTPUT_HOURS + 1)

plt.figure(figsize=(12, 6))
plt.plot(
    forecast_hours,
    y_test[0],
    marker="o",
    label="Actual Temperature",
)
plt.plot(
    forecast_hours,
    predictions[0],
    marker="o",
    label="Predicted Temperature",
)
plt.title("Example 24-Hour Temperature Forecast")
plt.xlabel("Forecast Hour")
plt.ylabel("Temperature (°C)")
plt.xticks(forecast_hours)
plt.legend()
plt.grid(True)
plt.savefig(
    FIGURE_DIR / "forecast_24h_example.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ------------------------------------
# Plot Error by Forecast Hour
# ------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(
    forecast_hours,
    forecast_metrics["mae_by_hour"],
    marker="o",
    label="LSTM MAE",
)
plt.plot(
    forecast_hours,
    previous_day_metrics["mae_by_hour"],
    marker="o",
    label="Previous-Day Baseline MAE",
)
plt.title("MAE by Forecast Horizon")
plt.xlabel("Forecast Hour")
plt.ylabel("Mean Absolute Error (°C)")
plt.xticks(forecast_hours)
plt.legend()
plt.grid(True)
plt.savefig(
    FIGURE_DIR / "mae_by_forecast_hour.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("\nFinished!")