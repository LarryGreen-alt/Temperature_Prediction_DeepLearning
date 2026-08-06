from __future__ import annotations

import os
from pathlib import Path

# Suppress TensorFlow INFO messages such as the oneDNN CPU notice.
# This must be set before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model


INPUT_HOURS = 72
OUTPUT_HOURS = 24

# Training defaults. The epoch count is only a maximum because EarlyStopping
# will restore the best validation-loss weights and end training when progress
# has stalled.
DEFAULT_MAX_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_LR_PATIENCE = 4
DEFAULT_MIN_DELTA = 1e-4

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
HOUR_SIN_INDEX = FEATURE_COLUMNS.index("hour_sin")
HOUR_COS_INDEX = FEATURE_COLUMNS.index("hour_cos")
DAY_SIN_INDEX = FEATURE_COLUMNS.index("day_sin")
DAY_COS_INDEX = FEATURE_COLUMNS.index("day_cos")


@tf.keras.utils.register_keras_serializable(package="Weather")
class TemperatureBaselines(layers.Layer):
    """
    Produce three useful baseline forecasts:

    1. Same 24 hours from the previous day.
    2. Persistence: repeat the latest observed temperature.
    3. A damped recent-trend forecast.

    The decoder learns how much to trust each baseline at every horizon.
    """

    def __init__(
        self,
        output_hours: int = OUTPUT_HOURS,
        feature_index: int = TEMPERATURE_FEATURE_INDEX,
        trend_lookback: int = 6,
        trend_decay_hours: float = 8.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_hours = int(output_hours)
        self.feature_index = int(feature_index)
        self.trend_lookback = int(trend_lookback)
        self.trend_decay_hours = float(trend_decay_hours)

    def call(self, inputs):
        temperature = inputs[:, :, self.feature_index]

        previous_day = temperature[:, -self.output_hours :]

        latest = temperature[:, -1:]
        persistence = tf.repeat(latest, repeats=self.output_hours, axis=1)

        earlier = temperature[:, -(self.trend_lookback + 1) : -self.trend_lookback]
        slope_per_hour = (latest - earlier) / tf.cast(
            self.trend_lookback,
            inputs.dtype,
        )

        horizon = tf.cast(
            tf.range(1, self.output_hours + 1)[tf.newaxis, :],
            inputs.dtype,
        )
        damping = tf.exp(
            -horizon / tf.cast(self.trend_decay_hours, inputs.dtype)
        )
        trend = persistence + slope_per_hour * horizon * damping

        return tf.stack(
            [previous_day, persistence, trend],
            axis=-1,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_hours": self.output_hours,
                "feature_index": self.feature_index,
                "trend_lookback": self.trend_lookback,
                "trend_decay_hours": self.trend_decay_hours,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Weather")
class FutureCalendarFeatures(layers.Layer):
    """
    Rotate the final observed cyclical time features into the next 24 hours.

    This gives the decoder the actual future time-of-day and approximate
    day-of-year position instead of relying only on a generic horizon number.
    """

    def __init__(
        self,
        output_hours: int = OUTPUT_HOURS,
        hour_sin_index: int = HOUR_SIN_INDEX,
        hour_cos_index: int = HOUR_COS_INDEX,
        day_sin_index: int = DAY_SIN_INDEX,
        day_cos_index: int = DAY_COS_INDEX,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_hours = int(output_hours)
        self.hour_sin_index = int(hour_sin_index)
        self.hour_cos_index = int(hour_cos_index)
        self.day_sin_index = int(day_sin_index)
        self.day_cos_index = int(day_cos_index)

    def call(self, inputs):
        dtype = inputs.dtype
        horizon = tf.cast(
            tf.range(1, self.output_hours + 1)[tf.newaxis, :],
            dtype,
        )

        last_hour_sin = inputs[:, -1, self.hour_sin_index][:, tf.newaxis]
        last_hour_cos = inputs[:, -1, self.hour_cos_index][:, tf.newaxis]
        last_day_sin = inputs[:, -1, self.day_sin_index][:, tf.newaxis]
        last_day_cos = inputs[:, -1, self.day_cos_index][:, tf.newaxis]

        hour_angle = horizon * tf.cast(2.0 * np.pi / 24.0, dtype)
        hour_cos_rotation = tf.cos(hour_angle)
        hour_sin_rotation = tf.sin(hour_angle)

        future_hour_sin = (
            last_hour_sin * hour_cos_rotation
            + last_hour_cos * hour_sin_rotation
        )
        future_hour_cos = (
            last_hour_cos * hour_cos_rotation
            - last_hour_sin * hour_sin_rotation
        )

        day_angle = horizon * tf.cast(2.0 * np.pi / (24.0 * 365.25), dtype)
        day_cos_rotation = tf.cos(day_angle)
        day_sin_rotation = tf.sin(day_angle)

        future_day_sin = (
            last_day_sin * day_cos_rotation
            + last_day_cos * day_sin_rotation
        )
        future_day_cos = (
            last_day_cos * day_cos_rotation
            - last_day_sin * day_sin_rotation
        )

        return tf.stack(
            [
                future_hour_sin,
                future_hour_cos,
                future_day_sin,
                future_day_cos,
            ],
            axis=-1,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_hours": self.output_hours,
                "hour_sin_index": self.hour_sin_index,
                "hour_cos_index": self.hour_cos_index,
                "day_sin_index": self.day_sin_index,
                "day_cos_index": self.day_cos_index,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Weather")
class HorizonEmbedding(layers.Layer):
    """Learn a separate representation for forecast hours 1 through 24."""

    def __init__(
        self,
        output_hours: int = OUTPUT_HOURS,
        embedding_dim: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_hours = int(output_hours)
        self.embedding_dim = int(embedding_dim)
        self.embedding = layers.Embedding(
            input_dim=self.output_hours,
            output_dim=self.embedding_dim,
        )

    def call(self, inputs):
        indices = tf.range(self.output_hours)
        embedded = self.embedding(indices)
        embedded = embedded[tf.newaxis, :, :]
        return tf.tile(embedded, [tf.shape(inputs)[0], 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_hours": self.output_hours,
                "embedding_dim": self.embedding_dim,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Weather")
class WeightedBaseline(layers.Layer):
    """Combine baseline candidates using learned per-hour softmax weights."""

    def call(self, inputs):
        baselines, weights = inputs
        return tf.reduce_sum(baselines * weights, axis=-1)


@tf.keras.utils.register_keras_serializable(package="Weather")
class LevelChangeHuber(tf.keras.losses.Loss):
    """
    Composite temperature loss.

    It penalizes:
    1. Temperature-level error at every forecast hour.
    2. Error in the hour-to-hour temperature change.

    Later forecast hours receive a modestly larger weight so the model does
    not optimize mainly for the easier first few hours.
    """

    def __init__(
        self,
        delta: float = 1.0,
        change_weight: float = 0.25,
        late_horizon_weight: float = 0.30,
        name: str = "level_change_huber",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.delta = float(delta)
        self.change_weight = float(change_weight)
        self.late_horizon_weight = float(late_horizon_weight)

    def _elementwise_huber(self, error: tf.Tensor) -> tf.Tensor:
        error = tf.convert_to_tensor(error)
        absolute_error = tf.abs(error)
        delta = tf.cast(self.delta, error.dtype)
        quadratic = tf.minimum(absolute_error, delta)
        linear = absolute_error - quadratic
        return 0.5 * tf.square(quadratic) + delta * linear

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)

        level_error = y_pred - y_true
        level_loss = self._elementwise_huber(level_error)

        horizon = tf.linspace(
            tf.cast(0.0, y_pred.dtype),
            tf.cast(1.0, y_pred.dtype),
            OUTPUT_HOURS,
        )
        horizon_weights = (
            1.0
            + tf.cast(self.late_horizon_weight, y_pred.dtype) * horizon
        )
        weighted_level_loss = tf.reduce_sum(
            level_loss * horizon_weights,
            axis=-1,
        ) / tf.reduce_sum(horizon_weights)

        true_change = y_true[:, 1:] - y_true[:, :-1]
        predicted_change = y_pred[:, 1:] - y_pred[:, :-1]
        change_loss = tf.reduce_mean(
            self._elementwise_huber(predicted_change - true_change),
            axis=-1,
        )

        return (
            weighted_level_loss
            + tf.cast(self.change_weight, y_pred.dtype) * change_loss
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "delta": self.delta,
                "change_weight": self.change_weight,
                "late_horizon_weight": self.late_horizon_weight,
            }
        )
        return config


def _make_optimizer(
    learning_rate: float,
    weight_decay: float,
) -> tf.keras.optimizers.Optimizer:
    """
    Prefer AdamW with weight decay and EMA. Fall back to Adam on older builds.
    """
    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=1.0,
            use_ema=True,
            ema_momentum=0.99,
        )
    except (AttributeError, TypeError):
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        )


def build_weather_model(
    X_train: np.ndarray,
    *,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
) -> Model:
    """
    Build a cross-attentive multi-baseline 72-to-24-hour forecast model.

    Main revisions:
    - Each forecast hour cross-attends to the full 72-hour encoded history.
    - The model retains the strong previous-day/persistence/trend baselines.
    - A level-and-change loss discourages overly smooth forecasts and gives
      modest extra importance to later forecast hours.
    """
    if X_train.ndim != 3:
        raise ValueError(
            "X_train must have shape (samples, hours, features); "
            f"received {X_train.shape}."
        )
    if X_train.shape[1] != INPUT_HOURS:
        raise ValueError(
            f"Expected {INPUT_HOURS} input hours; received {X_train.shape[1]}."
        )
    if X_train.shape[2] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} features; "
            f"received {X_train.shape[2]}."
        )

    normalizer = layers.Normalization(
        axis=-1,
        name="feature_normalization",
    )
    normalizer.adapt(X_train.reshape(-1, X_train.shape[-1]))

    inputs = tf.keras.Input(
        shape=(INPUT_HOURS, X_train.shape[-1]),
        name="weather_sequence",
    )
    x = normalizer(inputs)

    x = layers.Conv1D(
        48,
        kernel_size=5,
        padding="same",
        activation="swish",
        kernel_regularizer=regularizers.l2(8e-5),
        name="local_pattern_conv",
    )(x)
    x = layers.LayerNormalization(name="conv_normalization")(x)
    x = layers.SpatialDropout1D(
        0.12,
        name="conv_spatial_dropout",
    )(x)

    x = layers.LSTM(
        64,
        return_sequences=True,
        dropout=0.15,
        kernel_regularizer=regularizers.l2(8e-5),
        recurrent_regularizer=regularizers.l2(4e-5),
        name="history_lstm_1",
    )(x)
    x = layers.LSTM(
        40,
        return_sequences=True,
        dropout=0.15,
        kernel_regularizer=regularizers.l2(8e-5),
        recurrent_regularizer=regularizers.l2(4e-5),
        name="history_lstm_2",
    )(x)

    self_attention = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=20,
        dropout=0.15,
        name="history_self_attention",
    )(x, x)
    x = layers.Add(name="history_attention_residual")(
        [x, self_attention]
    )
    encoded_history = layers.LayerNormalization(
        name="encoded_history",
    )(x)

    average_context = layers.GlobalAveragePooling1D(
        name="average_context",
    )(encoded_history)
    maximum_context = layers.GlobalMaxPooling1D(
        name="maximum_context",
    )(encoded_history)
    latest_context = layers.Cropping1D(
        cropping=(INPUT_HOURS - 1, 0),
        name="latest_context_crop",
    )(encoded_history)
    latest_context = layers.Reshape(
        (40,),
        name="latest_context",
    )(latest_context)

    context = layers.Concatenate(name="combined_context")(
        [average_context, maximum_context, latest_context]
    )
    context = layers.Dense(
        80,
        activation="swish",
        kernel_regularizer=regularizers.l2(8e-5),
        name="context_dense",
    )(context)
    context = layers.Dropout(0.25, name="context_dropout")(context)

    horizon_embedding = HorizonEmbedding(
        output_hours=OUTPUT_HOURS,
        embedding_dim=12,
        name="horizon_embedding",
    )(context)
    future_calendar = FutureCalendarFeatures(
        output_hours=OUTPUT_HOURS,
        name="future_calendar",
    )(inputs)
    baselines = TemperatureBaselines(
        output_hours=OUTPUT_HOURS,
        feature_index=TEMPERATURE_FEATURE_INDEX,
        name="temperature_baselines",
    )(inputs)

    # Build one query per future hour and let that query inspect every
    # encoded historical hour. This preserves more temporal detail than
    # compressing the entire history into one repeated vector.
    forecast_queries = layers.Concatenate(name="forecast_query_features")(
        [horizon_embedding, future_calendar, baselines]
    )
    forecast_queries = layers.Dense(
        40,
        activation="swish",
        kernel_regularizer=regularizers.l2(8e-5),
        name="forecast_query_projection",
    )(forecast_queries)

    cross_attention = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=20,
        dropout=0.15,
        name="forecast_to_history_attention",
    )(
        query=forecast_queries,
        value=encoded_history,
        key=encoded_history,
    )
    forecast_queries = layers.Add(name="cross_attention_residual")(
        [forecast_queries, cross_attention]
    )
    forecast_queries = layers.LayerNormalization(
        name="cross_attention_normalization",
    )(forecast_queries)

    repeated_context = layers.RepeatVector(
        OUTPUT_HOURS,
        name="repeat_global_context",
    )(context)
    decoder_input = layers.Concatenate(name="decoder_input")(
        [
            forecast_queries,
            repeated_context,
            future_calendar,
            baselines,
        ]
    )
    decoder = layers.GRU(
        48,
        return_sequences=True,
        dropout=0.12,
        kernel_regularizer=regularizers.l2(8e-5),
        recurrent_regularizer=regularizers.l2(4e-5),
        name="forecast_decoder_gru",
    )(decoder_input)
    decoder = layers.Dense(
        40,
        activation="swish",
        kernel_regularizer=regularizers.l2(8e-5),
        name="decoder_dense",
    )(decoder)
    decoder = layers.Dropout(0.12, name="decoder_dropout")(decoder)

    baseline_weights = layers.Dense(
        3,
        activation="softmax",
        name="baseline_weights",
    )(decoder)
    blended_baseline = WeightedBaseline(
        name="blended_baseline",
    )([baselines, baseline_weights])

    correction = layers.Dense(
        1,
        kernel_regularizer=regularizers.l2(5e-5),
        name="temperature_correction",
    )(decoder)
    correction = layers.Reshape(
        (OUTPUT_HOURS,),
        name="temperature_correction_vector",
    )(correction)

    outputs = layers.Add(name="temperature_prediction")(
        [blended_baseline, correction]
    )

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="weather_lstm_cross_attention_24h",
    )
    model.compile(
        optimizer=_make_optimizer(learning_rate, weight_decay),
        loss=LevelChangeHuber(
            delta=1.0,
            change_weight=0.25,
            late_horizon_weight=0.30,
        ),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model



def get_custom_objects() -> dict[str, object]:
    return {
        "TemperatureBaselines": TemperatureBaselines,
        "Weather>TemperatureBaselines": TemperatureBaselines,
        "FutureCalendarFeatures": FutureCalendarFeatures,
        "Weather>FutureCalendarFeatures": FutureCalendarFeatures,
        "HorizonEmbedding": HorizonEmbedding,
        "Weather>HorizonEmbedding": HorizonEmbedding,
        "WeightedBaseline": WeightedBaseline,
        "Weather>WeightedBaseline": WeightedBaseline,
        "LevelChangeHuber": LevelChangeHuber,
        "Weather>LevelChangeHuber": LevelChangeHuber,
    }


def load_weather_model(
    model_path: str | Path,
    *,
    compile: bool = True,
) -> Model:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return tf.keras.models.load_model(
        path,
        custom_objects=get_custom_objects(),
        compile=compile,
    )


def build_training_callbacks(
    experiment_dir: str | Path,
    *,
    monitor: str = "val_loss",
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    lr_patience: int = DEFAULT_LR_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
) -> list[tf.keras.callbacks.Callback]:
    """Create callbacks for efficient and recoverable weather-model training.

    The checkpoint named ``weather_lstm.keras`` always contains the best full
    model observed according to ``monitor``. The in-memory model is also
    restored to those best weights when early stopping triggers.
    """
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1.")
    if lr_patience < 1:
        raise ValueError("lr_patience must be at least 1.")
    if lr_patience >= early_stopping_patience:
        raise ValueError(
            "lr_patience should be smaller than early_stopping_patience so "
            "the learning rate can decrease before training stops."
        )
    if min_delta < 0:
        raise ValueError("min_delta cannot be negative.")

    output_dir = Path(experiment_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "weather_lstm.keras"),
            monitor=monitor,
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="min",
            factor=0.5,
            patience=lr_patience,
            min_delta=min_delta,
            cooldown=1,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=early_stopping_patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(output_dir / "training_history.csv"),
            append=False,
        ),
    ]


def _validate_training_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    split_name: str,
) -> None:
    """Validate one input/target split before starting a long training run."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError(f"{split_name} X and y must both be NumPy arrays.")
    if X.ndim != 3:
        raise ValueError(
            f"{split_name} X must have shape (samples, {INPUT_HOURS}, features); "
            f"received {X.shape}."
        )
    if X.shape[1] != INPUT_HOURS:
        raise ValueError(
            f"{split_name} X must contain {INPUT_HOURS} input hours; "
            f"received {X.shape[1]}."
        )
    if X.shape[2] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"{split_name} X must contain {len(FEATURE_COLUMNS)} features; "
            f"received {X.shape[2]}."
        )
    if y.ndim != 2 or y.shape[1] != OUTPUT_HOURS:
        raise ValueError(
            f"{split_name} y must have shape (samples, {OUTPUT_HOURS}); "
            f"received {y.shape}."
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split_name} X and y contain different sample counts: "
            f"{X.shape[0]} and {y.shape[0]}."
        )
    if not np.isfinite(X).all():
        raise ValueError(f"{split_name} X contains NaN or infinite values.")
    if not np.isfinite(y).all():
        raise ValueError(f"{split_name} y contains NaN or infinite values.")


def train_weather_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    experiment_dir: str | Path,
    *,
    epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    lr_patience: int = DEFAULT_LR_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    verbose: int = 1,
) -> tf.keras.callbacks.History:
    """Train using time-series-safe defaults and retain the best model.

    Training uses ``shuffle=False`` so chronological samples are not randomly
    rearranged. ``epochs`` is the maximum; EarlyStopping may finish sooner.
    The best full model is written to ``experiment_dir/weather_lstm.keras``.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    _validate_training_data(X_train, y_train, split_name="Training")
    _validate_training_data(X_val, y_val, split_name="Validation")

    callbacks = build_training_callbacks(
        experiment_dir,
        early_stopping_patience=early_stopping_patience,
        lr_patience=lr_patience,
        min_delta=min_delta,
    )

    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
    )
