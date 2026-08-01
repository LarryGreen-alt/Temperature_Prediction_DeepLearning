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
    Build a regularized multi-baseline LSTM forecast model.

    The previous architecture was overfitting: training error kept improving
    while validation error worsened. This version is smaller, more strongly
    regularized, and can move away from the previous-day pattern during fronts.
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

    # All 72 input hours are historical, so "same" padding is safe here.
    x = layers.Conv1D(
        48,
        kernel_size=5,
        padding="same",
        activation="swish",
        kernel_regularizer=regularizers.l2(1e-4),
        name="local_pattern_conv",
    )(x)
    x = layers.LayerNormalization(name="conv_normalization")(x)
    x = layers.SpatialDropout1D(
        0.15,
        name="conv_spatial_dropout",
    )(x)

    x = layers.LSTM(
        64,
        return_sequences=True,
        dropout=0.20,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(5e-5),
        name="history_lstm_1",
    )(x)
    x = layers.LSTM(
        32,
        return_sequences=True,
        dropout=0.20,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(5e-5),
        name="history_lstm_2",
    )(x)

    attention = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=16,
        dropout=0.20,
        name="history_attention",
    )(x, x)
    x = layers.Add(name="attention_residual")([x, attention])
    x = layers.LayerNormalization(name="attention_normalization")(x)

    average_context = layers.GlobalAveragePooling1D(
        name="average_context",
    )(x)
    maximum_context = layers.GlobalMaxPooling1D(
        name="maximum_context",
    )(x)
    latest_context = layers.Cropping1D(
        cropping=(INPUT_HOURS - 1, 0),
        name="latest_context_crop",
    )(x)
    latest_context = layers.Reshape(
        (32,),
        name="latest_context",
    )(latest_context)

    context = layers.Concatenate(name="combined_context")(
        [average_context, maximum_context, latest_context]
    )
    context = layers.Dense(
        96,
        activation="swish",
        kernel_regularizer=regularizers.l2(1e-4),
        name="context_dense_1",
    )(context)
    context = layers.Dropout(0.35, name="context_dropout_1")(context)
    context = layers.Dense(
        64,
        activation="swish",
        kernel_regularizer=regularizers.l2(1e-4),
        name="context_dense_2",
    )(context)
    context = layers.Dropout(0.25, name="context_dropout_2")(context)

    repeated_context = layers.RepeatVector(
        OUTPUT_HOURS,
        name="repeat_context",
    )(context)
    horizon_embedding = HorizonEmbedding(
        output_hours=OUTPUT_HOURS,
        embedding_dim=8,
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

    decoder_input = layers.Concatenate(name="decoder_input")(
        [
            repeated_context,
            horizon_embedding,
            future_calendar,
            baselines,
        ]
    )
    decoder = layers.LSTM(
        48,
        return_sequences=True,
        dropout=0.15,
        kernel_regularizer=regularizers.l2(1e-4),
        name="forecast_decoder_lstm",
    )(decoder_input)
    decoder = layers.Dense(
        48,
        activation="swish",
        kernel_regularizer=regularizers.l2(1e-4),
        name="decoder_dense",
    )(decoder)
    decoder = layers.Dropout(0.15, name="decoder_dropout")(decoder)

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
        kernel_regularizer=regularizers.l2(1e-4),
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
        name="weather_lstm_multibaseline_24h",
    )
    model.compile(
        optimizer=_make_optimizer(learning_rate, weight_decay),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def get_custom_objects() -> dict[str, type[layers.Layer]]:
    return {
        "TemperatureBaselines": TemperatureBaselines,
        "Weather>TemperatureBaselines": TemperatureBaselines,
        "FutureCalendarFeatures": FutureCalendarFeatures,
        "Weather>FutureCalendarFeatures": FutureCalendarFeatures,
        "HorizonEmbedding": HorizonEmbedding,
        "Weather>HorizonEmbedding": HorizonEmbedding,
        "WeightedBaseline": WeightedBaseline,
        "Weather>WeightedBaseline": WeightedBaseline,
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