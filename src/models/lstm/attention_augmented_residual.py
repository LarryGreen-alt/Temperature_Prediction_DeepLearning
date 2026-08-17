from __future__ import annotations

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


@tf.keras.utils.register_keras_serializable(package="WeatherNotebook")
class PreviousDayTemperature(layers.Layer):
    """Return the last 24 observed temperatures as the residual baseline."""

    def __init__(
        self,
        output_hours: int = OUTPUT_HOURS,
        feature_index: int = TEMPERATURE_FEATURE_INDEX,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_hours = int(output_hours)
        self.feature_index = int(feature_index)

    def call(self, inputs):
        return inputs[:, -self.output_hours :, self.feature_index]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_hours": self.output_hours,
                "feature_index": self.feature_index,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="WeatherNotebook")
class HorizonEmbedding(layers.Layer):
    """Learn a separate decoder embedding for each future hour."""

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
        embedded = self.embedding(indices)[tf.newaxis, :, :]
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


def _validate_training_tensor(X_train: np.ndarray) -> None:
    if not isinstance(X_train, np.ndarray) or X_train.ndim != 3:
        raise ValueError(
            "X_train must have shape (samples, 72, 11)."
        )
    if X_train.shape[1:] != (INPUT_HOURS, len(FEATURE_COLUMNS)):
        raise ValueError(
            f"Expected (*, {INPUT_HOURS}, {len(FEATURE_COLUMNS)}); "
            f"received {X_train.shape}."
        )
    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN or infinite values.")


def build_attention_augmented_residual_model(
    X_train: np.ndarray,
    *,
    learning_rate: float = 5e-4,
) -> Model:
    """Build the 212,232-parameter CNN + LSTM + self-attention model."""
    _validate_training_tensor(X_train)

    normalizer = layers.Normalization(
        axis=-1,
        name="feature_normalization",
    )
    normalizer.adapt(X_train.reshape(-1, X_train.shape[-1]))

    inputs = tf.keras.Input(
        shape=(INPUT_HOURS, len(FEATURE_COLUMNS)),
        name="weather_sequence",
    )
    x = normalizer(inputs)

    causal = layers.Conv1D(
        64,
        kernel_size=5,
        padding="causal",
        activation="swish",
        kernel_regularizer=regularizers.l2(5e-5),
        name="causal_conv",
    )(x)
    local = layers.SeparableConv1D(
        64,
        kernel_size=3,
        padding="same",
        activation="swish",
        depthwise_regularizer=regularizers.l2(5e-5),
        pointwise_regularizer=regularizers.l2(5e-5),
        name="local_weather_block",
    )(causal)
    local = layers.Dropout(0.10, name="conv_dropout")(local)
    x = layers.Add(name="conv_residual_add")([causal, local])
    x = layers.LayerNormalization(name="conv_normalization")(x)

    x = layers.LSTM(
        96,
        return_sequences=True,
        dropout=0.10,
        kernel_regularizer=regularizers.l2(5e-5),
        name="lstm_encoder",
    )(x)

    attention = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=24,
        dropout=0.10,
        name="history_attention",
    )(x, x)
    x = layers.Add(name="attention_residual_add")([x, attention])
    x = layers.LayerNormalization(name="attention_normalization")(x)

    feed_forward = layers.Dense(
        192,
        activation="swish",
        name="attention_ff_1",
    )(x)
    feed_forward = layers.Dropout(
        0.10,
        name="attention_ff_dropout",
    )(feed_forward)
    feed_forward = layers.Dense(
        96,
        name="attention_ff_2",
    )(feed_forward)
    x = layers.Add(name="feed_forward_residual_add")([x, feed_forward])
    encoded = layers.LayerNormalization(
        name="encoder_output_normalization",
    )(x)

    latest = layers.Cropping1D(
        cropping=(INPUT_HOURS - 1, 0),
        name="latest_timestep_crop",
    )(encoded)
    average = layers.GlobalAveragePooling1D(
        name="average_context",
    )(encoded)
    maximum = layers.GlobalMaxPooling1D(
        name="maximum_context",
    )(encoded)
    latest = layers.Reshape((96,), name="latest_context")(latest)

    context = layers.Concatenate(name="combined_context")(
        [average, maximum, latest]
    )
    context = layers.Dense(
        160,
        activation="swish",
        kernel_regularizer=regularizers.l2(5e-5),
        name="forecast_features_1",
    )(context)
    context = layers.Dropout(0.20, name="forecast_dropout_1")(context)
    context = layers.Dense(
        80,
        activation="swish",
        kernel_regularizer=regularizers.l2(5e-5),
        name="forecast_features_2",
    )(context)
    context = layers.Dropout(0.10, name="forecast_dropout_2")(context)

    previous_day = PreviousDayTemperature(
        output_hours=OUTPUT_HOURS,
        feature_index=TEMPERATURE_FEATURE_INDEX,
        name="previous_day_temperature",
    )(inputs)
    repeated_context = layers.RepeatVector(
        OUTPUT_HOURS,
        name="repeat_context",
    )(context)
    horizon = HorizonEmbedding(
        output_hours=OUTPUT_HOURS,
        embedding_dim=12,
        name="horizon_embedding",
    )(context)
    previous_day_expanded = layers.Reshape(
        (OUTPUT_HOURS, 1),
        name="previous_day_expanded",
    )(previous_day)

    decoder = layers.Concatenate(name="horizon_decoder_input")(
        [repeated_context, horizon, previous_day_expanded]
    )
    decoder = layers.Dense(
        64,
        activation="swish",
        name="decoder_dense_1",
    )(decoder)
    decoder = layers.Dropout(0.10, name="decoder_dropout")(decoder)
    decoder = layers.Dense(
        32,
        activation="swish",
        name="decoder_dense_2",
    )(decoder)
    adjustment = layers.Dense(
        1,
        name="temperature_adjustment_per_hour",
    )(decoder)
    adjustment = layers.Reshape(
        (OUTPUT_HOURS,),
        name="temperature_adjustment",
    )(adjustment)
    outputs = layers.Add(name="temperature_prediction")(
        [previous_day, adjustment]
    )

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="weather_lstm_attention_24h",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.Huber(),
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def get_custom_objects() -> dict[str, object]:
    return {
        "PreviousDayTemperature": PreviousDayTemperature,
        "WeatherNotebook>PreviousDayTemperature": PreviousDayTemperature,
        "HorizonEmbedding": HorizonEmbedding,
        "WeatherNotebook>HorizonEmbedding": HorizonEmbedding,
    }
