import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model


def positional_encoding(length, depth):
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.zeros((length, depth))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.constant(pos_encoding, dtype=tf.float32)


def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate):

    # Self-attention sub-layer
    residual = x
    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LayerNormalization()(residual + x)

    # Feed-forward sub-layer
    residual = x
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dense(d_model)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LayerNormalization()(residual + x)

    return x


def build_model(
    window_size,
    num_features,
    d_model,
    num_heads,
    num_encoder_layers,
    ff_dim,
    dropout_rate,
    normalizer,
    num_cities=None,
    city_embed_dim=8
):
    """Assembles the Transformer architecture. `normalizer` must already be
    constructed (and adapted, if the caller wants normalization to be
    meaningful) — this function only wires up the graph, it never touches
    data. Returns an uncompiled model.

    `num_cities`, if set, adds a second `city_id` input: a learned embedding
    is projected to `d_model` and added to every timestep of the weather
    projection (the same way positional encoding is added), before entering
    the encoder blocks. The city id bypasses `normalizer` entirely — it's
    categorical, not a continuous quantity to be z-scored. Left at its
    default of None, the graph is identical to the single-input version."""

    inputs = tf.keras.Input(shape=(window_size, num_features), name="weather_sequence")

    x = layers.TimeDistributed(normalizer)(inputs)
    x = layers.Dense(d_model)(x)
    x = x + positional_encoding(window_size, d_model)

    model_inputs = inputs
    if num_cities is not None:
        city_input = tf.keras.Input(shape=(), dtype="int32", name="city_id")
        city_vec = layers.Embedding(num_cities, city_embed_dim, name="city_embedding")(city_input)
        city_vec = layers.Dense(d_model, name="city_projection")(city_vec)
        x = x + city_vec[:, tf.newaxis, :]
        model_inputs = [inputs, city_input]

    for _ in range(num_encoder_layers):
        x = transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, name="temperature_prediction")(x)

    return Model(model_inputs, outputs)
