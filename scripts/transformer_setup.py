# transformer_setup.py
# PAT encoder architecture — exactly mirrors the original PAT tutorial notebook
# so that pretrained weights load without shape or count mismatches.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from .config import PAT_NUM_LAYERS, PAT_INPUT_SIZE, PAT_PATCH_SIZE, PAT_EMBED_DIM


def get_positional_embeddings(num_patches: int, embed_dim: int) -> tf.Tensor:
    """Sine/cosine positional embeddings matching the PAT notebook implementation."""
    position  = tf.range(num_patches, dtype=tf.float32)[:, tf.newaxis]
    div_term  = tf.exp(
        tf.range(0, embed_dim, 2, dtype=tf.float32)
        * (-tf.math.log(10000.0) / embed_dim)
    )
    pos_embeddings = tf.concat(
        [tf.sin(position * div_term), tf.cos(position * div_term)], axis=-1
    )
    return pos_embeddings


def TransformerBlock(
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    rate: float = 0.1,
    name_prefix: str = "encoder_layer_1",
) -> tf.keras.Model:
    """
    Functional-API transformer block matching the PAT notebook exactly.

    Layer names inside the block:
      {name_prefix}_input       - Input
      {name_prefix}_attention   - MultiHeadAttention
      {name_prefix}_dropout     - Dropout after attention
      {name_prefix}_norm1       - LayerNormalization (post-attention)
      {name_prefix}_ff1         - Dense(ff_dim, relu)
      {name_prefix}_ff2         - Dense(embed_dim)
      {name_prefix}_dropout2    - Dropout after FFN
      {name_prefix}_norm2       - LayerNormalization (post-FFN)

    Returns a Model with outputs [final_output, attention_weights].
    """
    input_layer = layers.Input(
        shape=(None, embed_dim), name=f"{name_prefix}_input"
    )

    attention_layer = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        name=f"{name_prefix}_attention",
    )
    attention_output, attention_weights = attention_layer(
        input_layer, input_layer, return_attention_scores=True
    )
    attention_output = layers.Dropout(
        rate, name=f"{name_prefix}_dropout"
    )(attention_output)

    out1 = layers.LayerNormalization(
        epsilon=1e-6, name=f"{name_prefix}_norm1"
    )(input_layer + attention_output)

    ff_output = layers.Dense(
        ff_dim, activation="relu", name=f"{name_prefix}_ff1"
    )(out1)
    ff_output = layers.Dense(
        embed_dim, name=f"{name_prefix}_ff2"
    )(ff_output)
    ff_output = layers.Dropout(
        rate, name=f"{name_prefix}_dropout2"
    )(ff_output)

    final_output = layers.LayerNormalization(
        epsilon=1e-6, name=f"{name_prefix}_norm2"
    )(out1 + ff_output)

    return models.Model(
        inputs=input_layer,
        outputs=[final_output, attention_weights],
        name=f"{name_prefix}_transformer",
    )


def build_encoder_for_extraction(
    input_size: int = PAT_INPUT_SIZE,
    patch_size: int = PAT_PATCH_SIZE,
    embed_dim:  int = PAT_EMBED_DIM,
    num_layers: int = PAT_NUM_LAYERS,
    ff_dim:     int = 256,
    num_heads:  int = 12,
    rate:       float = 0.1,
) -> tf.keras.Model:
    """
    Builds the PAT encoder for feature extraction, matching the weight layout
    of PAT-L_29k_weights.h5 exactly.

    Encoder layers are named encoder_layer_1_transformer through
    encoder_layer_{num_layers}_transformer, matching the saved weight keys.

    Architecture
    ------------
    inputs (10 080,)
    -> reshape  (1120, 9)
    -> dense    (1120, 96)           [named "dense"]
    -> + positional embeddings
    -> encoder_layer_1_transformer  (functional TransformerBlock)
    -> ...
    -> encoder_layer_N_transformer
    -> GlobalAveragePooling1D
    -> embedding vector (96,)
    """
    num_patches = input_size // patch_size
    inputs      = layers.Input(shape=(input_size,), name="inputs")

    x = layers.Reshape((num_patches, patch_size), name="reshape")(inputs)
    x = layers.Dense(embed_dim, name="dense")(x)
    x = x + get_positional_embeddings(num_patches, embed_dim)

    for i in range(num_layers):
        # name_prefix matches the saved weight keys: encoder_layer_1, encoder_layer_2, ...
        x, _ = TransformerBlock(
            embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            rate=rate,
            name_prefix=f"encoder_layer_{i + 1}",
        )(x)

    embeddings = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    return models.Model(
        inputs=inputs, outputs=embeddings, name="PAT_Feature_Extractor"
    )