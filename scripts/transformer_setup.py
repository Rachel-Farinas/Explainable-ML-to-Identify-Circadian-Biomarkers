# transformer_setup.py
# PAT (Pretrained Actigraphy Transformer) architecture definition:
# TransformerBlock, positional embeddings, and encoder builder.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from .config import PAT_NUM_LAYERS


def get_positional_embeddings(num_patches: int, embed_dim: int) -> tf.Tensor:
    """
    Generates a fixed sine/cosine positional encoding matrix as used in the
    PAT paper.

    Parameters
    ----------
    num_patches : number of patches (sequence length)
    embed_dim   : embedding dimensionality

    Returns
    -------
    tf.Tensor of shape (1, num_patches, embed_dim)
    """
    pos         = np.arange(num_patches)[:, np.newaxis]
    i           = np.arange(embed_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
    angle_rads  = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class TransformerBlock(layers.Layer):
    """
    Single transformer encoder block: multi-head self-attention followed by a
    position-wise feed-forward network, with residual connections and layer
    normalization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        rate: float = 0.1,
        name_prefix: str = "transformer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.att       = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            name=f"{name_prefix}_attn",
        )
        self.ffn = models.Sequential(
            [
                layers.Dense(ff_dim, activation="relu", name=f"{name_prefix}_ffn_1"),
                layers.Dense(embed_dim, name=f"{name_prefix}_ffn_2"),
            ],
            name=f"{name_prefix}_ffn",
        )
        self.layernorm1 = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name_prefix}_norm1"
        )
        self.layernorm2 = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name_prefix}_norm2"
        )
        self.dropout1 = layers.Dropout(rate, name=f"{name_prefix}_drop1")
        self.dropout2 = layers.Dropout(rate, name=f"{name_prefix}_drop2")

    def call(self, inputs, training: bool = False):
        attn_output, weights = self.att(
            inputs, inputs,
            return_attention_scores=True,
            training=training,
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1        = self.layernorm1(inputs + attn_output)
        ffn_output  = self.ffn(out1, training=training)
        ffn_output  = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_encoder_for_extraction(
    input_size: int = 10080,
    patch_size: int = 144,
    embed_dim:  int = 128,
    num_layers: int = 5,
) -> tf.keras.Model:
    """
    Constructs the PAT encoder model used for feature extraction.

    Architecture
    ------------
    Raw actigraphy (10 080 min)
    → Reshape into patches (70 × 144)
    → Linear projection (70 × 128)
    → + Positional embeddings
    → N × TransformerBlock
    → GlobalAveragePooling1D  →  embedding vector (128,)

    Returns
    -------
    A compiled Keras Model named 'PAT_Feature_Extractor'.
    """
    num_patches = input_size // patch_size
    inputs      = layers.Input(shape=(input_size,), name="actigraphy_input")

    x = layers.Reshape((num_patches, patch_size), name="patch_reshape")(inputs)
    x = layers.Dense(embed_dim, name="patch_projection")(x)

    x = x + get_positional_embeddings(num_patches, embed_dim)

    for i in range(num_layers):
        x = TransformerBlock(
            embed_dim,
            num_heads=4,
            ff_dim=256,
            name=f"transformer_block_{i}",
        )(x)

    embeddings = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    return models.Model(
        inputs=inputs, outputs=embeddings, name="PAT_Feature_Extractor"
    )
