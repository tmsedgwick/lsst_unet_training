"""Noise- and PSF-aware multiband U-Net.

Encoder-decoder with skip connections over four scales. Each block is a residual pair of 3x3 convolutions with group
normalisation and swish activations. The PSF is not an image channel: a small encoder turns the per-band PSF stamps
into a 64-number embedding, which rescales and shifts every encoder feature map (FiLM, feature-wise linear
modulation), so the same weights adapt to any seeing. Five heads read the last decoder layer:

  galaxy_heatmap    probability of a galaxy centre at each pixel (the detection output)
  centroid_offset   sub-pixel offset from the peak pixel to the true centre
  source_structure  scaled log size, axis ratio, sin 2PA, cos 2PA
  clump_heatmap     star-forming clump centres (sub-structure head)
  tidal_heatmap     tidal blob centres (sub-structure head)

Weights are saved and loaded by layer name, so renaming or reordering layers makes saved models unloadable.
"""

import keras
import tensorflow as tf
from keras import layers, models

from .config import BANDS

HEATMAP_BIAS = -4.595  # sigmoid(-4.595) = 0.01: heatmaps start near "no source" everywhere


def normalisation_layer(filters, name):
    """Group normalisation with up to 8 groups that divide the filter count."""
    groups = min(8, filters)
    while filters % groups:
        groups -= 1
    return layers.GroupNormalization(groups=groups, axis=-1, name=name)


def conv_block(inputs, filters, name, dilation=1):
    """Residual block: conv-norm-swish-conv-norm, plus a (1x1-projected) shortcut, then swish."""
    x = layers.Conv2D(filters, 3, padding="same", dilation_rate=dilation, use_bias=False,
                      kernel_initializer="he_normal", name=f"{name}_conv1")(inputs)
    x = normalisation_layer(filters, f"{name}_norm1")(x)
    x = layers.Activation("swish", name=f"{name}_act1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal",
                      name=f"{name}_conv2")(x)
    x = normalisation_layer(filters, f"{name}_norm2")(x)
    shortcut = inputs
    if inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_shortcut")(shortcut)
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("swish", name=f"{name}_out")(x)


def film(features, psf_embedding, filters, name):
    """FiLM: features * (1 + gamma) + beta, with gamma and beta predicted from the PSF embedding."""
    gamma = layers.Dense(filters, name=f"{name}_gamma")(psf_embedding)
    gamma = layers.Reshape((1, 1, filters), name=f"{name}_gamma_r")(gamma)
    beta = layers.Dense(filters, name=f"{name}_beta")(psf_embedding)
    beta = layers.Reshape((1, 1, filters), name=f"{name}_beta_r")(beta)
    one_plus_gamma = layers.Lambda(lambda t: 1.0 + t, name=f"{name}_gp1")(gamma)
    return layers.Add(name=f"{name}_film")([layers.Multiply(name=f"{name}_scale")([features, one_plus_gamma]), beta])


def psf_encoder(psf_stamps):
    """Per-band PSF stamps -> 64-number embedding."""
    x = layers.Conv2D(16, 3, padding="same", activation="swish", name="psf_c1")(psf_stamps)
    x = layers.Conv2D(32, 3, padding="same", activation="swish", name="psf_c2")(x)
    x = layers.GlobalAveragePooling2D(name="psf_gap")(x)
    return layers.Dense(64, activation="swish", name="psf_embed")(x)


def decoder_block(inputs, skip, filters, name):
    """Upsample x2, project, concatenate the encoder skip, then a conv block."""
    x = layers.UpSampling2D(size=2, interpolation="bilinear", name=f"{name}_up")(inputs)
    x = layers.Conv2D(filters, 1, padding="same", name=f"{name}_project")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    return conv_block(x, filters, name=f"{name}_conv")


def heatmap_head(features, base_filters, name):
    x = layers.Conv2D(base_filters, 3, padding="same", activation="swish", name=f"{name}_pre")(features)
    bias = keras.initializers.Constant(HEATMAP_BIAS)
    return layers.Conv2D(1, 1, activation="sigmoid", name=name, bias_initializer=bias)(x)  # pyright: ignore


def build_unet(cfg):
    """The detector as a Keras model with inputs image_planes and psf_kernels and the five heads as outputs."""
    filters, input_size = cfg["base_filters"], cfg["tile_size"] + 2 * cfg["tile_halo"]
    image_in = layers.Input((input_size, input_size, 2 * len(BANDS)), name="image_planes")
    psf_in = layers.Input((cfg["psf_stamp"], cfg["psf_stamp"], len(BANDS)), name="psf_kernels")
    psf_embedding = psf_encoder(psf_in)

    enc1 = film(conv_block(image_in, filters, "enc1"), psf_embedding, filters, "enc1_film")
    enc2 = film(conv_block(layers.MaxPool2D()(enc1), filters * 2, "enc2"), psf_embedding, filters * 2, "enc2_film")
    enc3 = film(conv_block(layers.MaxPool2D()(enc2), filters * 4, "enc3"), psf_embedding, filters * 4, "enc3_film")
    enc4 = film(conv_block(layers.MaxPool2D()(enc3), filters * 8, "enc4"), psf_embedding, filters * 8, "enc4_film")
    bottleneck = conv_block(layers.MaxPool2D()(enc4), filters * 12, "bottleneck", dilation=2)
    bottleneck = layers.SpatialDropout2D(0.15)(film(bottleneck, psf_embedding, filters * 12, "bottleneck_film"))

    dec4 = decoder_block(bottleneck, enc4, filters * 8, "dec4")
    dec3 = decoder_block(dec4, enc3, filters * 4, "dec3")
    dec2 = decoder_block(dec3, enc2, filters * 2, "dec2")
    dec1 = decoder_block(dec2, enc1, filters, "dec1")

    outputs = {
        "galaxy_heatmap": heatmap_head(dec1, filters, "galaxy_heatmap"),
        "centroid_offset": layers.Conv2D(2, 1, activation="tanh", name="centroid_offset")(dec1),
        "source_structure": layers.Conv2D(4, 1, activation=None, name="source_structure")(dec1),
        "clump_heatmap": heatmap_head(dec1, filters, "clump_heatmap"),
        "tidal_heatmap": heatmap_head(dec1, filters, "tidal_heatmap"),
    }
    return models.Model(inputs={"image_planes": image_in, "psf_kernels": psf_in}, outputs=outputs,
                        name="mep_multiband_unet")


def focal_loss(y_true, y_pred):
    """Penalty-reduced focal loss for centre heatmaps (as in CenterNet). y_true = [heatmap, centre weight, valid].

    Centre pixels (heatmap = 1) are positives, weighted by their population weight; every other pixel is a negative,
    down-weighted by (1 - heatmap)^4 near a centre. Only valid (non-halo) pixels count.
    """
    target, centre_weight, valid = (tf.cast(y_true[..., i:i + 1], tf.float32) for i in range(3))
    prediction = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1 - 1e-6)
    positive = tf.cast(target >= 0.999, tf.float32)
    positive_loss = -tf.math.log(prediction) * tf.pow(1.0 - prediction, 2.0) * positive * centre_weight * valid
    negative_loss = (-tf.math.log(1.0 - prediction) * tf.pow(prediction, 2.0) * tf.pow(1.0 - target, 4.0)
                     * (1.0 - positive) * valid)
    normaliser = tf.maximum(tf.reduce_sum(positive * centre_weight * valid), 1.0)
    return (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / normaliser


def masked_huber(y_true, y_pred):
    """Huber loss on regression outputs, counted only where the weight channel (last in y_true) is set."""
    n_output = tf.shape(y_pred)[-1]
    target = tf.cast(y_true[..., :n_output], tf.float32)
    weight = tf.cast(y_true[..., n_output:n_output + 1], tf.float32)
    residual = tf.cast(y_pred, tf.float32) - target
    loss = tf.where(tf.abs(residual) < 1.0, 0.5 * residual ** 2, tf.abs(residual) - 0.5)
    return tf.reduce_sum(loss * weight) / tf.maximum(tf.reduce_sum(weight), 1.0)


def compile_unet(model, cfg):
    """Adam with gradient clipping; the sub-structure (clump and tidal) heads can be switched off in cfg."""
    loss_weights = dict(cfg["loss_weights"])
    loss_weights["clump_heatmap"] *= cfg["clump_head"]
    loss_weights["tidal_heatmap"] *= cfg["tidal_head"]
    losses = {name: focal_loss if name.endswith("heatmap") else masked_huber for name in loss_weights}
    model.compile(optimizer=keras.optimizers.Adam(cfg["learning_rate"], clipnorm=5.0), loss=losses,
                  loss_weights=loss_weights)
    return model
