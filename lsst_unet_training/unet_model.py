"""Noise- and PSF-aware multiband U-Net.

Encoder-decoder with skip connections over four scales. Each block is a residual pair of 3x3 convolutions with group
normalisation and swish activations. The PSF is not an image channel: a small encoder turns the per-band PSF stamps
into a 64-number embedding, which rescales and shifts every encoder feature map (FiLM, feature-wise linear
modulation), so the same weights adapt to any seeing. Heads read the last decoder layer (cfg["heads"] chooses which
of the optional ones a model has):

  galaxy_heatmap      probability of a galaxy centre at each pixel
  star_heatmap        probability of a star centre (optional)
  sfregion_map           probability that a pixel holds detectable light of a star-forming region (optional)
  tidal_map           probability that a pixel holds detectable light of a tidal stream or shell (optional)
  spike_map           probability that a pixel lies on a diffraction spike (optional)
  centroid_offset     sub-pixel offset from the peak pixel to the true centre
  source_structure    scaled log size, axis ratio, sin 2PA, cos 2PA
  detection_heatmap   probability of a source (galaxy or star) centre, from the decoder features and all the maps
                      above (optional): the output detections are taken from

Two more optional heads, sfregion_heatmap and tidal_heatmap, mark the centres of star-forming regions and tidal
blobs; when present they also feed the detection head.

The detection head starts as an exact copy of the galaxy heatmap (its learned correction starts at zero), so adding
it to a trained model changes nothing until it is trained; it then learns to add stars and to reject peaks that the
other maps say are star-forming regions, tidal features or spikes.

Missing bands: a band without data is fed as "no data" (zero S/N, the clipped maximum log variance), like the area
beyond the image edge. With cfg["band_adapter"], a small adapter network reads which bands have data at each pixel
and corrects the features after every FiLM layer: features + m * (features * gamma + beta), where gamma and beta come
from the adapter and m is the fraction of bands missing at that pixel where some, but not all, have data (0 where
all bands are present and where none are). Its outputs with all six bands are therefore exactly the backbone's. The
adapter's gamma and beta start at zero, so a new adapter changes nothing until it is trained.

Weights are saved and loaded by layer name and creation order, so renaming or reordering layers makes saved models
unloadable; new heads are always created after the original ones.
"""

import keras
import numpy as np
import tensorflow as tf
from keras import layers, models

from .config import BANDS

# Heads that are maps of the image and feed the detection head, in the order they enter it.
MAP_HEADS = ("galaxy_heatmap", "star_heatmap", "sfregion_map", "tidal_map", "spike_map", "sfregion_heatmap",
             "tidal_heatmap")

HEATMAP_BIAS = -4.595  # sigmoid(-4.595) = 0.01: heatmaps start near "no source" everywhere
COVERED_BELOW = 7.99  # a band has data where its normalised log-variance plane is below its clip value of 8


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


def probability_logit(p):
    """log(p / (1 - p)) of a probability map, clipped away from 0 and 1."""
    p = keras.ops.clip(p, 1e-6, 1.0 - 1e-6)
    return keras.ops.subtract(keras.ops.log(p), keras.ops.log(keras.ops.subtract(1.0, p)))


def detection_head(features, maps, base_filters):
    """Source-centre map from the decoder features and the other heads' maps (see the module docstring).

    logit = direct(logits of the maps) + correction(features, logits of the maps). direct starts as the identity on
    the galaxy map and correction at zero, so the head starts equal to the galaxy heatmap.
    """
    names = [name for name in MAP_HEADS if name in maps]
    logits = [layers.Lambda(probability_logit, name=f"detection_logit_{name}")(maps[name]) for name in names]
    stacked = layers.Concatenate(name="detection_maps")(logits) if len(logits) > 1 else logits[0]
    identity = np.zeros((1, 1, len(names), 1), np.float32)
    identity[0, 0, names.index("galaxy_heatmap"), 0] = 1.0
    identity_initializer = keras.initializers.Constant(identity)  # pyright: ignore[reportArgumentType]
    direct = layers.Conv2D(1, 1, name="detection_direct", bias_initializer="zeros",
                           kernel_initializer=identity_initializer)(stacked)  # pyright: ignore[reportArgumentType]
    context = layers.Concatenate(name="detection_context")([features, stacked])
    hidden = layers.Conv2D(base_filters, 3, padding="same", activation="swish", name="detection_pre")(context)
    correction = layers.Conv2D(1, 1, name="detection_correction", kernel_initializer="zeros",
                               bias_initializer="zeros")(hidden)
    logit = layers.Add(name="detection_sum")([direct, correction])
    return layers.Activation("sigmoid", name="detection_heatmap")(logit)


def band_coverage(image_planes):
    """(H, W, n_band) 1 where a band has data, from the log-variance planes (every second input plane)."""
    return keras.ops.cast(image_planes[..., 1::2] < COVERED_BELOW, "float32")


def missing_fraction(image_planes):
    """(H, W, 1) fraction of bands missing where some, but not all, bands have data; 0 elsewhere."""
    covered = band_coverage(image_planes)
    n_band = len(BANDS)
    n_missing = n_band - keras.ops.sum(covered, axis=-1, keepdims=True)
    partial = keras.ops.logical_and(n_missing > 0, n_missing < n_band)
    return keras.ops.where(partial, n_missing / n_band, keras.ops.zeros_like(n_missing))


def band_adapter(image_in, level_filters, adapter_filters):
    """Per-level (gamma, beta, m) corrections for the backbone, from the input planes and which bands have data.

    One small conv layer per backbone level (full resolution, then halved each time like the encoder), told the
    fraction of each band's pixels with data; gamma and beta start at zero. m is the missing fraction, averaged down
    to each level's resolution."""
    covered = layers.Lambda(band_coverage, name="adapter_coverage")(image_in)
    missing = layers.Lambda(missing_fraction, name="adapter_missing")(image_in)
    presence = layers.GlobalAveragePooling2D(name="adapter_presence")(covered)
    x = layers.Concatenate(name="adapter_input")([image_in, covered])
    corrections = []
    for level, filters in enumerate(level_filters):
        if level:
            x = layers.MaxPool2D(name=f"adapter_pool{level}")(x)
            missing = layers.AveragePooling2D(2, name=f"adapter_missing_pool{level}")(missing)
        x = layers.Conv2D(adapter_filters, 3, padding="same", activation="swish", name=f"adapter_conv{level}")(x)
        bands = layers.Reshape((1, 1, adapter_filters), name=f"adapter_bands_r{level}")(
            layers.Dense(adapter_filters, name=f"adapter_bands{level}")(presence))
        x = layers.Add(name=f"adapter_add{level}")([x, bands])
        gamma = layers.Conv2D(filters, 1, kernel_initializer="zeros", name=f"adapter_gamma{level}")(x)
        beta = layers.Conv2D(filters, 1, kernel_initializer="zeros", name=f"adapter_beta{level}")(x)
        corrections.append((gamma, beta, missing))
    return corrections


def adapt(features, correction, name):
    """features + m * (features * gamma + beta): unchanged wherever m is 0."""
    gamma, beta, missing = correction
    return layers.Lambda(lambda t: t[0] + t[3] * (t[0] * t[1] + t[2]), name=f"{name}_adapted")(
        [features, gamma, beta, missing])


def build_unet(cfg):
    """The detector as a Keras model with inputs image_planes and psf_kernels and one output per head (cfg["heads"],
    plus centroid_offset and source_structure)."""
    filters, input_size = cfg["base_filters"], cfg["tile_size"] + 2 * cfg["tile_halo"]
    image_in = layers.Input((input_size, input_size, 2 * len(BANDS)), name="image_planes")
    psf_in = layers.Input((cfg["psf_stamp"], cfg["psf_stamp"], len(BANDS)), name="psf_kernels")
    psf_embedding = psf_encoder(psf_in)

    level_filters = [filters, filters * 2, filters * 4, filters * 8, filters * 12]
    corrections = band_adapter(image_in, level_filters, cfg["adapter_filters"]) if cfg["band_adapter"] else None

    def conditioned(features, level, name):
        features = film(features, psf_embedding, level_filters[level], f"{name}_film")
        return adapt(features, corrections[level], name) if corrections else features

    enc1 = conditioned(conv_block(image_in, filters, "enc1"), 0, "enc1")
    enc2 = conditioned(conv_block(layers.MaxPool2D()(enc1), filters * 2, "enc2"), 1, "enc2")
    enc3 = conditioned(conv_block(layers.MaxPool2D()(enc2), filters * 4, "enc3"), 2, "enc3")
    enc4 = conditioned(conv_block(layers.MaxPool2D()(enc3), filters * 8, "enc4"), 3, "enc4")
    bottleneck = conv_block(layers.MaxPool2D()(enc4), filters * 12, "bottleneck", dilation=2)
    bottleneck = layers.SpatialDropout2D(0.15)(conditioned(bottleneck, 4, "bottleneck"))

    dec4 = decoder_block(bottleneck, enc4, filters * 8, "dec4")
    dec3 = decoder_block(dec4, enc3, filters * 4, "dec3")
    dec2 = decoder_block(dec3, enc2, filters * 2, "dec2")
    dec1 = decoder_block(dec2, enc1, filters, "dec1")

    heads = set(cfg["heads"])
    outputs = {  # always created first and in this order: saved weights are restored by creation order
        "galaxy_heatmap": heatmap_head(dec1, filters, "galaxy_heatmap"),
        "centroid_offset": layers.Conv2D(2, 1, activation="tanh", name="centroid_offset")(dec1),
        "source_structure": layers.Conv2D(4, 1, activation=None, name="source_structure")(dec1),
    }
    for name in ("sfregion_heatmap", "tidal_heatmap", "star_heatmap", "sfregion_map", "tidal_map", "spike_map"):
        if name in heads:
            outputs[name] = heatmap_head(dec1, filters, name)
    if "detection_heatmap" in heads:
        outputs["detection_heatmap"] = detection_head(dec1, outputs, filters)
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


def distilled_focal_loss(weight):
    """Loss for training the band adapter on a map head whose output stacks [map with bands missing, the same map
    with all bands]: the focal loss of the first plus weight x its Kullback-Leibler divergence from the second (held
    fixed), counted where the target's loss weight is set and normalised like the focal loss."""
    def loss(y_true, y_pred):
        prediction = tf.clip_by_value(tf.cast(y_pred[..., :1], tf.float32), 1e-6, 1 - 1e-6)
        reference = tf.clip_by_value(tf.stop_gradient(tf.cast(y_pred[..., 1:2], tf.float32)), 1e-6, 1 - 1e-6)
        target, centre_weight, valid = (tf.cast(y_true[..., i:i + 1], tf.float32) for i in range(3))
        divergence = (reference * tf.math.log(reference / prediction)
                      + (1.0 - reference) * tf.math.log((1.0 - reference) / (1.0 - prediction)))
        normaliser = tf.maximum(tf.reduce_sum(tf.cast(target >= 0.999, tf.float32) * centre_weight * valid), 1.0)
        return focal_loss(y_true, y_pred[..., :1]) + weight * tf.reduce_sum(divergence * valid) / normaliser
    return loss


def masked_huber(y_true, y_pred):
    """Huber loss on regression outputs, counted only where the weight channel (last in y_true) is set."""
    n_output = tf.shape(y_pred)[-1]
    target = tf.cast(y_true[..., :n_output], tf.float32)
    weight = tf.cast(y_true[..., n_output:n_output + 1], tf.float32)
    residual = tf.cast(y_pred, tf.float32) - target
    loss = tf.where(tf.abs(residual) < 1.0, 0.5 * residual ** 2, tf.abs(residual) - 0.5)
    return tf.reduce_sum(loss * weight) / tf.maximum(tf.reduce_sum(weight), 1.0)


def output_names(model):
    """The names of a model's outputs: the keys of its output dict (or, for a list of outputs, their layer names)."""
    return list(model.output_names) if isinstance(model.output, list) else list(model.output.keys())


def compile_unet(model, cfg, inactive=()):
    """Adam with gradient clipping, a loss for every output (focal for maps, masked Huber for regressions), weighted by
    cfg["loss_weights"]. Heads in inactive get zero weight, e.g. star_heatmap when the training data has no stars."""
    names = output_names(model)
    loss_weights = {name: 0.0 if name in inactive else float(cfg["loss_weights"].get(name, 0.0)) for name in names}
    losses = {name: masked_huber if name in ("centroid_offset", "source_structure") else focal_loss for name in names}
    model.compile(optimizer=keras.optimizers.Adam(cfg["learning_rate"], clipnorm=5.0), loss=losses,
                  loss_weights=loss_weights)
    return model


def transfer_weights(model, cfg):
    """A model built afresh for cfg (e.g. with more heads, or a band adapter), with every layer the given model also
    has (by name) taking its trained weights. New detection heads start equal to the galaxy heatmap and new adapters
    change nothing (see the module docstring), so the new model's outputs match the old model's until it is
    trained."""
    upgraded = build_unet(cfg)
    trained = {layer.name: layer for layer in model.layers}
    for layer in upgraded.layers:
        if layer.name in trained and layer.weights:
            layer.set_weights(trained[layer.name].get_weights())
    return upgraded
