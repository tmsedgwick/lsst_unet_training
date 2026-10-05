"""Train the U-Net: train catalogue fits the weights, valid catalogue decides early stopping.

Writes to model_dir: best weights, log-variance normalisation, model config (settings plus the structure scaling
needed to read predicted sizes), training history and a training-curve plot.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import keras  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from .coadd_data import CoaddStore, log_variance_normalisation  # noqa: E402
from .config import ARTEFACTS, CATALOGUE_STEM, CONFIG, ORIGINAL_HEADS  # noqa: E402
from .targets import MAP_COMPONENTS, TargetMaker  # noqa: E402
from .tile_sequence import CoaddTileSequence  # noqa: E402
from .unet_model import build_unet, compile_unet  # noqa: E402


def set_up_tensorflow(seed):
    """Seed everything and let TensorFlow grow GPU memory as needed."""
    keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"TensorFlow sees {len(gpus)} GPU(s)" + ("" if gpus else ": CPU only, expect training to be very slow"))


# Settings a trained model depends on, saved with it so calibration and inference rebuild the same network.
MODEL_SETTINGS = ("tile_size", "tile_halo", "psf_stamp", "base_filters", "min_peak_score", "max_peaks_per_tile",
                  "match_radius_pix", "heads", "edge_padding")
# What a model config without these settings means: it was saved before they existed.
SETTINGS_OF_OLDER_MODELS = dict(heads=ORIGINAL_HEADS, edge_padding="reflect")


def write_model_config(model_dir, cfg, target_maker, train_name, valid_name=None):
    """Save the model settings and the size scaling learned from the train catalogue."""
    (Path(model_dir) / ARTEFACTS["model_config"]).write_text(json.dumps(dict(
        cfg={key: cfg[key] for key in MODEL_SETTINGS}, log_re_mean=target_maker.log_re_mean,
        log_re_std=target_maker.log_re_std, train_catalogue=train_name, valid_catalogue=valid_name), indent=2))


def load_model(model_dir, cfg=None):
    """(model with trained weights, normalisation, model config) from a model directory written by train_unet."""
    model_dir = Path(model_dir)
    if not (model_dir / ARTEFACTS["model_config"]).exists():
        raise FileNotFoundError(f"{model_dir} has no {ARTEFACTS['model_config']}: is it a model folder written "
                                "by scripts/train_unet.py?")
    model_config = json.loads((model_dir / ARTEFACTS["model_config"]).read_text())
    cfg = {**CONFIG, **SETTINGS_OF_OLDER_MODELS, **model_config["cfg"], **(cfg or {})}
    cfg["heads"] = tuple(cfg["heads"])
    model = build_unet(cfg)
    model.load_weights(model_dir / ARTEFACTS["weights"])
    normalisation = json.loads((model_dir / ARTEFACTS["normalisation"]).read_text())
    return model, normalisation, {**model_config, "cfg": cfg}


def unteachable_heads(store, heads):
    """Heads the store's mocks hold no truth for (no stars simulated, or a phenomenon's light not saved): training
    gives them zero loss weight rather than teaching them that the thing never occurs."""
    missing = {head for head, component in MAP_COMPONENTS.items() if component not in store.component_images}
    if not store.has_stars:
        missing.add("star_heatmap")
    return sorted(missing & set(heads))


def plot_training_curve(history_path, out_path):
    history = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    monitored = "detection_heatmap" if "detection_heatmap_loss" in history else "galaxy_heatmap"
    ax.plot(history["epoch"], history[f"{monitored}_loss"], label="train")
    ax.plot(history["epoch"], history[f"val_{monitored}_loss"], label="valid")
    ax.set(title=f"U-Net {monitored} loss", xlabel="Epoch", ylabel="Focal loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def train_unet(catalogue_dir, image_dir, model_dir, cfg=None, train_name="train", valid_name="valid",
               stem=CATALOGUE_STEM):
    """Train from scratch and save the best weights (lowest validation loss of the detection map) in model_dir."""
    cfg = {**CONFIG, **(cfg or {})}
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    set_up_tensorflow(cfg["seed"])

    train_store = CoaddStore(train_name, catalogue_dir, image_dir, cfg, stem)
    valid_store = CoaddStore(valid_name, catalogue_dir, image_dir, cfg, stem)
    for store in (train_store, valid_store):
        print(f"[{store.name}] {store.nx} x {store.ny} px, {len(store.coadds)} coadds, "
              f"{int(store.truth_inside.sum()):,} truth galaxies, "
              f"coadds {'saved' if store.saved else 'rebuilt on the fly'}")
    target_maker = TargetMaker(train_store, cfg)
    for store in (train_store, valid_store):
        target_maker.prepare(store)

    normalisation = log_variance_normalisation(train_store)
    (model_dir / ARTEFACTS["normalisation"]).write_text(json.dumps(normalisation, indent=2))
    write_model_config(model_dir, cfg, target_maker, train_name, valid_name)

    train_sequence = CoaddTileSequence(train_store, None, normalisation, cfg, target_maker, shuffle=True,
                                       fresh_noise=True, resample=True, coadds_per_tile=cfg["train_coadds_per_tile"],
                                       seed=cfg["seed"], workers=cfg["data_workers"],
                                       edge_augment_fraction=cfg["edge_augment_fraction"])
    valid_index = valid_store.sample_index()
    valid_index = valid_index.sample(n=min(cfg["valid_samples"], len(valid_index)),
                                     random_state=cfg["seed"]).reset_index(drop=True)
    valid_sequence = CoaddTileSequence(valid_store, valid_index, normalisation, cfg, target_maker,
                                       workers=cfg["data_workers"])
    print(f"Train: {len(train_sequence.index):,} tiles per epoch ({len(train_sequence)} steps, new coadds each "
          "epoch) | "
          f"valid: {len(valid_sequence.index):,} fixed tiles")

    inactive = unteachable_heads(train_store, cfg["heads"])
    if inactive:
        print(f"No truth in these mocks for {', '.join(inactive)}: not trained")
    model = compile_unet(build_unet(cfg), cfg, inactive)
    weights_path, history_path = model_dir / ARTEFACTS["weights"], model_dir / ARTEFACTS["history"]
    # Early stopping and checkpoints follow the map detections are taken from.
    monitor = "val_detection_heatmap_loss" if "detection_heatmap" in cfg["heads"] else "val_galaxy_heatmap_loss"
    callbacks = [
        keras.callbacks.ModelCheckpoint(weights_path, monitor=monitor, mode="min", save_best_only=True,
                                        save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor=monitor, mode="min", patience=cfg["patience"], restore_best_weights=True,
                                      verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(history_path),
    ]
    model.fit(train_sequence, validation_data=valid_sequence, epochs=cfg["epochs"], callbacks=callbacks)
    plot_training_curve(history_path, model_dir / "mep_unet_training_curve.png")
    print(f"Saved best weights -> {weights_path}")
    return model


def log_re_scaling(model_config):
    """(mean, std) that turn the structure head's first output back into log(1 + Re / pixel)."""
    return model_config["log_re_mean"], model_config["log_re_std"]
