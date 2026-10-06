"""Checks of missing bands: the band adapter leaves six-band results exactly unchanged, bands are dropped and the
targets follow what the remaining bands can show, and the adapter trains, calibrates per band set and evaluates."""

import json

import numpy as np
import pytest

from lsst_unet_training import ARTEFACTS, BANDS, CONFIG, calibrate_unet, evaluate_unet, load_model, train_band_adapter
from lsst_unet_training.calibration import nearest_band_set, read_thresholds
from lsst_unet_training.coadd_data import CoaddStore
from lsst_unet_training.masking import coverage, drop_bands
from lsst_unet_training.targets import TargetMaker
from lsst_unet_training.unet_model import build_unet, transfer_weights

SMALL = {**CONFIG, "base_filters": 8}


def predict(model, planes, psf):
    return model.predict_on_batch({"image_planes": planes, "psf_kernels": psf})


def test_adapter_changes_nothing_with_all_bands():
    backbone = build_unet(SMALL)
    adapted = transfer_weights(backbone, {**SMALL, "band_adapter": True})
    rng = np.random.default_rng(0)
    for layer in adapted.layers:  # as if trained: the adapter's corrections are no longer zero
        if layer.name.startswith(("adapter_gamma", "adapter_beta")):
            layer.set_weights([w + rng.normal(0, 0.1, w.shape).astype("float32") for w in layer.get_weights()])
    planes = rng.normal(size=(1, 320, 320, 12)).astype("float32")
    planes[..., 1::2] = np.clip(planes[..., 1::2], -3, 3)
    psf = np.full((1, 25, 25, 6), 1 / 625, "float32")
    edge = planes.copy()
    edge[:, :40, :, 0::2], edge[:, :40, :, 1::2] = 0.0, 8.0  # beyond an image edge every band is missing
    for inputs in (planes, edge):
        before, after = predict(backbone, inputs, psf), predict(adapted, inputs, psf)
        assert all(np.array_equal(before[name], after[name]) for name in before)
    dropped = planes.copy()
    dropped[..., 0], dropped[..., 1] = 0.0, 8.0  # u missing
    assert not np.allclose(predict(backbone, dropped, psf)["detection_heatmap"],
                           predict(adapted, dropped, psf)["detection_heatmap"])


def test_drop_bands():
    signal, variance = np.ones((6, 20, 20), np.float32), np.ones((6, 20, 20), np.float32)
    psf = np.stack([np.full((25, 25), b + 1.0) for b in range(6)], axis=-1) / 625
    region = np.zeros((20, 20), bool)
    region[:, :5] = True
    s, v, p = drop_bands(signal, variance, psf, "uy", CONFIG)
    covered = coverage(v, CONFIG)
    assert not covered[[0, 5]].any() and covered[1:5].all() and (s[0] == 0).all()
    assert np.allclose(p[..., 0], p[..., 5]) and np.isclose(p[..., 0].sum(), 1.0)  # mean of the others' stamps
    s, v, p = drop_bands(signal, variance, psf, "r", CONFIG, region)
    assert not coverage(v, CONFIG)[2][:, :5].any() and coverage(v, CONFIG)[2][:, 5:].all()
    assert np.array_equal(p, psf)  # a band covering part of the tile keeps its PSF
    assert nearest_band_set("griz", ["ugrizy", "griz"]) == "griz"
    assert nearest_band_set("grz", ["ugrizy", "griz", "gri"]) == "griz"


def test_targets_follow_the_bands_left(dataset):
    catalogue_dir, image_dir = dataset
    store = CoaddStore("train", catalogue_dir, image_dir, CONFIG)
    maker = TargetMaker(store, CONFIG)
    maker.prepare(store)
    full = CONFIG["tile_size"] + 2 * CONFIG["tile_halo"]
    only_u = np.zeros((6, full, full), bool)
    only_u[0] = True  # 1-year u alone: few sources stay detectable
    every = maker.tile_targets(store, 0, 0, 0, "1y_fwhm110")
    fewer = maker.tile_targets(store, 0, 0, 0, "1y_fwhm110", coverage=only_u)
    assert fewer["detection_heatmap"][..., 1].sum() < every["detection_heatmap"][..., 1].sum()
    assert fewer["detection_heatmap"][..., 2].sum() < every["detection_heatmap"][..., 2].sum()  # unknown, not "no"
    all_bands = maker.tile_targets(store, 0, 0, 0, "1y_fwhm110", coverage=np.ones((6, full, full), bool))
    assert np.array_equal(all_bands["detection_heatmap"], every["detection_heatmap"])


def test_train_calibrate_and_evaluate_the_adapter(dataset, trained_model):
    catalogue_dir, image_dir = dataset
    cfg = dict(batch_size=2, data_workers=1, valid_samples=4, adapter_epochs=1,
               calibration_band_sets=("ugrizy", "gri", "r"))
    adapted_dir = train_band_adapter(catalogue_dir, image_dir, trained_model, "bands", cfg)
    with pytest.raises(FileExistsError):
        train_band_adapter(catalogue_dir, image_dir, trained_model, "bands", cfg)
    backbone, _, _ = load_model(trained_model)
    adapted, _, config = load_model(adapted_dir)
    assert config["cfg"]["band_adapter"]
    rng = np.random.default_rng(1)
    planes = np.clip(rng.normal(size=(1, 320, 320, 12)), -3, 3).astype("float32")
    psf = np.full((1, 25, 25, 6), 1 / 625, "float32")
    before, after = predict(backbone, planes, psf), predict(adapted, planes, psf)
    assert all(np.array_equal(before[name], after[name]) for name in before)  # the backbone did not move

    thresholds = calibrate_unet(catalogue_dir, image_dir, adapted_dir, cfg)
    assert set(thresholds) == {"ugrizy", "gri", "r"} and set(read_thresholds(adapted_dir)) == set(thresholds)
    summary = evaluate_unet(catalogue_dir, image_dir, adapted_dir, ["10y_fwhm110"], band_sets=("ugrizy", "gri"))
    assert list(summary["band_set"]) == ["ugrizy", "gri"] and summary["purity"].between(0, 1).all()
    assert json.loads((adapted_dir / ARTEFACTS["model_config"]).read_text())["adapter_added_to"] == str(trained_model)
