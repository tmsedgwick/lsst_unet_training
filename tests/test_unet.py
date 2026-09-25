"""Unit checks of targets, calibration and the network, plus a one-epoch train -> calibrate -> evaluate run."""

import json

import numpy as np
import pytest

from lsst_unet_training import ARTEFACTS, CONFIG, calibrate_unet, evaluate_unet, train_unet
from lsst_unet_training.calibration import choose_threshold, wilson_lower
from lsst_unet_training.coadd_data import CoaddStore, encode_planes
from lsst_unet_training.targets import TargetMaker, paint_gaussian
from lsst_unet_training.unet_model import build_unet

TINY = dict(epochs=1, batch_size=2, base_filters=8, valid_samples=4, data_workers=1)


def test_paint_gaussian_marks_centre():
    heatmap = np.zeros((20, 20), np.float32)
    assert paint_gaussian(heatmap, 10.3, 5.6, 1.5) == (10, 6)
    assert heatmap[6, 10] == 1.0 and heatmap.max() == 1.0 and heatmap[0, 0] == 0.0
    assert paint_gaussian(heatmap, -50, -50, 1.5) is None


def test_tile_targets(dataset):
    catalogue_dir, image_dir = dataset
    store = CoaddStore("train", catalogue_dir, image_dir, CONFIG)
    maker = TargetMaker(store, CONFIG)
    maker.prepare(store)
    targets = maker.tile_targets(store, 0, 0, 0)
    full = CONFIG["tile_size"] + 2 * CONFIG["tile_halo"]
    assert targets["galaxy_heatmap"].shape == (full, full, 3) and targets["source_structure"].shape == (full, full, 5)
    n_in_tile = len(store.truth_by_tile.get(0, []))
    assert 0 < (targets["galaxy_heatmap"][..., 1] > 0).sum() <= n_in_tile  # one weighted centre pixel per galaxy
    assert np.allclose(store.population_weight[store.truth_inside].mean(), 1.0, atol=0.5)


def test_inputs_are_deterministic_for_rebuilt_coadds(dataset):
    catalogue_dir, image_dir = dataset
    store = CoaddStore("valid", catalogue_dir, image_dir, CONFIG)
    first, second = store.coadd_tile("10y_fwhm110", 0, 0), store.coadd_tile("10y_fwhm110", 0, 0)
    assert np.array_equal(first[0], second[0])
    planes = encode_planes(*first, dict(logvar_centre=[0.0] * 6, logvar_scale=[1.0] * 6))
    assert planes.shape == (320, 320, 12) and np.isfinite(planes).all()


def test_threshold_choice():
    assert wilson_lower(99, 100, 1.64) == pytest.approx(0.9566, abs=1e-4)
    p_real = np.linspace(1, 0, 1000)
    is_real = np.r_[np.ones(900), np.zeros(100)].astype(int)  # the top 900 are all real
    threshold, row, status = choose_threshold(p_real, is_real, 0.99, 1.64)
    assert status == "met" and row["purity_lower"] >= 0.99 and 0.09 < threshold < 0.2


def test_architecture_matches_trained_models():
    # 3,596,337 parameters at the default width: any change here breaks loading the existing trained weights.
    assert build_unet(CONFIG).count_params() == 3_596_337


def test_train_calibrate_evaluate(dataset, tmp_path):
    catalogue_dir, image_dir = dataset
    train_unet(catalogue_dir, image_dir, tmp_path, TINY)
    for artefact in ("weights", "normalisation", "model_config", "history"):
        assert (tmp_path / ARTEFACTS[artefact]).exists(), artefact
    calibrate_unet(catalogue_dir, image_dir, tmp_path, dict(reference_coadd="10y_fwhm110"))
    threshold = json.loads((tmp_path / ARTEFACTS["threshold"]).read_text())
    assert 0.0 <= threshold["threshold"] <= 1.0
    summary = evaluate_unet(catalogue_dir, image_dir, tmp_path, ["1y_fwhm110", "10y_fwhm110"])
    assert list(summary["coadd"]) == ["1y_fwhm110", "10y_fwhm110"]
    assert summary["purity"].between(0, 1).all()
