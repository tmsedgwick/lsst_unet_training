"""Unit checks of data, targets, calibration and the network, plus a one-epoch train -> calibrate -> evaluate run."""

import json

import numpy as np
import pytest
from conftest import STARS

from lsst_unet_training import ARTEFACTS, CONFIG, evaluate_unet
from lsst_unet_training.calibration import choose_threshold, wilson_lower
from lsst_unet_training.coadd_data import CoaddStore, encode_planes
from lsst_unet_training.config import ORIGINAL_HEADS
from lsst_unet_training.targets import TargetMaker, paint_gaussian, truth_map
from lsst_unet_training.unet_model import add_heads, build_unet


def test_paint_gaussian_marks_centre():
    heatmap = np.zeros((20, 20), np.float32)
    assert paint_gaussian(heatmap, 10.3, 5.6, 1.5) == (10, 6)
    assert heatmap[6, 10] == 1.0 and heatmap.max() == 1.0 and heatmap[0, 0] == 0.0
    assert paint_gaussian(heatmap, -50, -50, 1.5) is None


def test_tile_targets(dataset):
    catalogue_dir, image_dir = dataset
    store = CoaddStore("train", catalogue_dir, image_dir, CONFIG)
    assert store.has_stars and len(store.stars) == 4 and len(store.truth) == 60
    assert set(store.component_images) == {"clumps", "tidal", "spikes"}
    maker = TargetMaker(store, CONFIG)
    maker.prepare(store)
    targets = maker.tile_targets(store, 0, 0, 0, "10y_fwhm110")
    full = CONFIG["tile_size"] + 2 * CONFIG["tile_halo"]
    assert set(targets) == {*CONFIG["heads"], "centroid_offset", "source_structure"}
    assert targets["galaxy_heatmap"].shape == (full, full, 3) and targets["source_structure"].shape == (full, full, 5)
    n_in_tile = len(store.truth_by_tile.get(0, []))
    assert 0 < (targets["galaxy_heatmap"][..., 1] > 0).sum() <= n_in_tile  # one weighted centre pixel per galaxy
    assert np.allclose(store.population_weight[store.truth_inside].mean(), 1.0, atol=0.5)
    # extended galaxies get broader centre peaks; stars and galaxies both appear in the detection target
    assert store.target_sigma[:3].min() > store.target_sigma[3:].max() - 1e-9
    n_stars_in_tile = len(store.star_by_tile.get(0, []))
    assert (targets["detection_heatmap"][..., 1] > 0).sum() == (targets["galaxy_heatmap"][..., 1] > 0).sum() + \
        (targets["star_heatmap"][..., 1] > 0).sum() and (targets["star_heatmap"][..., 1] > 0).sum() == n_stars_in_tile
    # truth maps: spikes cross the whole tile, tidal blobs and clumps cover small areas
    for head in ("clump_map", "tidal_map", "spike_map"):
        mask = targets[head][..., 0]
        assert set(np.unique(mask)) <= {0.0, 1.0} and 0 < mask.mean() < 0.5, head
    assert store.tile_weight.min() >= 1.0 and store.tile_weight.max() <= CONFIG["max_tile_oversampling"]


def test_truth_map_follows_the_noise():
    light = np.zeros((6, 64, 64), np.float32)
    light[:, 30:34, 10:54] = 1.0  # a faint stripe
    cfg = {**CONFIG}
    assert truth_map(light, np.full(6, 0.2), cfg).sum() > 0  # shallow noise: detectable
    assert truth_map(light, np.full(6, 20.0), cfg).sum() == 0  # deep noise: not


def test_bright_stars_saturate_in_rebuilt_tiles(dataset):
    catalogue_dir, image_dir = dataset
    store = CoaddStore("valid", catalogue_dir, image_dir, CONFIG)
    x, y = int(STARS["x"][0]), int(STARS["y"][0])
    signal, _ = store.coadd_tile("10y_fwhm110", 0, 0)
    core = signal[2, y + 32 - 2:y + 32 + 3, x + 32 - 2:x + 32 + 3]  # the tile's halo is 32 px
    assert all(np.ptp(row) < 1e-3 * row.mean() for row in core)  # flat rows: saturated


def test_no_data_beyond_the_edge(dataset):
    catalogue_dir, image_dir = dataset
    store = CoaddStore("calib", catalogue_dir, image_dir, CONFIG)
    signal, variance = store.coadd_tile("10y_fwhm110", 0, 0)
    assert (signal[:, :32, :] == 0).all() and (variance[:, :, :32] == CONFIG["no_data_variance"]).all()
    reflecting = CoaddStore("calib", catalogue_dir, image_dir, {**CONFIG, "edge_padding": "reflect"})
    assert reflecting.coadd_tile("10y_fwhm110", 0, 0)[1][:, :32, :].max() < 1e6


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


def test_architecture_is_unchanged():
    # 3,596,337 parameters in the original layout at the default width: a change here means models saved earlier no
    # longer load.
    assert build_unet({**CONFIG, "heads": ORIGINAL_HEADS}).count_params() == 3_596_337


def test_added_heads_leave_detections_unchanged():
    small = {**CONFIG, "base_filters": 8}
    old = build_unet({**small, "heads": ORIGINAL_HEADS})
    new = add_heads(old, small)
    rng = np.random.default_rng(0)
    inputs = {"image_planes": rng.normal(size=(1, 320, 320, 12)).astype("float32"),
              "psf_kernels": np.full((1, 25, 25, 6), 1 / 625, "float32")}
    before, after = old.predict_on_batch(inputs), new.predict_on_batch(inputs)
    assert np.allclose(after["detection_heatmap"], before["galaxy_heatmap"], atol=1e-6)
    assert np.allclose(after["galaxy_heatmap"], before["galaxy_heatmap"])


def test_train_calibrate_evaluate(dataset, trained_model):
    catalogue_dir, image_dir = dataset
    for artefact in ("weights", "normalisation", "model_config", "history"):
        assert (trained_model / ARTEFACTS[artefact]).exists(), artefact
    threshold = json.loads((trained_model / ARTEFACTS["threshold"]).read_text())
    assert 0.0 <= threshold["threshold"] <= 1.0
    config = json.loads((trained_model / ARTEFACTS["model_config"]).read_text())
    assert config["cfg"]["heads"] == list(CONFIG["heads"]) and config["cfg"]["edge_padding"] == "no_data"
    summary = evaluate_unet(catalogue_dir, image_dir, trained_model, ["1y_fwhm110", "10y_fwhm110"])
    assert list(summary["coadd"]) == ["1y_fwhm110", "10y_fwhm110"]
    assert summary["purity"].between(0, 1).all()
    assert {"extended_completeness", "star_completeness"} <= set(summary.columns)
    assert list(trained_model.glob("mep_*_test_completeness_vs_size.png"))
