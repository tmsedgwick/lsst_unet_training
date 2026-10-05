"""Checks of updating a model from auxiliary data: labels, split, threshold choice, partial-label targets, and
both commands end to end on the tiny dataset (its saved 10-year test coadd stands in for the auxiliary coadd)."""

import json

import numpy as np
import pandas as pd
import pytest

from lsst_unet_training import ARTEFACTS, BANDS, CONFIG, update_threshold_on_aux, update_weights_on_aux
from lsst_unet_training.coadd_data import gaussian_psf_kernel
from lsst_unet_training.update import (AuxCoadd, AuxTiles, choose_weighted_threshold, label_metrics,
                                              read_labels, split_labels)

SMALL_BLOCKS = dict(aux_block_pix=80)  # the tiny images are 320 px, so 16 blocks


@pytest.fixture(scope="module")
def aux_data(dataset, tmp_path_factory):
    """A auxiliary coadd .npz (bands stored in reverse order) and two review files: one fully reviewed, one sampled."""
    catalogue_dir, image_dir = dataset
    root = tmp_path_factory.mktemp("real")
    signal = np.load(image_dir / "test" / "test_10y_fwhm110_signal.npy")
    variance = np.load(image_dir / "test" / "test_10y_fwhm110_variance.npy")
    psf = np.stack([gaussian_psf_kernel(1.1, CONFIG) for _ in BANDS], axis=-1)
    np.savez(root / "coadd.npz", signal=signal[::-1], variance=variance[::-1], psf_kernels=psf[..., ::-1],
             bands=np.array(BANDS[::-1]), origin=np.array([0, 0]))
    truth = pd.read_csv(catalogue_dir / "mock_catalogue_test.csv")
    rng = np.random.default_rng(3)
    entry = lambda x, y, decision: dict(x=float(x), y=float(y), x_patch=float(x), y_patch=float(y), significance=1.0,
                                        decision=decision)
    sampled = {str(i): entry(r.x_pix, r.y_pix, "real") for i, r in enumerate(truth.iloc[:30].itertuples())}
    sampled["30"] = entry(*rng.uniform(10, 310, 2), "unsure")
    full = {str(i): entry(*rng.uniform(10, 310, 2), "spurious") for i in range(30)}
    files = [root / "feedback_sampled.json", root / "feedback_full.json"]
    files[0].write_text(json.dumps(dict(category="sampled", origin=[0, 0], n=62, reviewed=sampled, missed=[])))
    files[1].write_text(json.dumps(dict(category="full", origin=[0, 0], n=30, reviewed=full, missed=[
        dict(x=float(truth.x_pix[40]), y=float(truth.y_pix[40]), x_patch=0.0, y_patch=0.0, near_detection=0)])))
    return root / "coadd.npz", files


def test_labels_are_weighted_and_deduplicated(tmp_path):
    review = dict(category="u", n=40, missed=[dict(x=50.4, y=50.0)], reviewed={
        "0": dict(x=10.0, y=10.0, decision="real"), "1": dict(x=10.5, y=10.2, decision="real"),
        "2": dict(x=30.0, y=30.0, decision="real"), "3": dict(x=30.4, y=30.0, decision="spurious"),
        "4": dict(x=70.0, y=70.0, decision="unsure")})
    (tmp_path / "f.json").write_text(json.dumps(review))
    pd.DataFrame(dict(x=[50.0, 90.0], y=[50.0, 90.0], label=["real", "Spurious"])).to_csv(tmp_path / "l.csv")
    labels = read_labels([tmp_path / "f.json", tmp_path / "l.csv"])
    # the duplicate at (10, 10) is merged, the conflict at (30, 30) dropped, unsure skipped, and the missed galaxy
    # at (50.4, 50) merged with the CSV's real one
    assert sorted(map(tuple, labels[["x", "y"]].round().to_numpy())) == [(10, 10), (50, 50), (90, 90)]
    assert labels.set_index("x").loc[10.0, "weight"] == 40 / 5 and labels["is_real"].sum() == 2


def test_split_is_spatial_and_reproducible():
    rng = np.random.default_rng(0)
    labels = pd.DataFrame(dict(x=rng.uniform(0, 1000, 500), y=rng.uniform(0, 1000, 500), is_real=True, weight=1.0))
    first, second = split_labels(labels, 0.3, 250, 1), split_labels(labels, 0.3, 250, 1)
    assert first["split"].equals(second["split"])
    block = (labels["x"] // 250).astype(int) * 10 + (labels["y"] // 250).astype(int)
    assert (first.groupby(block)["split"].nunique() == 1).all()
    assert 0.15 < (first["split"] == "test").mean() < 0.45


def test_weighted_threshold_and_metrics():
    p_real = np.r_[np.linspace(0.99, 0.5, 50), np.linspace(0.49, 0.01, 50), 0.0]
    is_real = np.r_[np.ones(50, bool), np.zeros(50, bool), True]
    weight = np.ones(101)
    threshold, row, status = choose_weighted_threshold(p_real, is_real, weight, 0.93, 1.64)
    assert status == "met" and threshold == pytest.approx(0.5) and row["purity"] == 1.0
    metrics = label_metrics(p_real, is_real, weight, threshold)
    assert metrics["real_found"] == "50/51" and metrics["spurious_kept"] == "0/50"
    # a heavily weighted spurious label at the top makes the purity target unreachable
    weight[50] = 100.0
    p_real[50] = 1.0
    assert choose_weighted_threshold(p_real, is_real, weight, 0.9, 1.64)[2] == "unmet"


def test_aux_tile_targets_only_count_near_labels(aux_data):
    path, files = aux_data
    coadd = AuxCoadd(path, CONFIG)
    assert coadd.signal.shape == (6, 320, 320) and np.allclose(coadd.psf.sum(axis=(0, 1)), 1)
    labels = split_labels(read_labels(files, coadd.nx, coadd.ny), 0.3, 80, 0)
    tiles = AuxTiles(coadd, labels, dict(logvar_centre=[0.0] * 6, logvar_scale=[1.0] * 6),
                      {**CONFIG, **SMALL_BLOCKS})
    heatmap = tiles.targets(0, 0)["galaxy_heatmap"]
    train = labels[labels["split"] == "train"]
    halo, radius = CONFIG["tile_halo"], CONFIG["label_radius_pix"]
    area = np.pi * radius ** 2
    assert 0 < (heatmap[..., 2] > 0).sum() <= len(train) * area * 1.1
    assert heatmap[..., 1].sum() <= train["is_real"].sum() and heatmap[..., 1].sum() > 0
    real = train[train["is_real"] & (train["x"] < 256) & (train["y"] < 256)].iloc[0]
    cx, cy = int(np.rint(real["x"])) + halo, int(np.rint(real["y"])) + halo
    assert heatmap[cy, cx, 0] == 1.0 and heatmap[cy, cx, 2] == CONFIG["update_aux_weight"]
    planes, psf, targets = tiles.sample(np.random.default_rng(0))
    assert planes.shape == (320, 320, 12) and psf.shape == (25, 25, 6) and targets["galaxy_heatmap"][..., 2].any()


def test_updates_on_aux_write_new_models(dataset, trained_model, aux_data):
    catalogue_dir, image_dir = dataset
    path, files = aux_data
    original = (trained_model / ARTEFACTS["threshold"]).read_text()
    cfg = dict(SMALL_BLOCKS, aux_purity=0.5)
    out = update_threshold_on_aux(trained_model, path, files, catalogue_dir, image_dir, "thr", cfg)
    assert out.name == f"{trained_model.name}_thr" and (trained_model / ARTEFACTS["threshold"]).read_text() == original
    threshold = json.loads((out / ARTEFACTS["threshold"]).read_text())
    assert threshold["chosen_on"] == "auxiliary train labels" and 0 < threshold["threshold"] <= 1
    assert (out / ARTEFACTS["weights"]).exists() and (out / "mep_aux_labels.csv").exists()
    with pytest.raises(FileExistsError):
        update_threshold_on_aux(trained_model, path, files, catalogue_dir, image_dir, "thr", cfg)

    cfg = dict(SMALL_BLOCKS, batch_size=2, data_workers=1, valid_samples=4, update_epochs=1,
               update_steps_per_epoch=2)
    tuned = update_weights_on_aux(trained_model, path, files, catalogue_dir, image_dir, "ft", cfg)
    assert (trained_model / ARTEFACTS["threshold"]).read_text() == original
    for artefact in ("weights", "normalisation", "model_config", "calib_peaks", "threshold"):
        assert (tuned / ARTEFACTS[artefact]).exists(), artefact
    assert json.loads((tuned / ARTEFACTS["model_config"]).read_text())["updated_from"] == str(trained_model)
    report = json.loads((tuned / "mep_update_report.json").read_text())
    assert set(report["aux"]["test"]) == {"before", "after"}
