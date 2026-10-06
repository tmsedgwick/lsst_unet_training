"""Checks of update.py: reading and splitting labels, choosing the threshold, the training targets on auxiliary
tiles, and both update commands end to end. The tiny dataset's saved 10-year test coadd serves as the auxiliary
coadd."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from lsst_unet_training import ARTEFACTS, BANDS, CONFIG, update_threshold_on_aux, update_weights_on_aux
from lsst_unet_training.coadd_data import gaussian_psf_kernel
from lsst_unet_training.unet_model import build_unet
from lsst_unet_training.update import (AuxCoadd, AuxTrainingTiles, choose_weighted_threshold, label_metrics,
                                       read_labels, split_labels)

SMALL_BLOCKS = dict(aux_block_pix=80)  # the tiny images are 320 px, so 16 blocks


@pytest.fixture(scope="module")
def aux_data(dataset, tmp_path_factory):
    """An auxiliary coadd .npz (bands stored in reverse order, to test reordering) and two review files: one
    category fully reviewed, one only sampled, with a few selected labels carrying reasons."""
    catalogue_dir, image_dir = dataset
    root = tmp_path_factory.mktemp("aux")
    signal = np.load(image_dir / "test" / "test_10y_fwhm110_signal.npy")
    variance = np.load(image_dir / "test" / "test_10y_fwhm110_variance.npy")
    psf = np.stack([gaussian_psf_kernel(1.1, CONFIG) for _ in BANDS], axis=-1)
    np.savez(root / "coadd.npz", signal=signal[::-1], variance=variance[::-1], psf_kernels=psf[..., ::-1],
             bands=np.array(BANDS[::-1]), origin=np.array([0, 0]))
    truth = pd.read_csv(catalogue_dir / "mock_catalogue_test.csv")
    rng = np.random.default_rng(3)
    entry = lambda x, y, decision: dict(x=float(x), y=float(y), x_patch=float(x), y_patch=float(y), significance=1.0,
                                        decision=decision)
    galaxies = truth[truth["type"] != "star"]
    sampled = {str(i): entry(r.x_pix, r.y_pix, "real") for i, r in enumerate(galaxies.iloc[:30].itertuples())}
    sampled["30"] = entry(*rng.uniform(10, 310, 2), "unsure")
    sampled["31"] = dict(entry(galaxies.x_pix.iloc[31] + 4, galaxies.y_pix.iloc[31], "spurious"), how="selected",
                         reason="tidal")
    star = truth[truth["type"] == "star"].iloc[1]
    additional = {"900": dict(entry(star.x_pix, star.y_pix, "real"), how="selected", reason="star",
                              category="both")}
    full = {str(i): entry(*rng.uniform(10, 310, 2), "spurious") for i in range(30)}
    files = [root / "feedback_sampled.json", root / "feedback_full.json"]
    files[0].write_text(json.dumps(dict(category="sampled", origin=[0, 0], n=62, reviewed=sampled, missed=[],
                                        additional_reviewed=additional)))
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
    # at (50.4, 50) merged with the CSV's source label (the random one is kept)
    assert sorted(map(tuple, labels[["x", "y"]].round().to_numpy())) == [(10, 10), (50, 50), (90, 90)]
    assert labels.set_index("x").loc[10.0, "weight"] == 40 / 5 and labels["is_source"].sum() == 2
    assert (labels["how"] == "random").all()


def test_selected_and_missed_labels_weigh_one_and_are_not_random(tmp_path):
    review = dict(category="u", n=100, missed=[dict(x=5.0, y=5.0)], reviewed={
        "0": dict(x=10.0, y=10.0, decision="real"), "1": dict(x=20.0, y=20.0, decision="spurious"),
        "2": dict(x=30.0, y=30.0, decision="spurious", how="selected", reason="spike")},
        additional_reviewed={"7": dict(x=40.0, y=40.0, decision="real", how="selected", reason="star",
                                       category="both")})
    (tmp_path / "f.json").write_text(json.dumps(review))
    labels = read_labels([tmp_path / "f.json"]).set_index("x")
    assert labels.loc[10.0, "weight"] == 50.0  # 100 candidates, 2 reviewed at random
    assert labels.loc[30.0, "how"] == "selected" and labels.loc[30.0, "weight"] == 1.0
    assert labels.loc[30.0, "reason"] == "spike" and labels.loc[40.0, "category"] == "both"
    assert labels.loc[5.0, "how"] == "missed" and labels.loc[5.0, "category"] == "u_missed"


def test_split_is_spatial_and_reproducible():
    rng = np.random.default_rng(0)
    labels = pd.DataFrame(dict(x=rng.uniform(0, 1000, 500), y=rng.uniform(0, 1000, 500), is_source=True, weight=1.0,
                               category="a"))
    first, second = split_labels(labels, 0.3, 250, 1), split_labels(labels, 0.3, 250, 1)
    assert first["split"].equals(second["split"])
    block = (labels["x"] // 250).astype(int) * 10 + (labels["y"] // 250).astype(int)
    assert (first.groupby(block)["split"].nunique() == 1).all()
    assert 0.15 < (first["split"] == "test").mean() < 0.45
    # a category confined to a strip of the image still gets test labels of its own
    strip = pd.DataFrame(dict(x=rng.uniform(0, 1000, 60), y=rng.uniform(0, 240, 60), is_source=True, weight=1.0,
                              category="b"))
    both = split_labels(pd.concat([labels, strip], ignore_index=True), 0.3, 250, 1)
    assert set(both.loc[both["category"] == "b", "split"]) == {"train", "test"}


def test_weighted_threshold_and_metrics():
    p_real = np.r_[np.linspace(0.99, 0.5, 50), np.linspace(0.49, 0.01, 50), 0.0]
    is_source = np.r_[np.ones(50, bool), np.zeros(50, bool), True]
    weight = np.ones(101)
    threshold, row, status = choose_weighted_threshold(p_real, is_source, weight, 0.93, 1.64)
    assert status == "met" and threshold == pytest.approx(0.5) and row["purity"] == 1.0
    metrics = label_metrics(p_real, is_source, weight, threshold)
    assert metrics["sources_detected"] == "50/51" and metrics["spurious_detected"] == "0/50"
    # a heavily weighted spurious label at the top makes the purity target unreachable
    weight[50] = 100.0
    p_real[50] = 1.0
    threshold, row, status = choose_weighted_threshold(p_real, is_source, weight, 0.9, 1.64)
    assert status == "unmet" and threshold is None and 0 < row["best_threshold"] <= 1  # keep the old threshold


def test_aux_tile_targets_only_count_near_labels(aux_data):
    path, files = aux_data
    coadd = AuxCoadd(path, CONFIG)
    assert coadd.signal.shape == (6, 320, 320) and np.allclose(coadd.psf.sum(axis=(0, 1)), 1)
    labels = split_labels(read_labels(files, coadd.nx, coadd.ny), 0.3, 80, 0)
    labels["loss_weight"] = CONFIG["update_aux_weight"]
    labels["split"] = "train"  # every label in play, to check each kind of target
    tiles = AuxTrainingTiles(coadd, labels, dict(logvar_centre=[0.0] * 6, logvar_scale=[1.0] * 6),
                             {**CONFIG, **SMALL_BLOCKS}, CONFIG["heads"])
    targets = tiles.targets(0, 0)
    detection = targets["detection_heatmap"]
    halo, radius = CONFIG["tile_halo"], CONFIG["label_radius_pix"]
    assert 0 < (detection[..., 2] > 0).sum() <= len(labels) * np.pi * radius ** 2 * 1.1
    assert 0 < detection[..., 1].sum() <= labels["is_source"].sum()
    source = labels[labels["is_source"] & (labels["x"] < 256) & (labels["y"] < 256) & (labels["reason"] == "")]
    cx, cy = int(np.rint(source["x"].iloc[0])) + halo, int(np.rint(source["y"].iloc[0])) + halo
    assert detection[cy, cx, 0] == 1.0 and detection[cy, cx, 2] == CONFIG["update_aux_weight"]
    assert targets["galaxy_heatmap"][cy, cx, 2] == 0  # an untagged source may be a galaxy or a star
    tidal = labels[labels["reason"] == "tidal"].iloc[0]
    tx, ty = int(np.rint(tidal["x"])) + halo, int(np.rint(tidal["y"])) + halo
    assert targets["tidal_map"][ty, tx, 0] == 1.0 and targets["tidal_map"][ty, tx, 2] > 0  # the reason teaches its map
    star = labels[labels["reason"] == "star"].iloc[0]
    star_tile = tiles.targets(int(star["x"]) // 256 * 256, int(star["y"]) // 256 * 256)
    assert star_tile["star_heatmap"][..., 1].sum() == 1  # a tagged star teaches star_heatmap
    planes, psf, sample = tiles.sample(np.random.default_rng(0))
    assert planes.shape == (320, 320, 12) and psf.shape == (25, 25, 6) and sample["detection_heatmap"][..., 2].any()


def test_updates_on_aux_write_new_models(dataset, trained_model, aux_data):
    catalogue_dir, image_dir = dataset
    path, files = aux_data
    original = (trained_model / ARTEFACTS["threshold"]).read_text()
    cfg = dict(SMALL_BLOCKS, aux_purity=0.5)
    out = update_threshold_on_aux(trained_model, path, files, catalogue_dir, image_dir, "thr", cfg)
    assert out.name == f"{trained_model.name}_thr" and (trained_model / ARTEFACTS["threshold"]).read_text() == original
    threshold = json.loads((out / ARTEFACTS["threshold"]).read_text())
    assert threshold["chosen_on"] == "auxiliary training labels" and 0 < threshold["threshold"] <= 1
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
    assert set(report["aux_labels"]["test"]) == {"before", "after"}


def test_update_gives_a_model_the_heads_it_lacks(dataset, trained_model, aux_data, tmp_path):
    """A model with centre heatmaps only and mirror-image edges is given the current heads and "no data" edges."""
    catalogue_dir, image_dir = dataset
    path, files = aux_data
    old = tmp_path / "centre_model"
    old.mkdir()
    heads = ("galaxy_heatmap", "sfregion_heatmap", "tidal_heatmap")
    build_unet({**CONFIG, "base_filters": 8, "heads": heads}).save_weights(old / ARTEFACTS["weights"])
    for artefact in ("normalisation", "calib_peaks", "threshold"):
        shutil.copy2(trained_model / ARTEFACTS[artefact], old / ARTEFACTS[artefact])
    config = json.loads((trained_model / ARTEFACTS["model_config"]).read_text())
    config["cfg"].update(heads=list(heads), edge_padding="reflect")
    (old / ARTEFACTS["model_config"]).write_text(json.dumps(config))
    cfg = dict(SMALL_BLOCKS, batch_size=2, data_workers=1, valid_samples=4, update_epochs=1,
               update_steps_per_epoch=1)
    tuned = update_weights_on_aux(old, path, files, catalogue_dir, image_dir, "ft", cfg)
    saved = json.loads((tuned / ARTEFACTS["model_config"]).read_text())["cfg"]
    assert saved["heads"] == list(CONFIG["heads"]) and saved["edge_padding"] == "no_data"
