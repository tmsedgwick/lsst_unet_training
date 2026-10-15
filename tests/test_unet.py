"""Unit checks of data, targets, calibration and the network, plus a one-epoch train -> calibrate -> evaluate run."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import STARS

from lsst_unet_training import ARTEFACTS, BANDS, CONFIG, evaluate_unet
from lsst_unet_training.calibration import choose_threshold, wilson_lower
from lsst_unet_training.coadd_data import CoaddStore, encode_planes
from lsst_unet_training.evaluation import galaxy_classes
from lsst_unet_training.targets import TargetMaker, combined_snr, paint_gaussian, truth_map
from lsst_unet_training.unet_model import band_coverage, build_unet, transfer_weights


def test_paint_gaussian_marks_centre():
    heatmap = np.zeros((20, 20), np.float32)
    assert paint_gaussian(heatmap, 10.3, 5.6, 1.5) == (10, 6)
    assert heatmap[6, 10] == 1.0 and heatmap.max() == 1.0 and heatmap[0, 0] == 0.0
    assert paint_gaussian(heatmap, -50, -50, 1.5) is None


def test_tile_targets(dataset):
    catalogue_dir, image_dir = dataset
    store = CoaddStore("train", catalogue_dir, image_dir, CONFIG)
    assert store.has_stars and len(store.stars) == 4 and len(store.truth) == 60
    assert set(store.component_images) == {"sfregions", "tidal", "spikes"}
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
    # truth maps: spikes cross the whole tile, tidal blobs and star-forming regions cover small areas
    for head in ("sfregion_map", "tidal_map", "spike_map"):
        mask = targets[head][..., 0]
        assert set(np.unique(mask)) <= {0.0, 1.0} and 0 < mask.mean() < 0.5, head
    assert store.tile_weight.min() >= 0.25 and store.tile_weight.max() <= CONFIG["max_tile_oversampling"]  # balanced



def test_only_visible_sources_are_positives(dataset):
    """Positives at S/N >= detectable_snr; between invisible_snr and that, no loss around the source; below,
    background (nothing painted, loss counted)."""
    catalogue_dir, image_dir = dataset
    store = CoaddStore("train", catalogue_dir, image_dir, CONFIG)
    maker = TargetMaker(store, CONFIG)
    maker.prepare(store)
    key, halo = "1y_fwhm110", CONFIG["tile_halo"]
    targets = maker.tile_targets(store, 0, 0, 0, key)
    snr = combined_snr(store, key, store.truth_flux, store.truth_re_arcsec / CONFIG["pixscale"], CONFIG)
    rows = store.truth_by_tile[0]
    zones = dict(positive=snr[rows] >= CONFIG["detectable_snr"],
                 hidden=(snr[rows] >= CONFIG["invisible_snr"]) & (snr[rows] < CONFIG["detectable_snr"]),
                 background=snr[rows] < CONFIG["invisible_snr"])
    assert all(zone.any() for zone in zones.values()), {name: int(zone.sum()) for name, zone in zones.items()}
    heatmap, valid = targets["galaxy_heatmap"][..., 0], targets["galaxy_heatmap"][..., 2]
    xi = np.rint(store.truth_x[rows]).astype(int) + halo
    yi = np.rint(store.truth_y[rows]).astype(int) + halo
    isolated = np.array([np.hypot(store.truth_x[rows] - store.truth_x[i], store.truth_y[rows] - store.truth_y[i])
                         .__lt__(15).sum() == 1 for i in rows])  # no other galaxy within 15 px
    for name, zone in zones.items():
        sel = zone & isolated
        assert sel.any(), name
        on_centre, counted = heatmap[yi[sel], xi[sel]], valid[yi[sel], xi[sel]]
        if name == "positive":
            assert np.all(on_centre == 1.0) and np.all(counted == 1.0)
        elif name == "hidden":
            assert np.all(on_centre < 1.0) and np.all(counted == 0.0)
        else:
            assert np.all(on_centre < 1.0) and np.all(counted == 1.0)


def test_balanced_weights(dataset):
    """weighting="balanced": in each coadd and S/N bin every galaxy class's positives weigh as much in total as the
    largest class's count there, stars weigh 1, sources below detectable_snr nothing; tiles drawn w times as often
    have their loss scaled by 1 / w (stratified sampling)."""
    catalogue_dir, image_dir = dataset
    cfg = {**CONFIG, "weighting": "balanced", "max_population_weight": 1e6}
    store = CoaddStore("train", catalogue_dir, image_dir, cfg)
    maker = TargetMaker(store, cfg)
    maker.prepare(store)
    for key in store.coadds:
        galaxy_weights, star_weights = maker.weights(store, key)
        galaxy_cells, star_cells = maker.cells(store, key)
        cells, weights = galaxy_cells[store.truth_inside], galaxy_weights[store.truth_inside]
        positive = cells != ""
        assert np.all(weights[~positive] == 0) and np.all(weights[positive] >= 1)
        stars = star_cells[store.star_inside] != ""
        assert np.all(star_weights[store.star_inside][stars] == 1) and np.all(star_weights[store.star_inside][~stars] == 0)
        table = pd.DataFrame(dict(cell=cells[positive], weight=weights[positive]))
        table["snr_bin"] = table["cell"].str.rsplit("|", n=1).str[1]
        totals = table.groupby(["snr_bin", "cell"])["weight"].sum()
        largest = table.groupby(["snr_bin", "cell"]).size().groupby(level=0).max()
        assert np.allclose(totals.to_numpy(), largest.reindex(totals.index.get_level_values(0)).to_numpy())
    assert np.allclose(store.tile_weight * store.tile_loss_scale, store.tile_weight.mean())


def test_footprint_negatives(dataset):
    """Balanced weighting with footprint_negatives: around a galaxy weighted w > 1 the centre heads' loss counts
    w^footprint_power times, and its own centre counts w in all (positive weight w / w^power on top); elsewhere 1."""
    catalogue_dir, image_dir = dataset
    for power in (1.0, 0.5):
        cfg = {**CONFIG, "weighting": "balanced", "footprint_negatives": True, "footprint_power": power,
               "max_population_weight": 1e6}
        store = CoaddStore("train", catalogue_dir, image_dir, cfg)
        maker = TargetMaker(store, cfg)
        maker.prepare(store)
        key, halo = "10y_fwhm110", CONFIG["tile_halo"]
        weights = maker.weights(store, key)[0]
        targets = maker.tile_targets(store, 0, 0, 0, key)["detection_heatmap"]
        boosted = [i for i in store.truth_by_tile[0] if weights[i] > 1]
        assert boosted
        i = max(boosted, key=lambda j: weights[j])
        cx, cy = int(np.rint(store.truth_x[i])) + halo, int(np.rint(store.truth_y[i])) + halo
        # the footprint counts w^power; the centre counts w in all (its positive weight times the footprint's)
        assert targets[cy, cx, 2] >= weights[i] ** power - 1e-3
        assert targets[cy, cx, 1] * targets[cy, cx, 2] == pytest.approx(weights[i], rel=1e-3)
        assert targets[halo + 2, halo + 2, 2] in (0.0, 1.0) or targets[halo + 2, halo + 2, 2] >= 1.0

def test_galaxy_classes(dataset):
    catalogue_dir, image_dir = dataset
    store = CoaddStore("test", catalogue_dir, image_dir, CONFIG)
    classes = galaxy_classes(store, 0.5)
    resolved = store.truth_re_arcsec >= 0.5
    assert np.array_equal(classes["unresolved"], ~resolved)
    by_size = [members for name, members in classes.items() if name.startswith("resolved, Re")]
    assert np.array_equal(np.sum(by_size, axis=0), resolved)  # every resolved galaxy in exactly one size bin
    assert [int(classes[name].sum()) for name in ("BCGs", "UDGs", "extended dIrrs", "almost-dark galaxies")] == \
        [1, 1, 1, 0]
    assert not classes["resolved ellipticals"][0]  # an E3, but a BCG: not of the main population
    types = store.truth_hubble_type
    main = np.arange(len(types)) >= 3
    assert np.array_equal(classes["resolved spirals"], resolved & main & np.isin(types, ["Sb", "SBc"]))
    assert np.array_equal(classes["resolved lenticulars"], resolved & main & (types == "S0"))


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
    planes = encode_planes(*first, dict(logvar_centre=[0.0] * 6, logvar_scale=[1.0] * 6), CONFIG["snr_stretches"])
    assert planes.shape == (320, 320, len(BANDS) * (1 + len(CONFIG["snr_stretches"]))) and np.isfinite(planes).all()



def test_extra_stretches(dataset):
    """A second, gentler stretch adds one plane per band after the usual planes, which stay unchanged; the network's
    input grows to match and band coverage still reads only the log-variance planes."""
    catalogue_dir, image_dir = dataset
    store = CoaddStore("valid", catalogue_dir, image_dir, CONFIG)
    signal, variance = store.coadd_tile("10y_fwhm110", 0, 0)
    norm = dict(logvar_centre=[0.0] * 6, logvar_scale=[1.0] * 6)
    one, two = encode_planes(signal, variance, norm, (3.0,)), encode_planes(signal, variance, norm, (3.0, 100.0))
    assert two.shape == (320, 320, 18) and np.array_equal(two[..., :12], one)
    assert np.allclose(two[..., 12 + 2], np.clip(np.arcsinh(signal[2] / np.sqrt(variance[2]) / 100.0), -8, 8),
                       atol=1e-5)
    model = build_unet({**CONFIG, "base_filters": 8, "snr_stretches": (3.0, 100.0)})
    assert model.inputs[0].shape[-1] == 18
    covered = np.asarray(band_coverage(two[None]))  # beyond the image edge (the halo at tile 0, 0) no band has data
    assert covered.shape == (1, 320, 320, 6) and np.array_equal(covered, np.asarray(band_coverage(one[None])))
    assert covered[0, 100, 100].all() and not covered[0, 5, 5].any()



def test_extended_head_targets(dataset):
    """extended_heatmap: positives only at visible resolved galaxies, every resolved (class, S/N) cell the same total
    weight; it is built after (and does not feed) the detection head."""
    catalogue_dir, image_dir = dataset
    cfg = {**CONFIG, "heads": CONFIG["heads"] + ("extended_heatmap",), "max_population_weight": 1e6}
    store = CoaddStore("train", catalogue_dir, image_dir, cfg)
    maker = TargetMaker(store, cfg)
    maker.prepare(store)
    key = "10y_fwhm110"
    weights = maker.extended_weights(store, key)
    cells = maker.cells(store, key)[0]
    resolved = store.truth_re_arcsec >= store.coadd_settings(key)[1]["r"]
    positive = store.truth_inside & (weights > 0)
    assert positive.any() and np.all(resolved[positive]) and np.all(cells[positive] != "")
    totals = [weights[positive & (cells == cell)].sum() for cell in np.unique(cells[positive])]
    assert np.allclose(totals, totals[0])
    targets = maker.tile_targets(store, 0, 0, 0, key)
    ext, det = targets["extended_heatmap"], targets["detection_heatmap"]
    assert 0 < (ext[..., 1] > 0).sum() <= (det[..., 1] > 0).sum()
    assert np.all(det[..., 0][ext[..., 1] > 0] >= 0.999)  # every extended positive is also a detection positive
    model = build_unet({**cfg, "base_filters": 8})
    names = [layer.name for layer in model.layers]
    assert names.index("extended_heatmap") > names.index("detection_heatmap")
    assert "detection_logit_extended_heatmap" not in names

def test_threshold_choice():
    assert wilson_lower(99, 100, 1.64) == pytest.approx(0.9566, abs=1e-4)
    p_real = np.linspace(1, 0, 1000)
    is_real = np.r_[np.ones(900), np.zeros(100)].astype(int)  # the top 900 are all real
    threshold, row, status = choose_threshold(p_real, is_real, 0.99, 1.64)
    assert status == "met" and row["purity_lower"] >= 0.99 and 0.09 < threshold < 0.2


CENTRE_HEADS = ("galaxy_heatmap", "sfregion_heatmap", "tidal_heatmap")  # a model with centre heatmaps only


def test_architecture_is_unchanged():
    # Parameter counts at the default width and the original input: a change here means saved models no longer load.
    assert build_unet({**CONFIG, "heads": CENTRE_HEADS, "snr_stretches": (3.0,)}).count_params() == 3_596_337


def test_added_heads_leave_detections_unchanged():
    small = {**CONFIG, "base_filters": 8, "snr_stretches": (3.0,)}
    old = build_unet({**small, "heads": CENTRE_HEADS})
    new = transfer_weights(old, small)
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
    assert 0.0 <= threshold["threshold"] <= 1.0 and set(threshold["band_sets"]) == {"ugrizy"}  # no band adapter
    config = json.loads((trained_model / ARTEFACTS["model_config"]).read_text())
    assert config["cfg"]["heads"] == list(CONFIG["heads"]) and config["cfg"]["edge_padding"] == "no_data"
    summary = evaluate_unet(catalogue_dir, image_dir, trained_model, ["1y_fwhm110", "10y_fwhm110"])
    assert list(summary["coadd"]) == ["1y_fwhm110", "10y_fwhm110"]
    assert summary["purity"].between(0, 1).all()
    assert {"extended_completeness", "star_completeness"} <= set(summary.columns)
    assert list(trained_model.glob("mep_*_test_completeness_vs_size.png"))
    by_class = pd.read_csv(trained_model / "mep_1y_fwhm110_10y_fwhm110_test_by_class.csv")
    assert {"stars", "unresolved", "UDGs", "resolved spirals"} <= set(by_class["source_class"])
    measured = by_class.dropna(subset=["completeness"])
    assert ((measured["completeness_low"] <= measured["completeness"])
            & (measured["completeness"] <= measured["completeness_high"])).all()
    assert (trained_model / "mep_1y_fwhm110_10y_fwhm110_ugrizy_test_completeness_by_class.png").exists()
