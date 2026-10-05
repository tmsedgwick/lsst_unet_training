"""Learn from visual inspection of a real coadd: re-choose the detection threshold, or fine-tune the network.

Labels come from the review tool in RunOnCoadd.ipynb (feedback_<category>.json: candidates the reviewer called real
or spurious, plus galaxies they marked as missed), or from a CSV with x, y, label (real / spurious) and optionally
weight. Positions are pixels of the real coadd, an .npz with signal and variance (band, y, x), psf_kernels and bands.

A review file covers one candidate category (U-Net only, peak finder only, both), of which the reviewer may have
inspected only a sample, so each reviewed candidate stands for n / n_reviewed candidates of its category: that is its
weight in purity and recall (missed galaxies weigh 1). Labels are split into train and test by square blocks of the
image, so test labels sit away from the training labels; the split depends only on the labels, block size, test
fraction and seed, so both commands reproduce it.

recalibrate_on_real: score every label with the model (the p_real of the best peak within real_match_radius_pix, or 0
if there is none), choose the lowest p_real whose weighted purity on the train labels meets real_target_purity
(Wilson lower bound with the Kish effective sample size), and compare the old and new thresholds on the real test
labels and the mock test coadds. Writes a copy of the model folder with the new threshold.

finetune_on_real: continue training at a low learning rate on batches that mix mock tiles (full truth) with real tiles
around train labels, where the loss is counted only within label_radius_pix of a label: a galaxy centre at a real
one, background at a spurious one, nothing elsewhere. The new model is calibrated on the mock calib catalogue as usual,
then both models are compared on the real test labels and the mock test coadds.

The labels only describe the candidates that were reviewed. A lower threshold also admits unreviewed peaks elsewhere
in the image, so the reports give the number of detections over the whole real coadd at each threshold too.
"""

import json
import re
import shutil
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

from .calibration import fit_calibrator, wilson_lower
from .coadd_data import CoaddStore, encode_planes
from .config import ARTEFACTS, BANDS, CATALOGUE_STEM, CONFIG
from .evaluation import calibrate_unet, score_test, summarise
from .peak_detection import predict_peaks
from .targets import HEATMAP_SIGMA_PIX, TargetMaker, paint_gaussian
from .tile_sequence import CoaddTileSequence
from .training import load_model, log_re_scaling, set_up_tensorflow
from .unet_model import compile_unet

DECISIONS = dict(real=True, spurious=False)  # anything else (e.g. unsure) is left out
DUPLICATE_RADIUS_PIX = 1.5  # labels closer than this are one source


class RealCoadd:
    """A real coadd behind the interface predict_peaks uses: one 'coadd', held in RAM, with its own PSF stamps."""

    key = "real"

    def __init__(self, path, cfg):
        data = np.load(path)
        bands = [str(band) for band in data["bands"]]
        order = [bands.index(band) for band in BANDS]
        self.signal = np.asarray(data["signal"], np.float32)[order]
        self.variance = np.asarray(data["variance"], np.float32)[order]
        kernels = np.asarray(data["psf_kernels"], np.float32)
        if kernels.shape[-1] != len(bands):  # (band, y, x) -> (y, x, band)
            kernels = np.moveaxis(kernels, 0, -1)
        self.psf = fit_stamps(kernels[..., order], cfg["psf_stamp"])
        self.name, self.cfg, self.coadds = Path(path).stem, cfg, [self.key]
        _, self.ny, self.nx = self.signal.shape

    extract_halo = CoaddStore.extract_halo
    tile_grid = CoaddStore.tile_grid
    sample_index = CoaddStore.sample_index

    def psf_kernels(self, key):
        return self.psf

    def full_coadd(self, key, seed):
        return self.signal, self.variance


def fit_stamps(kernels, size):
    """(k, k, band) PSF stamps centre-cropped or zero-padded to (size, size, band), each normalised to unit sum."""
    k = kernels.shape[0]
    if k > size:
        start = (k - size) // 2
        kernels = kernels[start:start + size, start:start + size]
    elif k < size:
        before = (size - k) // 2
        kernels = np.pad(kernels, ((before, size - k - before), (before, size - k - before), (0, 0)))
    return (kernels / np.maximum(kernels.sum(axis=(0, 1), keepdims=True), 1e-12)).astype(np.float32)


def review_labels(path):
    """Labels from one review-tool feedback file, weighted by the reviewed fraction of its category."""
    review = json.loads(Path(path).read_text())
    category = review.get("category", Path(path).stem)
    reviewed = pd.DataFrame(list(review.get("reviewed", {}).values()))
    frames = []
    if len(reviewed):
        decided = reviewed[reviewed["decision"].isin(DECISIONS)]
        weight = max(int(review.get("n", len(reviewed))), len(reviewed)) / len(reviewed)
        frames.append(pd.DataFrame(dict(x=decided["x"].to_numpy(float), y=decided["y"].to_numpy(float),
                                        is_real=decided["decision"].map(DECISIONS).to_numpy(bool), weight=weight,
                                        source=category)))
    missed = pd.DataFrame(review.get("missed", []))
    if len(missed):
        frames.append(pd.DataFrame(dict(x=missed["x"].to_numpy(float), y=missed["y"].to_numpy(float), is_real=True,
                                        weight=1.0, source=f"{category}_missed")))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["x", "y", "is_real", "weight",
                                                                                      "source"])


def csv_labels(path):
    """Labels from a CSV with x, y, label (real / spurious) and optionally weight."""
    table = pd.read_csv(path)
    label = table["label"].astype(str).str.strip().str.lower()
    table = table[label.isin(DECISIONS)]
    weight = table["weight"].to_numpy(float) if "weight" in table else 1.0
    return pd.DataFrame(dict(x=table["x"].to_numpy(float), y=table["y"].to_numpy(float),
                             is_real=label[label.isin(DECISIONS)].map(DECISIONS).to_numpy(bool), weight=weight,
                             source=Path(path).stem))


def read_labels(paths, nx=None, ny=None):
    """All labels from review JSON files and / or CSVs. Labels of one source closer than DUPLICATE_RADIUS_PIX are
    merged (first kept); groups whose decisions disagree are dropped. With nx, ny, labels off the image are dropped."""
    labels = pd.concat([review_labels(p) if Path(p).suffix == ".json" else csv_labels(p) for p in paths],
                       ignore_index=True)
    labels = labels[np.isfinite(labels["x"]) & np.isfinite(labels["y"])].reset_index(drop=True)
    if nx is not None:
        labels = labels[labels["x"].between(0, nx - 1) & labels["y"].between(0, ny - 1)].reset_index(drop=True)
    if len(labels) > 1:
        pairs = np.array(sorted(KDTree(labels[["x", "y"]].to_numpy()).query_pairs(DUPLICATE_RADIUS_PIX)))
        pairs = pairs.reshape(-1, 2)
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(labels),) * 2)
        _, group = connected_components(graph, directed=False)
        agree = labels.groupby(group)["is_real"].transform("nunique") == 1
        labels = labels[agree.to_numpy() & ~pd.Series(group).duplicated().to_numpy()].reset_index(drop=True)
    labels["is_real"] = labels["is_real"].astype(bool)
    return labels


def split_labels(labels, test_fraction, block_pix, seed):
    """Add split = train / test, by assigning whole block_pix squares of the image to the test set at random."""
    column, row = ((labels[axis] // block_pix).astype(int).astype(str) for axis in ("x", "y"))
    block = column + "," + row
    blocks = np.array(sorted(set(block)))
    n_test = int(np.clip(round(test_fraction * len(blocks)), 1, max(len(blocks) - 1, 1)))
    test_blocks = set(np.random.default_rng(seed).permutation(blocks)[:n_test])
    return labels.assign(split=np.where(block.isin(test_blocks), "test", "train"))


def score_labels(labels, peaks, radius):
    """Each label's p_real: that of the best peak within radius pixels, or 0 if none (undetected at any threshold)."""
    p_real = np.zeros(len(labels))
    if len(peaks) and len(labels):
        values = peaks["p_real"].to_numpy(float)
        near = KDTree(peaks[["x", "y"]].to_numpy(float)).query_ball_point(labels[["x", "y"]].to_numpy(float), r=radius)
        p_real = np.array([values[n].max() if len(n) else 0.0 for n in near])
    return p_real


def label_metrics(p_real, is_real, weight, threshold):
    """Counts and weighted recall / purity of the labels kept at a threshold (a label with p_real 0 is never kept)."""
    p_real, is_real, weight = np.asarray(p_real, float), np.asarray(is_real, bool), np.asarray(weight, float)
    kept = (p_real > 0) & (p_real >= threshold)
    w_kept = weight[kept].sum()
    return dict(threshold=float(threshold), real_found=f"{int((kept & is_real).sum())}/{int(is_real.sum())}",
                spurious_kept=f"{int((kept & ~is_real).sum())}/{int((~is_real).sum())}",
                recall=float(weight[kept & is_real].sum() / max(weight[is_real].sum(), 1e-12)),
                purity=float(weight[kept & is_real].sum() / w_kept) if w_kept > 0 else np.nan)


def choose_weighted_threshold(p_real, is_real, weight, target_purity, z):
    """(threshold, dict(purity, purity_lower, n_eff), status): the lowest p_real whose weighted purity above it meets
    the target by its Wilson lower bound with the Kish effective sample size ('met'), else the best one ('unmet')."""
    p_real, is_real, weight = np.asarray(p_real, float), np.asarray(is_real, bool), np.asarray(weight, float)
    rows = []
    for threshold in np.unique(p_real[p_real > 0])[::-1]:
        w = weight[p_real >= threshold]
        purity = weight[(p_real >= threshold) & is_real].sum() / w.sum()
        n_eff = w.sum() ** 2 / (w ** 2).sum()
        rows.append((threshold, dict(purity=float(purity), purity_lower=float(wilson_lower(purity * n_eff, n_eff, z)),
                                     n_eff=float(n_eff))))
    if not rows:
        raise RuntimeError("no train label lies on a U-Net peak, so no threshold can be chosen")
    meets = [i for i, (_, row) in enumerate(rows) if row["purity_lower"] >= target_purity]
    i = meets[-1] if meets else int(np.argmax([row["purity_lower"] for _, row in rows]))
    return float(rows[i][0]), rows[i][1], "met" if meets else "unmet"


def derived_model_dir(model_dir, suffix):
    """<model_dir>_<suffix>, which must not exist yet: a derived model never replaces another one."""
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*", str(suffix or "")):
        raise ValueError(f"suffix {suffix!r}: use letters, digits, '.', '_' or '-'")
    out = Path(model_dir).parent / f"{Path(model_dir).name}_{suffix}"
    if out.exists():
        raise FileExistsError(f"{out} already exists: choose another suffix")
    return out


def real_peaks(model, normalisation, model_config, coadd, calib_peaks):
    """Peaks over the whole real coadd, with p_real from the model's mock calibration."""
    peaks = predict_peaks(model, coadd, normalisation, model_config["cfg"], log_re_scaling(model_config))
    peaks["p_real"] = fit_calibrator(calib_peaks).predict(peaks["raw_score"]) if len(peaks) else []
    return peaks


def mock_summary(catalogue_dir, image_dir, model_dir, threshold, coadds, test_name, stem, tile_cap, cfg=None):
    """Per-coadd mock test purity / completeness of a model at a threshold (matched peaks are cached per model)."""
    store, peaks = score_test(catalogue_dir, image_dir, model_dir, coadds, cfg, test_name, stem, tile_cap)
    return summarise(store, peaks, threshold, store.cfg)[0], (store, peaks)


def comparison_table(rows):
    """{name: metrics dict} -> one DataFrame, for printing."""
    return pd.DataFrame.from_dict(rows, orient="index")


def report(title, table):
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(f"\n{title}\n{table.round(4).to_string()}")


def labelled_split(labels_paths, coadd, cfg):
    labels = read_labels(labels_paths, coadd.nx, coadd.ny)
    labels = split_labels(labels, cfg["real_test_fraction"], cfg["real_block_pix"], cfg["seed"])
    for split, group in labels.groupby("split"):
        print(f"{split}: {int(group['is_real'].sum())} real, {int((~group['is_real']).sum())} spurious labels "
              f"({', '.join(f'{s} {n}' for s, n in group['source'].value_counts().items())})")
    return labels


def real_comparison(labels, scores, thresholds, peaks):
    """Train and test label metrics plus whole-image detection counts for each (name: threshold)."""
    tables = {}
    for split in ["train", "test"]:
        part = labels["split"].to_numpy() == split
        tables[split] = comparison_table({
            name: dict(**label_metrics(scores[name][part], labels["is_real"][part], labels["weight"][part], threshold),
                       image_detections=int((peaks[name]["p_real"] >= threshold).sum()))
            for name, threshold in thresholds.items()})
    return tables


def mock_comparison(summaries):
    columns = dict(n_detections="detections", purity="purity", completeness_above_limit="complete_above",
                   completeness_below_limit="complete_below")
    return pd.concat({name: summary.set_index("coadd")[list(columns)].rename(columns=columns)
                      for name, summary in summaries.items()}, axis=1)


def write_report(out, tables, mock, extra):
    (out / "mep_real_feedback_report.json").write_text(json.dumps(dict(
        **extra, real={split: json.loads(t.to_json(orient="index")) for split, t in tables.items()},
        mock=json.loads(mock.to_json(orient="split"))), indent=2, default=str))
    mock.to_csv(out / "mep_real_feedback_mock_test_summary.csv")


def recalibrate_on_real(model_dir, real_coadd, label_paths, catalogue_dir, image_dir, suffix, cfg=None,
                        stem=CATALOGUE_STEM, test_name="test", mock_coadds=None, tile_cap=None):
    """Choose a new p_real threshold on the real train labels; write <model_dir>_<suffix> with it. Returns the new
    folder."""
    model_dir = Path(model_dir)
    out = derived_model_dir(model_dir, suffix)
    model, normalisation, model_config = load_model(model_dir, cfg)
    cfg = model_config["cfg"]
    old = json.loads((model_dir / ARTEFACTS["threshold"]).read_text())
    calib_peaks = pd.read_parquet(model_dir / ARTEFACTS["calib_peaks"])

    coadd = RealCoadd(real_coadd, cfg)
    labels = labelled_split(label_paths, coadd, cfg)
    peaks = real_peaks(model, normalisation, model_config, coadd, calib_peaks)
    labels["p_real"] = score_labels(labels, peaks, cfg["real_match_radius_pix"])

    train = labels[labels["split"] == "train"]
    threshold, row, status = choose_weighted_threshold(train["p_real"], train["is_real"], train["weight"],
                                                       cfg["real_target_purity"], cfg["wilson_z"])
    print(f"\nNew p_real threshold {threshold:.5f} (was {old['threshold']:.5f}): weighted purity on real train labels "
          f"{row['purity']:.4f}, Wilson lower {row['purity_lower']:.4f} (n_eff {row['n_eff']:.1f}, target "
          f"{cfg['real_target_purity']}, {status})")
    if status != "met":
        print("WARNING: the target purity could not be certified on the real train labels; this is the threshold with "
              "the best lower bound.")

    thresholds = dict(before=old["threshold"], after=threshold)
    tables = real_comparison(labels, {name: labels["p_real"].to_numpy() for name in thresholds}, thresholds,
                             {name: peaks for name in thresholds})
    report("Real train labels (recall and purity weighted by review sampling)", tables["train"])
    report("Real test labels", tables["test"])

    coadds = mock_coadds or [cfg["reference_coadd"]]
    store, mock_peaks = score_test(catalogue_dir, image_dir, model_dir, coadds, cfg, test_name, stem, tile_cap)
    mock = mock_comparison({name: summarise(store, mock_peaks, value, store.cfg)[0]
                            for name, value in thresholds.items()})
    report("Mock test coadds", mock)

    out.mkdir(parents=True)
    for artefact in ("weights", "normalisation", "model_config", "history", "calib_peaks"):
        if (model_dir / ARTEFACTS[artefact]).exists():
            shutil.copy2(model_dir / ARTEFACTS[artefact], out / ARTEFACTS[artefact])
    (out / ARTEFACTS["threshold"]).write_text(json.dumps(dict(
        threshold=threshold, target_purity=cfg["real_target_purity"], status=status,
        reference_combo=old.get("reference_combo"), chosen_on="real train labels", real_coadd=str(real_coadd),
        labels=[str(p) for p in label_paths], previous_threshold=old["threshold"], previous_model=str(model_dir),
        **row),
        indent=2))
    labels.to_csv(out / "mep_real_labels.csv", index=False)
    write_report(out, tables, mock, dict(threshold=threshold, previous_threshold=old["threshold"], status=status))
    print(f"\nSaved the recalibrated model -> {out} (the original in {model_dir} is unchanged)")
    return out


class RealTiles:
    """Training tiles of the real coadd around train labels, with partial-label targets and random rotations."""

    def __init__(self, coadd, labels, normalisation, cfg):
        self.coadd, self.normalisation, self.cfg = coadd, normalisation, cfg
        self.train = labels[labels["split"] == "train"].reset_index(drop=True)
        self.test_xy = labels.loc[labels["split"] == "test", ["x", "y"]].to_numpy(float)
        self.by_class = [np.flatnonzero(self.train["is_real"].to_numpy() == value) for value in (True, False)]
        self.by_class = [rows for rows in self.by_class if len(rows)]
        full = cfg["tile_size"] + 2 * cfg["tile_halo"]
        self.yy, self.xx = np.mgrid[0:full, 0:full]

    def targets(self, x0, y0):
        """Targets of the tile at (x0, y0): the loss is counted only within label_radius_pix of a train label (and
        not near a test label), weighted finetune_real_weight; real labels are galaxy centres, spurious ones
        background. The other heads get no loss."""
        size, halo, radius = self.cfg["tile_size"], self.cfg["tile_halo"], self.cfg["label_radius_pix"]
        full = size + 2 * halo
        heatmap, centre_weight, valid = (np.zeros((full, full), np.float32) for _ in range(3))

        def disks(xy):
            local = xy - [x0 - halo, y0 - halo]
            local = local[(local >= -radius).all(axis=1) & (local < full + radius).all(axis=1)]
            mask = np.zeros((full, full), bool)
            for lx, ly in local:
                mask |= (self.xx - lx) ** 2 + (self.yy - ly) ** 2 <= radius ** 2
            return mask, local

        mask, _ = disks(self.train[["x", "y"]].to_numpy(float))
        valid[mask] = self.cfg["finetune_real_weight"]
        valid[disks(self.test_xy)[0]] = 0.0
        in_tile = np.zeros((full, full), bool)
        in_tile[halo:halo + min(size, self.coadd.ny - y0), halo:halo + min(size, self.coadd.nx - x0)] = True
        valid[~in_tile] = 0.0
        for lx, ly in disks(self.train.loc[self.train["is_real"], ["x", "y"]].to_numpy(float))[1]:
            centre = paint_gaussian(heatmap, lx, ly, HEATMAP_SIGMA_PIX["galaxy"])
            if centre is not None:
                centre_weight[centre[1], centre[0]] = 1.0
        none = np.zeros((full, full), np.float32)
        return {"galaxy_heatmap": np.dstack([heatmap, centre_weight, valid]),
                "centroid_offset": np.zeros((full, full, 3), np.float32),
                "source_structure": np.zeros((full, full, 5), np.float32),
                "clump_heatmap": np.dstack([none, none, none]), "tidal_heatmap": np.dstack([none, none, none])}

    def sample(self, rng):
        """(image planes, PSF stamps, targets) of a tile holding a random train label (real and spurious equally
        likely) at a random place, rotated and flipped at random."""
        size = self.cfg["tile_size"]
        rows = self.by_class[rng.integers(len(self.by_class))]
        label = self.train.iloc[int(rng.choice(rows))]
        x0 = int(np.clip(label["x"] - rng.uniform(0, size), 0, max(self.coadd.nx - size, 0)))
        y0 = int(np.clip(label["y"] - rng.uniform(0, size), 0, max(self.coadd.ny - size, 0)))
        planes = encode_planes(self.coadd.extract_halo(self.coadd.signal, x0, y0),
                               self.coadd.extract_halo(self.coadd.variance, x0, y0), self.normalisation)
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))

        def augment(array):
            array = np.rot90(array, turns, axes=(0, 1))
            return np.ascontiguousarray(array[:, ::-1] if flip else array)

        return (augment(planes), augment(self.coadd.psf),
                {name: augment(value) for name, value in self.targets(x0, y0).items()})


class MixedSequence(keras.utils.PyDataset):
    """Batches of finetune_real_fraction real tiles (RealTiles) and the rest mock tiles (a CoaddTileSequence)."""

    def __init__(self, mock_sequence, real_tiles, cfg, workers=1):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=16)
        self.mock, self.real = mock_sequence, real_tiles
        self.batch_size = int(cfg["batch_size"])
        self.n_real = int(np.clip(round(cfg["finetune_real_fraction"] * self.batch_size), 1, self.batch_size))
        self.steps, self.seed, self.epoch = int(cfg["finetune_steps_per_epoch"]), int(cfg["seed"]), 0

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        self.epoch += 1
        self.mock.on_epoch_end()

    def __getitem__(self, step):
        rng = np.random.default_rng([self.seed, self.epoch, step])
        real = [self.real.sample(rng) for _ in range(self.n_real)]
        images, psfs = [r[0] for r in real], [r[1] for r in real]
        targets = {name: [r[2][name] for r in real] for name in real[0][2]}
        n_mock = self.batch_size - self.n_real
        if n_mock:
            inputs, mock_targets = self.mock[int(rng.integers(len(self.mock)))]
            images += list(inputs["image_planes"][:n_mock])
            psfs += list(inputs["psf_kernels"][:n_mock])
            for name in targets:
                targets[name] += list(mock_targets[name][:n_mock])
        return ({"image_planes": np.stack(images), "psf_kernels": np.stack(psfs)},
                {name: np.stack(value) for name, value in targets.items()})


def finetune_on_real(model_dir, real_coadd, label_paths, catalogue_dir, image_dir, suffix, cfg=None,
                     stem=CATALOGUE_STEM, train_name="train", valid_name="valid", calib_name="calib", test_name="test",
                     mock_coadds=None, tile_cap=None):
    """Fine-tune on real train labels mixed with mock tiles, calibrate on the mock calib catalogue, and compare with
    the original model on the real test labels and the mock test coadds. Writes <model_dir>_<suffix>; returns it."""
    model_dir = Path(model_dir)
    out = derived_model_dir(model_dir, suffix)
    user_cfg = dict(cfg or {})
    set_up_tensorflow({**CONFIG, **user_cfg}["seed"])
    model, normalisation, model_config = load_model(model_dir, user_cfg)
    cfg = model_config["cfg"]
    old_threshold = json.loads((model_dir / ARTEFACTS["threshold"]).read_text())["threshold"]

    coadd = RealCoadd(real_coadd, cfg)
    labels = labelled_split(label_paths, coadd, cfg)
    peaks = dict(before=real_peaks(model, normalisation, model_config, coadd,
                                   pd.read_parquet(model_dir / ARTEFACTS["calib_peaks"])))

    train_store = CoaddStore(train_name, catalogue_dir, image_dir, cfg, stem)
    valid_store = CoaddStore(valid_name, catalogue_dir, image_dir, cfg, stem)
    target_maker = TargetMaker(train_store, cfg)
    target_maker.log_re_mean, target_maker.log_re_std = log_re_scaling(model_config)  # keep the model's size scale
    for store in (train_store, valid_store):
        target_maker.prepare(store)
    mock_sequence = CoaddTileSequence(train_store, None, normalisation, cfg, target_maker, shuffle=True,
                                      fresh_noise=True, resample=True, coadds_per_tile=cfg["train_coadds_per_tile"],
                                      seed=cfg["seed"])
    valid_index = valid_store.sample_index()
    valid_index = valid_index.sample(n=min(cfg["valid_samples"], len(valid_index)),
                                     random_state=cfg["seed"]).reset_index(drop=True)
    valid_sequence = CoaddTileSequence(valid_store, valid_index, normalisation, cfg, target_maker,
                                       workers=cfg["data_workers"])
    sequence = MixedSequence(mock_sequence, RealTiles(coadd, labels, normalisation, cfg), cfg, cfg["data_workers"])
    print(f"Fine-tuning: {cfg['finetune_epochs']} epochs x {len(sequence)} steps, {sequence.n_real} real + "
          f"{sequence.batch_size - sequence.n_real} mock tiles per batch, learning rate {cfg['finetune_learning_rate']}"
          f"; validation loss on {len(valid_index)} mock valid tiles")

    out.mkdir(parents=True)
    compile_unet(model, {**cfg, "learning_rate": cfg["finetune_learning_rate"]})
    baseline = model.evaluate(valid_sequence, return_dict=True, verbose=0)["galaxy_heatmap_loss"]
    print(f"Mock valid galaxy-heatmap loss before fine-tuning: {baseline:.5f}")
    model.fit(sequence, validation_data=valid_sequence, epochs=cfg["finetune_epochs"],
              callbacks=[keras.callbacks.CSVLogger(out / "mep_unet_finetune_history.csv")])
    model.save_weights(out / ARTEFACTS["weights"])
    shutil.copy2(model_dir / ARTEFACTS["normalisation"], out / ARTEFACTS["normalisation"])
    settings = ("finetune_learning_rate", "finetune_epochs", "finetune_steps_per_epoch", "finetune_real_fraction",
                "finetune_real_weight", "label_radius_pix", "real_test_fraction", "real_block_pix", "seed")
    (out / ARTEFACTS["model_config"]).write_text(json.dumps(dict(
        {key: value for key, value in model_config.items() if key != "cfg"},
        cfg=json.loads((model_dir / ARTEFACTS["model_config"]).read_text())["cfg"], finetuned_from=str(model_dir),
        real_coadd=str(real_coadd), labels=[str(p) for p in label_paths], finetune={k: cfg[k] for k in settings},
        mock_valid_loss_before=baseline), indent=2))

    print("\nCalibrating the fine-tuned model on the mock calib catalogue...")
    new_threshold = calibrate_unet(catalogue_dir, image_dir, out, user_cfg, calib_name, stem)
    new_model, _, new_config = load_model(out, user_cfg)
    peaks["after"] = real_peaks(new_model, normalisation, new_config, coadd,
                                pd.read_parquet(out / ARTEFACTS["calib_peaks"]))

    thresholds = dict(before=old_threshold, after=new_threshold)
    scores = {name: score_labels(labels, peaks[name], cfg["real_match_radius_pix"]) for name in thresholds}
    labels["p_real_before"], labels["p_real_after"] = scores["before"], scores["after"]
    tables = real_comparison(labels, scores, thresholds, peaks)
    report("Real train labels (original model -> fine-tuned; recall and purity weighted by review sampling)",
           tables["train"])
    report("Real test labels", tables["test"])

    coadds = mock_coadds or [cfg["reference_coadd"]]
    mock = mock_comparison({name: mock_summary(catalogue_dir, image_dir, directory, thresholds[name], coadds,
                                               test_name, stem, tile_cap, user_cfg)[0]
                            for name, directory in dict(before=model_dir, after=out).items()})
    report("Mock test coadds", mock)

    labels.to_csv(out / "mep_real_labels.csv", index=False)
    write_report(out, tables, mock, dict(threshold=new_threshold, previous_threshold=old_threshold,
                                         finetuned_from=str(model_dir)))
    print(f"\nSaved the fine-tuned model -> {out} (the original in {model_dir} is unchanged). To also tune its "
          "threshold on the real labels, run scripts/recalibrate_on_real.py on it.")
    return out
