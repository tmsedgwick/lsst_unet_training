"""Update a trained model using labelled detections on an extra ("auxiliary") coadd. This could be from real data or
another mock.

The model is trained and calibrated on mock coadds. The auxiliary coadd is any other image with labelled detections:
a real coadd inspected by eye, a mock made with different settings, a reprocessed image. Two updates are offered:

update_threshold_on_aux
    Keep the network, change only the detection threshold. The new threshold is the lowest p_real at which the
    labelled detections above it are pure enough (aux_purity, judged on the training half of the labels).

update_weights_on_aux
    Keep training the network (fine-tuning) on a mix of auxiliary and mock tiles, at a low learning rate, then
    recalibrate it on the mock calib catalogue as usual.

Both write a new model folder <model_dir>_<suffix> and never modify the original. Both print, before and after the
update, how many labelled sources and spurious detections are detected on the held-out half of the labels, how many
detections there are over the whole auxiliary coadd, and purity / completeness on the mock test coadds (to check the
update has not made the model worse on mocks).

Labels
------
A label is a pixel position on the auxiliary coadd marked either as a source (a genuine galaxy or star) or as
spurious (an artefact or noise). They are read from:
  * review JSON files (feedback_<category>.json), one per candidate category (e.g. "detected by the U-Net only"):
    {"category": ..., "n": number of candidates in the category,
     "reviewed": {id: {"x": ..., "y": ..., "decision": "real" | "spurious" | "unsure"}, ...},
     "missed": [{"x": ..., "y": ...}, ...]}
    "real" marks a source, "spurious" an artefact; "unsure" decisions are ignored; "missed" lists sources the
    reviewer found that no detection caught.
  * CSV files with columns x, y, label ("real" or "spurious") and optionally weight.

Weights: a reviewer may inspect only a sample of a category (e.g. 57 of 11,324 candidates), while another category is
inspected in full. So that each category counts in proportion to its size, a reviewed candidate gets weight
n_candidates / n_reviewed for its category. Missed sources, and CSV rows without a weight, get weight 1.

Train / test split: the image is cut into square blocks of aux_block_pix pixels and a random aux_test_fraction of the
blocks is held out for testing, so test labels are spatially separated from the labels the update learns from. The
split depends only on the labels, the block size, the test fraction and the seed, so both commands hold out the same
labels.

"p_real" (used throughout this package) is the model's calibrated probability that a peak is a genuine source. Each
label is given the p_real of the highest-scoring U-Net peak within aux_match_radius_pix of it, or 0 if there is no
peak nearby, in which case it counts as undetected at every threshold.

Caveat: the labels only describe the candidates that were reviewed. Lowering the threshold also lets through
unreviewed peaks elsewhere in the image, so the reports also give the number of detections over the whole coadd.
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

# Review decisions that become labels, mapped to "is a source". Anything else (e.g. "unsure") is ignored.
DECISIONS = dict(real=True, spurious=False)
DUPLICATE_RADIUS_PIX = 1.5  # labels closer together than this are taken to be the same object


class AuxCoadd:
    """The auxiliary coadd, read from an .npz file with arrays signal and variance (band, y, x), psf_kernels
    (y, x, band) and bands (band names).

    The detector's inference code (peak_detection.predict_peaks) was written for CoaddStore, which serves the many
    mock coadds of one catalogue. This class provides the same methods (tile_grid, sample_index, full_coadd,
    psf_kernels, extract_halo) for a single coadd, so that inference runs on it unchanged. The whole image is kept in
    memory. Bands are reordered to the package's band order and the PSF stamps are resized to the stamp size the
    network expects.
    """

    COADD_KEY = "aux"  # the name of the single coadd, where CoaddStore would use e.g. "10y_fwhm110"

    def __init__(self, path, cfg):
        data = np.load(path)
        bands = [str(band) for band in data["bands"]]
        order = [bands.index(band) for band in BANDS]
        self.signal = np.asarray(data["signal"], np.float32)[order]
        self.variance = np.asarray(data["variance"], np.float32)[order]
        kernels = np.asarray(data["psf_kernels"], np.float32)
        if kernels.shape[-1] != len(bands):  # stored as (band, y, x): move bands last
            kernels = np.moveaxis(kernels, 0, -1)
        self.psf = resize_psf_stamps(kernels[..., order], cfg["psf_stamp"])
        self.name, self.cfg, self.coadds = Path(path).stem, cfg, [self.COADD_KEY]
        _, self.ny, self.nx = self.signal.shape

    # Tiling helpers shared with CoaddStore; they only use self.cfg, self.nx, self.ny and self.coadds.
    extract_halo = CoaddStore.extract_halo
    tile_grid = CoaddStore.tile_grid
    sample_index = CoaddStore.sample_index

    def psf_kernels(self, key):
        return self.psf

    def full_coadd(self, key, seed):
        return self.signal, self.variance


def resize_psf_stamps(kernels, size):
    """Centre-crop or zero-pad (k, k, band) PSF stamps to (size, size, band) and normalise each to unit sum."""
    k = kernels.shape[0]
    if k > size:
        start = (k - size) // 2
        kernels = kernels[start:start + size, start:start + size]
    elif k < size:
        before = (size - k) // 2
        kernels = np.pad(kernels, ((before, size - k - before), (before, size - k - before), (0, 0)))
    return (kernels / np.maximum(kernels.sum(axis=(0, 1), keepdims=True), 1e-12)).astype(np.float32)


LABEL_COLUMNS = ["x", "y", "is_source", "weight", "category"]


def review_labels(path):
    """Labels from one review JSON file: columns x, y, is_source, weight and category (see the module docstring)."""
    review = json.loads(Path(path).read_text())
    category = review.get("category", Path(path).stem)
    reviewed = pd.DataFrame(list(review.get("reviewed", {}).values()))
    frames = []
    if len(reviewed):
        decided = reviewed[reviewed["decision"].isin(DECISIONS)]
        n_candidates = max(int(review.get("n", len(reviewed))), len(reviewed))
        frames.append(pd.DataFrame(dict(x=decided["x"].to_numpy(float), y=decided["y"].to_numpy(float),
                                        is_source=decided["decision"].map(DECISIONS).to_numpy(bool),
                                        weight=n_candidates / len(reviewed), category=category)))
    missed = pd.DataFrame(review.get("missed", []))
    if len(missed):
        frames.append(pd.DataFrame(dict(x=missed["x"].to_numpy(float), y=missed["y"].to_numpy(float), is_source=True,
                                        weight=1.0, category=f"{category}_missed")))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LABEL_COLUMNS)


def csv_labels(path):
    """Labels from a CSV with columns x, y, label ("real" or "spurious") and optionally weight."""
    table = pd.read_csv(path)
    label = table["label"].astype(str).str.strip().str.lower()
    keep = label.isin(DECISIONS)
    weight = table.loc[keep, "weight"].to_numpy(float) if "weight" in table else 1.0
    return pd.DataFrame(dict(x=table.loc[keep, "x"].to_numpy(float), y=table.loc[keep, "y"].to_numpy(float),
                             is_source=label[keep].map(DECISIONS).to_numpy(bool), weight=weight,
                             category=Path(path).stem))


def read_labels(paths, nx=None, ny=None):
    """All labels from a list of review JSON files and / or CSVs, as one table (columns LABEL_COLUMNS).

    Labels closer together than DUPLICATE_RADIUS_PIX are treated as one object: if they agree, the first is kept; if
    they disagree (one says source, another spurious), all are dropped. If the image size nx, ny is given, labels
    outside the image are dropped.
    """
    labels = pd.concat([review_labels(p) if Path(p).suffix == ".json" else csv_labels(p) for p in paths],
                       ignore_index=True)
    labels = labels[np.isfinite(labels["x"]) & np.isfinite(labels["y"])].reset_index(drop=True)
    if nx is not None:
        labels = labels[labels["x"].between(0, nx - 1) & labels["y"].between(0, ny - 1)].reset_index(drop=True)
    if len(labels) > 1:
        close_pairs = np.array(sorted(KDTree(labels[["x", "y"]].to_numpy()).query_pairs(DUPLICATE_RADIUS_PIX)))
        close_pairs = close_pairs.reshape(-1, 2)
        graph = coo_matrix((np.ones(len(close_pairs)), (close_pairs[:, 0], close_pairs[:, 1])),
                           shape=(len(labels),) * 2)
        _, group = connected_components(graph, directed=False)  # group[i]: which object label i belongs to
        agree = labels.groupby(group)["is_source"].transform("nunique") == 1
        first_of_group = ~pd.Series(group).duplicated()
        labels = labels[agree.to_numpy() & first_of_group.to_numpy()].reset_index(drop=True)
    labels["is_source"] = labels["is_source"].astype(bool)
    return labels


def split_labels(labels, test_fraction, block_pix, seed):
    """Add a column split ("train" or "test"): a random test_fraction of the block_pix x block_pix image blocks that
    contain labels is held out for testing (at least one block in each half when possible)."""
    block_x, block_y = ((labels[axis] // block_pix).astype(int).astype(str) for axis in ("x", "y"))
    block = block_x + "," + block_y
    blocks = np.array(sorted(set(block)))
    n_test = int(np.clip(round(test_fraction * len(blocks)), 1, max(len(blocks) - 1, 1)))
    test_blocks = set(np.random.default_rng(seed).permutation(blocks)[:n_test])
    return labels.assign(split=np.where(block.isin(test_blocks), "test", "train"))


def label_p_real(labels, peaks, radius):
    """For each label, the p_real of the best peak within radius pixels, or 0 if there is none."""
    p_real = np.zeros(len(labels))
    if len(peaks) and len(labels):
        peak_p_real = peaks["p_real"].to_numpy(float)
        nearby = KDTree(peaks[["x", "y"]].to_numpy(float)).query_ball_point(labels[["x", "y"]].to_numpy(float),
                                                                            r=radius)
        p_real = np.array([peak_p_real[found].max() if len(found) else 0.0 for found in nearby])
    return p_real


def label_metrics(p_real, is_source, weight, threshold):
    """How the labels fare at a threshold. A label is detected if its p_real is positive and at least the threshold.

    Returns counts (sources detected out of all sources, spurious detections out of all spurious labels) and, using
    the weights, recall (fraction of sources detected) and purity (fraction of detected labels that are sources).
    """
    p_real, is_source, weight = np.asarray(p_real, float), np.asarray(is_source, bool), np.asarray(weight, float)
    detected = (p_real > 0) & (p_real >= threshold)
    detected_weight = weight[detected].sum()
    return dict(threshold=float(threshold),
                sources_detected=f"{int((detected & is_source).sum())}/{int(is_source.sum())}",
                spurious_detected=f"{int((detected & ~is_source).sum())}/{int((~is_source).sum())}",
                recall=float(weight[detected & is_source].sum() / max(weight[is_source].sum(), 1e-12)),
                purity=float(weight[detected & is_source].sum() / detected_weight) if detected_weight > 0 else np.nan)


def choose_weighted_threshold(p_real, is_source, weight, purity_goal, z):
    """The lowest threshold whose weighted purity is at least purity_goal with ~95% confidence.

    Every distinct positive p_real is tried as a threshold. For the labels at or above it, the weighted purity is
    computed, and its Wilson lower bound (z standard deviations) is taken with the effective number of labels
    n_eff = (sum of weights)^2 / (sum of squared weights), so that a few heavily weighted labels do not count as many.

    Returns (threshold, dict(purity, purity_lower, n_eff), status), where status is "met" if some threshold reaches
    purity_goal, or "unmet", in which case the threshold with the highest lower bound is returned.
    """
    p_real, is_source, weight = np.asarray(p_real, float), np.asarray(is_source, bool), np.asarray(weight, float)
    candidates = []
    for threshold in np.unique(p_real[p_real > 0])[::-1]:  # highest first
        above = p_real >= threshold
        purity = weight[above & is_source].sum() / weight[above].sum()
        n_eff = weight[above].sum() ** 2 / (weight[above] ** 2).sum()
        candidates.append((threshold, dict(purity=float(purity), n_eff=float(n_eff),
                                           purity_lower=float(wilson_lower(purity * n_eff, n_eff, z)))))
    if not candidates:
        raise RuntimeError("no training label has a U-Net peak nearby, so no threshold can be chosen")
    meeting_goal = [i for i, (_, stats) in enumerate(candidates) if stats["purity_lower"] >= purity_goal]
    if meeting_goal:
        i = meeting_goal[-1]  # the last, i.e. lowest, threshold that meets the goal
    else:
        i = int(np.argmax([stats["purity_lower"] for _, stats in candidates]))
    return float(candidates[i][0]), candidates[i][1], "met" if meeting_goal else "unmet"


def new_model_dir(model_dir, suffix):
    """The folder for an updated model, <model_dir>_<suffix>. It must not exist yet, so no model is ever overwritten."""
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*", str(suffix or "")):
        raise ValueError(f"suffix {suffix!r}: use letters, digits, '.', '_' or '-'")
    out = Path(model_dir).parent / f"{Path(model_dir).name}_{suffix}"
    if out.exists():
        raise FileExistsError(f"{out} already exists: choose another suffix")
    return out


def predict_aux_peaks(model, normalisation, model_config, coadd, calib_peaks):
    """Run the model over the whole auxiliary coadd: one row per peak, with p_real from the model's mock
    calibration (calib_peaks)."""
    peaks = predict_peaks(model, coadd, normalisation, model_config["cfg"], log_re_scaling(model_config))
    peaks["p_real"] = fit_calibrator(calib_peaks).predict(peaks["raw_score"]) if len(peaks) else []
    return peaks


def mock_test_summary(catalogue_dir, image_dir, model_dir, threshold, coadds, test_name, stem, tile_cap, cfg=None):
    """Per-coadd purity and completeness of a model on the mock test catalogue at the given threshold."""
    store, peaks = score_test(catalogue_dir, image_dir, model_dir, coadds, cfg, test_name, stem, tile_cap)
    return summarise(store, peaks, threshold, store.cfg)[0]


def print_table(title, table):
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(f"\n{title}\n{table.round(4).to_string()}")


def load_and_split_labels(label_paths, coadd, cfg):
    """Read the labels, split them into train and test, and print how many of each kind there are."""
    labels = read_labels(label_paths, coadd.nx, coadd.ny)
    labels = split_labels(labels, cfg["aux_test_fraction"], cfg["aux_block_pix"], cfg["seed"])
    for split, group in labels.groupby("split"):
        by_category = ", ".join(f"{category} {n}" for category, n in group["category"].value_counts().items())
        print(f"{split} labels: {int(group['is_source'].sum())} sources, {int((~group['is_source']).sum())} "
              f"spurious ({by_category})")
    return labels


def compare_on_labels(labels, label_scores, thresholds, peaks):
    """Before / after comparison on the auxiliary coadd, for the train and test labels separately.

    label_scores, thresholds and peaks are dicts keyed by "before" and "after" holding, for each model state, the
    labels' p_real, the threshold, and all peaks over the coadd. Returns {"train": table, "test": table}, each with one
    row per state: label_metrics plus the number of detections over the whole coadd.
    """
    tables = {}
    for split in ["train", "test"]:
        in_split = labels["split"].to_numpy() == split
        tables[split] = pd.DataFrame.from_dict({
            state: dict(**label_metrics(label_scores[state][in_split], labels["is_source"][in_split],
                                        labels["weight"][in_split], threshold),
                        detections_in_coadd=int((peaks[state]["p_real"] >= threshold).sum()))
            for state, threshold in thresholds.items()}, orient="index")
    return tables


def combine_mock_summaries(summaries):
    """{"before": summary, "after": summary} -> one table with the main columns of each side by side."""
    columns = dict(n_detections="detections", purity="purity", completeness_above_limit="complete_above",
                   completeness_below_limit="complete_below")
    return pd.concat({state: summary.set_index("coadd")[list(columns)].rename(columns=columns)
                      for state, summary in summaries.items()}, axis=1)


def write_report(out, label_tables, mock_table, details):
    """Save the before / after comparison in the new model folder (JSON, plus the mock table as CSV)."""
    (out / "mep_update_report.json").write_text(json.dumps(dict(
        **details, aux_labels={split: json.loads(t.to_json(orient="index")) for split, t in label_tables.items()},
        mock_test=json.loads(mock_table.to_json(orient="split"))), indent=2, default=str))
    mock_table.to_csv(out / "mep_update_mock_test_summary.csv")


def update_threshold_on_aux(model_dir, aux_coadd, label_paths, catalogue_dir, image_dir, suffix, cfg=None,
                            stem=CATALOGUE_STEM, test_name="test", mock_coadds=None, tile_cap=None):
    """Choose a new detection threshold from the auxiliary training labels and save the model with it in
    <model_dir>_<suffix> (weights unchanged). Prints the before / after comparison and returns the new folder.

    aux_coadd is the .npz image, label_paths the review JSON files and / or CSVs, and catalogue_dir / image_dir
    the mock data used for the mock test comparison (mock_coadds, default: the reference coadd).
    """
    model_dir = Path(model_dir)
    out = new_model_dir(model_dir, suffix)
    model, normalisation, model_config = load_model(model_dir, cfg)
    cfg = model_config["cfg"]
    old = json.loads((model_dir / ARTEFACTS["threshold"]).read_text())
    calib_peaks = pd.read_parquet(model_dir / ARTEFACTS["calib_peaks"])

    coadd = AuxCoadd(aux_coadd, cfg)
    labels = load_and_split_labels(label_paths, coadd, cfg)
    peaks = predict_aux_peaks(model, normalisation, model_config, coadd, calib_peaks)
    labels["p_real"] = label_p_real(labels, peaks, cfg["aux_match_radius_pix"])

    train = labels[labels["split"] == "train"]
    threshold, stats, status = choose_weighted_threshold(train["p_real"], train["is_source"], train["weight"],
                                                         cfg["aux_purity"], cfg["wilson_z"])
    print(f"\nNew p_real threshold {threshold:.5f} (was {old['threshold']:.5f}). On the auxiliary training labels: "
          f"weighted purity {stats['purity']:.4f}, lower bound {stats['purity_lower']:.4f} from n_eff "
          f"{stats['n_eff']:.1f} labels; goal aux_purity = {cfg['aux_purity']} ({status})")
    if status != "met":
        print("WARNING: no threshold reaches aux_purity on the auxiliary training labels; using the threshold with the "
              "highest lower bound on purity.")

    thresholds = dict(before=old["threshold"], after=threshold)
    label_tables = compare_on_labels(labels, {state: labels["p_real"].to_numpy() for state in thresholds},
                                     thresholds, {state: peaks for state in thresholds})
    print_table("Auxiliary training labels, before and after (recall and purity use the label weights)",
                label_tables["train"])
    print_table("Auxiliary test labels (held out), before and after", label_tables["test"])

    coadds = mock_coadds or [cfg["reference_coadd"]]
    store, mock_peaks = score_test(catalogue_dir, image_dir, model_dir, coadds, cfg, test_name, stem, tile_cap)
    mock_table = combine_mock_summaries({state: summarise(store, mock_peaks, value, store.cfg)[0]
                                         for state, value in thresholds.items()})
    print_table("Mock test coadds, before and after", mock_table)

    out.mkdir(parents=True)
    for artefact in ("weights", "normalisation", "model_config", "history", "calib_peaks"):
        if (model_dir / ARTEFACTS[artefact]).exists():
            shutil.copy2(model_dir / ARTEFACTS[artefact], out / ARTEFACTS[artefact])
    (out / ARTEFACTS["threshold"]).write_text(json.dumps(dict(
        threshold=threshold, target_purity=cfg["aux_purity"], status=status,
        reference_combo=old.get("reference_combo"), chosen_on="auxiliary training labels", aux_coadd=str(aux_coadd),
        aux_labels=[str(p) for p in label_paths], previous_threshold=old["threshold"], previous_model=str(model_dir),
        **stats), indent=2))
    labels.to_csv(out / "mep_aux_labels.csv", index=False)
    write_report(out, label_tables, mock_table,
                 dict(threshold=threshold, previous_threshold=old["threshold"], status=status))
    print(f"\nSaved the model with the new threshold -> {out} (the original in {model_dir} is unchanged)")
    return out


class AuxTrainingTiles:
    """Training examples cut from the auxiliary coadd, each centred loosely on a training label.

    Only the neighbourhood of each label is known, so the loss is restricted to it ("partial labels"): within
    label_radius_pix of a source label the network is taught a galaxy centre, within label_radius_pix of a spurious
    label it is taught background, and everywhere else the loss is zero. Pixels near a test label are always
    excluded so the test labels stay unseen. Tiles are rotated and flipped at random.
    """

    def __init__(self, coadd, labels, normalisation, cfg):
        self.coadd, self.normalisation, self.cfg = coadd, normalisation, cfg
        self.train = labels[labels["split"] == "train"].reset_index(drop=True)
        self.test_xy = labels.loc[labels["split"] == "test", ["x", "y"]].to_numpy(float)
        # Row numbers of source and spurious training labels; each tile picks one of the two kinds with equal odds.
        self.rows_by_kind = [np.flatnonzero(self.train["is_source"].to_numpy() == kind) for kind in (True, False)]
        self.rows_by_kind = [rows for rows in self.rows_by_kind if len(rows)]
        full = cfg["tile_size"] + 2 * cfg["tile_halo"]
        self.pixel_y, self.pixel_x = np.mgrid[0:full, 0:full]

    def near(self, xy, x0, y0):
        """(mask of tile pixels within label_radius_pix of any of the positions xy, the positions in tile pixels)."""
        halo, radius = self.cfg["tile_halo"], self.cfg["label_radius_pix"]
        full = self.cfg["tile_size"] + 2 * halo
        in_tile = xy - [x0 - halo, y0 - halo]
        in_tile = in_tile[(in_tile >= -radius).all(axis=1) & (in_tile < full + radius).all(axis=1)]
        mask = np.zeros((full, full), bool)
        for x, y in in_tile:
            mask |= (self.pixel_x - x) ** 2 + (self.pixel_y - y) ** 2 <= radius ** 2
        return mask, in_tile

    def targets(self, x0, y0):
        """Training targets for the tile whose corner (without halo) is at (x0, y0), in the layout of
        TargetMaker.tile_targets. The galaxy heatmap target stacks [heatmap, centre weight, loss weight]; its loss
        weight is update_aux_weight near training labels and 0 elsewhere. The other outputs get zero loss weight."""
        size, halo = self.cfg["tile_size"], self.cfg["tile_halo"]
        full = size + 2 * halo
        heatmap, centre_weight, loss_weight = (np.zeros((full, full), np.float32) for _ in range(3))
        loss_weight[self.near(self.train[["x", "y"]].to_numpy(float), x0, y0)[0]] = self.cfg["update_aux_weight"]
        loss_weight[self.near(self.test_xy, x0, y0)[0]] = 0.0
        inside_tile = np.zeros((full, full), bool)  # the halo and anything beyond the image edge get no loss
        inside_tile[halo:halo + min(size, self.coadd.ny - y0), halo:halo + min(size, self.coadd.nx - x0)] = True
        loss_weight[~inside_tile] = 0.0
        source_xy = self.train.loc[self.train["is_source"], ["x", "y"]].to_numpy(float)
        for x, y in self.near(source_xy, x0, y0)[1]:
            centre = paint_gaussian(heatmap, x, y, HEATMAP_SIGMA_PIX["galaxy"])
            if centre is not None:
                centre_weight[centre[1], centre[0]] = 1.0
        zeros = np.zeros((full, full), np.float32)
        return {"galaxy_heatmap": np.dstack([heatmap, centre_weight, loss_weight]),
                "centroid_offset": np.zeros((full, full, 3), np.float32),
                "source_structure": np.zeros((full, full, 5), np.float32),
                "clump_heatmap": np.dstack([zeros, zeros, zeros]), "tidal_heatmap": np.dstack([zeros, zeros, zeros])}

    def sample(self, rng):
        """One training example (network input planes, PSF stamps, targets): a tile containing a randomly chosen
        training label at a random position, randomly rotated and flipped."""
        size = self.cfg["tile_size"]
        rows = self.rows_by_kind[rng.integers(len(self.rows_by_kind))]
        label = self.train.iloc[int(rng.choice(rows))]
        x0 = int(np.clip(label["x"] - rng.uniform(0, size), 0, max(self.coadd.nx - size, 0)))
        y0 = int(np.clip(label["y"] - rng.uniform(0, size), 0, max(self.coadd.ny - size, 0)))
        planes = encode_planes(self.coadd.extract_halo(self.coadd.signal, x0, y0),
                               self.coadd.extract_halo(self.coadd.variance, x0, y0), self.normalisation)
        quarter_turns, flip = int(rng.integers(4)), bool(rng.integers(2))

        def rotate_and_flip(array):
            array = np.rot90(array, quarter_turns, axes=(0, 1))
            return np.ascontiguousarray(array[:, ::-1] if flip else array)

        return (rotate_and_flip(planes), rotate_and_flip(self.coadd.psf),
                {name: rotate_and_flip(value) for name, value in self.targets(x0, y0).items()})


class MixedBatches(keras.utils.PyDataset):
    """Training batches that are part auxiliary tiles (a fraction update_aux_fraction of each batch, from
    AuxTrainingTiles) and part mock training tiles (from a CoaddTileSequence), so the model keeps seeing mock data
    while it learns from the auxiliary labels."""

    def __init__(self, mock_tiles, aux_tiles, cfg, workers=1):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=16)
        self.mock_tiles, self.aux_tiles = mock_tiles, aux_tiles
        self.batch_size = int(cfg["batch_size"])
        self.n_aux = int(np.clip(round(cfg["update_aux_fraction"] * self.batch_size), 1, self.batch_size))
        self.steps, self.seed, self.epoch = int(cfg["update_steps_per_epoch"]), int(cfg["seed"]), 0

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        self.epoch += 1
        self.mock_tiles.on_epoch_end()

    def __getitem__(self, step):
        rng = np.random.default_rng([self.seed, self.epoch, step])
        examples = [self.aux_tiles.sample(rng) for _ in range(self.n_aux)]
        images, psfs = [e[0] for e in examples], [e[1] for e in examples]
        targets = {name: [e[2][name] for e in examples] for name in examples[0][2]}
        n_mock = self.batch_size - self.n_aux
        if n_mock:
            mock_inputs, mock_targets = self.mock_tiles[int(rng.integers(len(self.mock_tiles)))]
            images += list(mock_inputs["image_planes"][:n_mock])
            psfs += list(mock_inputs["psf_kernels"][:n_mock])
            for name in targets:
                targets[name] += list(mock_targets[name][:n_mock])
        return ({"image_planes": np.stack(images), "psf_kernels": np.stack(psfs)},
                {name: np.stack(value) for name, value in targets.items()})


def update_weights_on_aux(model_dir, aux_coadd, label_paths, catalogue_dir, image_dir, suffix, cfg=None,
                          stem=CATALOGUE_STEM, train_name="train", valid_name="valid", calib_name="calib",
                          test_name="test", mock_coadds=None, tile_cap=None):
    """Fine-tune the model on the auxiliary training labels mixed with mock training tiles, recalibrate it on the
    mock calib catalogue, and save it in <model_dir>_<suffix>. Prints the before / after comparison (auxiliary test
    labels and mock test coadds) and returns the new folder.

    Arguments are as for update_threshold_on_aux, plus the names of the mock train, valid and calib catalogues.
    """
    model_dir = Path(model_dir)
    out = new_model_dir(model_dir, suffix)
    user_cfg = dict(cfg or {})
    set_up_tensorflow({**CONFIG, **user_cfg}["seed"])
    model, normalisation, model_config = load_model(model_dir, user_cfg)
    cfg = model_config["cfg"]
    old_threshold = json.loads((model_dir / ARTEFACTS["threshold"]).read_text())["threshold"]

    coadd = AuxCoadd(aux_coadd, cfg)
    labels = load_and_split_labels(label_paths, coadd, cfg)
    peaks = dict(before=predict_aux_peaks(model, normalisation, model_config, coadd,
                                          pd.read_parquet(model_dir / ARTEFACTS["calib_peaks"])))

    train_store = CoaddStore(train_name, catalogue_dir, image_dir, cfg, stem)
    valid_store = CoaddStore(valid_name, catalogue_dir, image_dir, cfg, stem)
    target_maker = TargetMaker(train_store, cfg)
    target_maker.log_re_mean, target_maker.log_re_std = log_re_scaling(model_config)  # keep the model's size scale
    for store in (train_store, valid_store):
        target_maker.prepare(store)
    mock_tiles = CoaddTileSequence(train_store, None, normalisation, cfg, target_maker, shuffle=True,
                                   fresh_noise=True, resample=True, coadds_per_tile=cfg["train_coadds_per_tile"],
                                   seed=cfg["seed"])
    valid_index = valid_store.sample_index()
    valid_index = valid_index.sample(n=min(cfg["valid_samples"], len(valid_index)),
                                     random_state=cfg["seed"]).reset_index(drop=True)
    valid_tiles = CoaddTileSequence(valid_store, valid_index, normalisation, cfg, target_maker,
                                    workers=cfg["data_workers"])
    batches = MixedBatches(mock_tiles, AuxTrainingTiles(coadd, labels, normalisation, cfg), cfg, cfg["data_workers"])
    print(f"Fine-tuning: {cfg['update_epochs']} epochs of {len(batches)} batches, each {batches.n_aux} auxiliary + "
          f"{batches.batch_size - batches.n_aux} mock tiles, learning rate {cfg['update_learning_rate']}; validation "
          f"loss on {len(valid_index)} mock valid tiles")

    out.mkdir(parents=True)
    compile_unet(model, {**cfg, "learning_rate": cfg["update_learning_rate"]})
    loss_before = model.evaluate(valid_tiles, return_dict=True, verbose=0)["galaxy_heatmap_loss"]
    print(f"Mock valid galaxy-heatmap loss before fine-tuning: {loss_before:.5f}")
    model.fit(batches, validation_data=valid_tiles, epochs=cfg["update_epochs"],
              callbacks=[keras.callbacks.CSVLogger(out / "mep_update_history.csv")])
    model.save_weights(out / ARTEFACTS["weights"])
    shutil.copy2(model_dir / ARTEFACTS["normalisation"], out / ARTEFACTS["normalisation"])
    update_settings = ("update_learning_rate", "update_epochs", "update_steps_per_epoch", "update_aux_fraction",
                       "update_aux_weight", "label_radius_pix", "aux_test_fraction", "aux_block_pix", "seed")
    (out / ARTEFACTS["model_config"]).write_text(json.dumps(dict(
        {key: value for key, value in model_config.items() if key != "cfg"},
        cfg=json.loads((model_dir / ARTEFACTS["model_config"]).read_text())["cfg"], updated_from=str(model_dir),
        aux_coadd=str(aux_coadd), aux_labels=[str(p) for p in label_paths],
        update={key: cfg[key] for key in update_settings}, mock_valid_loss_before=loss_before), indent=2))

    print("\nCalibrating the fine-tuned model on the mock calib catalogue...")
    new_threshold = calibrate_unet(catalogue_dir, image_dir, out, user_cfg, calib_name, stem)
    new_model, _, new_config = load_model(out, user_cfg)
    peaks["after"] = predict_aux_peaks(new_model, normalisation, new_config, coadd,
                                       pd.read_parquet(out / ARTEFACTS["calib_peaks"]))

    thresholds = dict(before=old_threshold, after=new_threshold)
    label_scores = {state: label_p_real(labels, peaks[state], cfg["aux_match_radius_pix"]) for state in thresholds}
    labels["p_real_before"], labels["p_real_after"] = label_scores["before"], label_scores["after"]
    label_tables = compare_on_labels(labels, label_scores, thresholds, peaks)
    print_table("Auxiliary training labels, original model (before) and fine-tuned model (after); recall and purity "
                "use the label weights", label_tables["train"])
    print_table("Auxiliary test labels (held out), before and after", label_tables["test"])

    coadds = mock_coadds or [cfg["reference_coadd"]]
    mock_table = combine_mock_summaries({
        state: mock_test_summary(catalogue_dir, image_dir, folder, thresholds[state], coadds, test_name, stem,
                                 tile_cap, user_cfg)
        for state, folder in dict(before=model_dir, after=out).items()})
    print_table("Mock test coadds, before and after", mock_table)

    labels.to_csv(out / "mep_aux_labels.csv", index=False)
    write_report(out, label_tables, mock_table,
                 dict(threshold=new_threshold, previous_threshold=old_threshold, updated_from=str(model_dir)))
    print(f"\nSaved the fine-tuned model -> {out} (the original in {model_dir} is unchanged). To also choose its "
          "threshold from the auxiliary labels, run scripts/update_threshold_on_aux.py on it.")
    return out
