"""Update a trained model using labelled detections on an extra ("auxiliary") coadd. This could be from real data or
another mock.

The model is trained and calibrated on mock coadds. The auxiliary coadd is any other image with labelled detections:
a real coadd inspected by eye, a mock made with different settings, a reprocessed image. Two updates are offered:

update_threshold_on_aux
    Keep the network, change only the detection threshold. The new threshold is the lowest p_detection_centroid at
    which the labelled detections above it are pure enough (aux_purity, judged on the training half of the labels).
    If no threshold can be shown to reach aux_purity, the old threshold is kept.

update_weights_on_aux
    Keep training the network (fine-tuning) on a mix of auxiliary and mock tiles, at a low learning rate, then
    recalibrate it on the mock calib catalogue as usual. A model without some of the current heads is first given
    those that can be taught here (unet_model.transfer_weights: its detections are unchanged until it is trained) and
    switched to "no data" edge padding, which the fine-tuning teaches with artificial edges.

Both write a new model folder <model_dir>_<suffix> and never modify the original. Both print, before and after the
update, how many labelled sources and spurious detections are detected on the held-out half of the labels, how many
detections there are over the whole auxiliary coadd, and purity / completeness on the mock test coadds (to check the
update has not made the model worse on mocks).

Labels
------
A label is a pixel position on the auxiliary coadd marked either as a source (a genuine galaxy or star) or as
spurious (an artefact or noise). They are read from:
  * review JSON files (feedback_<category>.json, written by lsst_unet_detection's review tool), one per candidate
    category (e.g. "detected by the U-Net only"):
    {"category": ..., "n": number of candidates in the category,
     "reviewed": {id: {"x", "y", "decision", "how", "reason"}, ...},
     "additional_reviewed": {id: {"x", "y", "decision", "how", "reason", "category"}, ...},
     "missed": [{"x", "y"}, ...]}
    decision is "real" (a source), "spurious" or "unsure" (ignored). how is "random" for candidates that came up in
    the random review order and "selected" for ones the reviewer picked by clicking; additional_reviewed holds picked
    detections of other categories; missed lists sources the reviewer found that no detection caught. The optional
    reason says what a label is: REASONS below.
  * CSV files with columns x, y, label ("real" or "spurious") and optionally weight and reason; their rows count as
    random.

Only random labels are an unbiased sample, so only they are used for statistics (choosing the threshold, purity,
recall); selected and missed labels are used for fine-tuning only. A reviewer may inspect only a sample of a category
(e.g. 157 of 11,324 candidates) while another is inspected in full, so each random label gets weight n_candidates /
n_reviewed for its category, so that each category counts in proportion to its size. Other labels get weight 1.

Train / test split: within each category, the image is cut into square blocks of aux_block_pix pixels and a random
aux_test_fraction of the blocks is held out for testing, so every category has test labels and test labels are
spatially separated from the labels of their category that the update learns from. The split depends only on the
labels, the block size, the test fraction and the seed, so both commands hold out the same labels.

Each label is given the p_detection_centroid of the highest-scoring U-Net peak within aux_match_radius_pix of it, or
0 if there is no peak nearby, in which case it counts as undetected at every threshold.

Caveat: the labels only describe the candidates that were reviewed. Lowering the threshold also lets through
unreviewed peaks elsewhere in the image, so the reports also give the number of detections over the whole coadd.
"""

import json
import shutil
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

from .calibration import calibrator_for, nearest_band_set, read_threshold, wilson_lower
from .coadd_data import CoaddStore, encode_planes
from .masking import coverage
from .config import ARTEFACTS, BANDS, CATALOGUE_STEM, CONFIG
from .evaluation import calibrate_unet, score_test, summarise
from .peak_detection import predict_peaks
from .targets import TargetMaker, paint_gaussian
from .tile_sequence import CoaddTileSequence
from .training import load_model, log_re_scaling, new_model_dir, set_up_tensorflow, unteachable_heads
from .unet_model import compile_unet, transfer_weights

# Review decisions that become labels, mapped to "is a source". Anything else (e.g. "unsure") is ignored.
DECISIONS = dict(real=True, spurious=False)
DUPLICATE_RADIUS_PIX = 1.5  # labels closer together than this are taken to be the same object
# What a label's optional reason says it is. Spurious detections: on a diffraction spike, on the bridge between two
# sources that should have been two detections, on a star-forming region or tidal feature of a galaxy, a badly placed
# centre, or nothing visible at all. Real ones: a star. Reasons that name a phenomenon teach its map (or star_heatmap).
REASONS = ("spike", "bridge", "sfregion", "tidal", "bad_centroid", "hallucination", "star")
REASON_HEADS = dict(spike="spike_map", sfregion="sfregion_map", tidal="tidal_map", star="star_heatmap")
REASON_RADIUS_PIX = 2.0  # a reason's map is taught "yes" within this radius of the label
PEAK_COLUMN = "p_detection_centroid"


class AuxCoadd:
    """The auxiliary coadd, read from an .npz file with arrays signal and variance (band, y, x), psf_kernels
    (y, x, band) and bands (band names).

    The detector's inference code (peak_detection.predict_peaks) was written for CoaddStore, which serves the many
    mock coadds of one catalogue. This class provides the same methods (tile_grid, sample_index, full_coadd,
    psf_kernels, halo_pair) for a single coadd, so that inference runs on it unchanged. The whole image is kept in
    memory. Bands are reordered to the package's band order and the PSF stamps are resized to the stamp size the
    network expects. band_set names the bands that have data anywhere in it.
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
        has_data = coverage(self.variance, cfg).any(axis=(1, 2))
        self.band_set = "".join(band for band, present in zip(BANDS, has_data) if present)

    # Tiling helpers shared with CoaddStore; they only use self.cfg, self.nx, self.ny and self.coadds.
    extract_halo = CoaddStore.extract_halo
    outside_image = CoaddStore.outside_image
    mark_no_data = CoaddStore.mark_no_data
    halo_pair = CoaddStore.halo_pair
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


LABEL_COLUMNS = ["x", "y", "is_source", "weight", "category", "how", "reason"]


def label_table(entries, category, how=None, weight=1.0):
    """Labels (LABEL_COLUMNS) from review entries with x, y, decision and optionally how, reason and category."""
    table = pd.DataFrame(list(entries))
    if len(table) == 0:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    table = table[table["decision"].isin(DECISIONS)] if "decision" in table else table.assign(decision="real")
    column = lambda name, default: table[name].fillna(default).astype(str) if name in table else default
    return pd.DataFrame(dict(x=table["x"].to_numpy(float), y=table["y"].to_numpy(float),
                             is_source=table["decision"].map(DECISIONS).to_numpy(bool), weight=weight,
                             category=column("category", category), how=how or column("how", "random"),
                             reason=column("reason", "")))


def review_labels(path):
    """Labels from one review JSON file (see the module docstring)."""
    review = json.loads(Path(path).read_text())
    category = review.get("category", Path(path).stem)
    reviewed = list(review.get("reviewed", {}).values())
    labels = label_table(reviewed, category)
    n_random = sum(entry.get("how", "random") == "random" for entry in reviewed)  # unsure ones included
    if n_random:
        random = (labels["how"] == "random").to_numpy()
        labels.loc[random, "weight"] = max(int(review.get("n", n_random)), n_random) / n_random
    additional = label_table(review.get("additional_reviewed", {}).values(), category, how="selected")
    missed = label_table(review.get("missed", []), f"{category}_missed", how="missed")
    return pd.concat([frame for frame in (labels, additional, missed) if len(frame)] or [labels], ignore_index=True)


def csv_labels(path):
    """Labels from a CSV with columns x, y, label ("real" or "spurious") and optionally weight and reason."""
    table = pd.read_csv(path)
    label = table["label"].astype(str).str.strip().str.lower()
    keep = label.isin(DECISIONS)
    weight = table.loc[keep, "weight"].to_numpy(float) if "weight" in table else 1.0
    reason = table.loc[keep, "reason"].fillna("").astype(str).to_numpy() if "reason" in table else ""
    return pd.DataFrame(dict(x=table.loc[keep, "x"].to_numpy(float), y=table.loc[keep, "y"].to_numpy(float),
                             is_source=label[keep].map(DECISIONS).to_numpy(bool), weight=weight,
                             category=Path(path).stem, how="random", reason=reason))


def read_labels(paths, nx=None, ny=None):
    """All labels from a list of review JSON files and / or CSVs, as one table (columns LABEL_COLUMNS).

    Labels closer together than DUPLICATE_RADIUS_PIX are treated as one object: if they agree, the first is kept
    (a random label before a selected or missed one); if they disagree (one says source, another spurious), all are
    dropped. If the image size nx, ny is given, labels outside the image are dropped.
    """
    labels = pd.concat([review_labels(p) if Path(p).suffix == ".json" else csv_labels(p) for p in paths],
                       ignore_index=True)
    labels = labels.sort_values("how", key=lambda how: how.map(dict(random=0, selected=1, missed=2)).fillna(3),
                                kind="stable").reset_index(drop=True)
    labels = labels[np.isfinite(labels["x"]) & np.isfinite(labels["y"])].reset_index(drop=True)
    if nx is not None and ny is not None:
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
    """Add a column split ("train" or "test"): within each category, a random test_fraction of the block_pix x
    block_pix image blocks holding its labels is held out for testing (at least one block in each half when the
    category has two or more blocks)."""
    block_x, block_y = ((labels[axis] // block_pix).astype(int).astype(str) for axis in ("x", "y"))
    block = (block_x + "," + block_y).to_numpy()
    split = np.full(len(labels), "train", dtype=object)
    for number, category in enumerate(sorted(labels["category"].unique())):
        rows = np.flatnonzero(labels["category"].to_numpy() == category)
        blocks = np.array(sorted(set(block[rows])))
        if len(blocks) < 2:
            continue
        n_test = int(np.clip(round(test_fraction * len(blocks)), 1, len(blocks) - 1))
        test_blocks = set(np.random.default_rng([seed, number]).permutation(blocks)[:n_test])
        split[rows[np.isin(block[rows], list(test_blocks))]] = "test"
    return labels.assign(split=split)


def label_scores(labels, peaks, radius):
    """For each label, the p_detection_centroid of the best peak within radius pixels, or 0 if there is none."""
    scores = np.zeros(len(labels))
    if len(peaks) and len(labels):
        peak_scores = peaks[PEAK_COLUMN].to_numpy(float)
        nearby = KDTree(peaks[["x", "y"]].to_numpy(float)).query_ball_point(labels[["x", "y"]].to_numpy(float),
                                                                            r=radius)
        scores = np.array([peak_scores[found].max() if len(found) else 0.0 for found in nearby])
    return scores


def label_metrics(scores, is_source, weight, threshold):
    """How the labels fare at a threshold. A label is detected if its score is positive and at least the threshold.

    Returns counts (sources detected out of all sources, spurious detections out of all spurious labels) and, using
    the weights, recall (fraction of sources detected) and purity (fraction of detected labels that are sources).
    """
    scores, is_source, weight = np.asarray(scores, float), np.asarray(is_source, bool), np.asarray(weight, float)
    detected = (scores > 0) & (scores >= threshold)
    detected_weight = weight[detected].sum()
    return dict(threshold=float(threshold),
                sources_detected=f"{int((detected & is_source).sum())}/{int(is_source.sum())}",
                spurious_detected=f"{int((detected & ~is_source).sum())}/{int((~is_source).sum())}",
                recall=float(weight[detected & is_source].sum() / max(weight[is_source].sum(), 1e-12)),
                purity=float(weight[detected & is_source].sum() / detected_weight) if detected_weight > 0 else np.nan)


def choose_weighted_threshold(scores, is_source, weight, purity_goal, z):
    """The lowest threshold whose weighted purity is at least purity_goal with ~95% confidence.

    Every distinct positive score is tried as a threshold. For the labels at or above it, the weighted purity is
    computed, and its Wilson lower bound (z standard deviations) is taken with the effective number of labels
    n_eff = (sum of weights)^2 / (sum of squared weights), so that a few heavily weighted labels do not count as many.

    Returns (threshold, dict(purity, purity_lower, n_eff), status): status is "met" if some threshold reaches
    purity_goal; otherwise threshold is None (keep the current one) and the dict describes the best one found.
    """
    scores, is_source, weight = np.asarray(scores, float), np.asarray(is_source, bool), np.asarray(weight, float)
    candidates = []
    for threshold in np.unique(scores[scores > 0])[::-1]:  # highest first
        above = scores >= threshold
        purity = weight[above & is_source].sum() / weight[above].sum()
        n_eff = weight[above].sum() ** 2 / (weight[above] ** 2).sum()
        candidates.append((threshold, dict(purity=float(purity), n_eff=float(n_eff),
                                           purity_lower=float(wilson_lower(purity * n_eff, n_eff, z)))))
    if not candidates:
        raise RuntimeError("no random training label has a U-Net peak nearby, so no threshold can be chosen")
    meeting_goal = [i for i, (_, stats) in enumerate(candidates) if stats["purity_lower"] >= purity_goal]
    if meeting_goal:
        i = meeting_goal[-1]  # the last, i.e. lowest, threshold that meets the goal
        return float(candidates[i][0]), candidates[i][1], "met"
    best = int(np.argmax([stats["purity_lower"] for _, stats in candidates]))
    return None, dict(candidates[best][1], best_threshold=float(candidates[best][0])), "unmet"


def predict_aux_peaks(model, normalisation, model_config, coadd, calib_peaks):
    """Run the model over the whole auxiliary coadd: one row per peak, with p_detection_centroid from the model's
    mock calibration (calib_peaks) of the coadd's band set (or the calibrated set nearest to it)."""
    peaks = predict_peaks(model, coadd, normalisation, model_config["cfg"], log_re_scaling(model_config))
    calibrator = calibrator_for(calib_peaks, nearest_band_set(coadd.band_set, set(calib_peaks["band_set"])))
    peaks[PEAK_COLUMN] = calibrator.predict(peaks["raw_score"]) if len(peaks) else []
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
        random = group["how"] == "random"
        print(f"{split} labels: {int(group['is_source'].sum())} sources, {int((~group['is_source']).sum())} "
              f"spurious; {int(random.sum())} random, {int((~random).sum())} selected or missed ({by_category})")
    return labels


def compare_on_labels(labels, scores, thresholds, peaks):
    """Before / after comparison on the auxiliary coadd's random labels, for the train and test halves separately.

    scores, thresholds and peaks are dicts keyed by "before" and "after" holding, for each model state, the labels'
    scores, the threshold, and all peaks over the coadd. Returns {"train": table, "test": table}, each with one row
    per state: label_metrics plus the number of detections over the whole coadd.
    """
    tables = {}
    for split in ["train", "test"]:
        rows = (labels["split"].to_numpy() == split) & (labels["how"].to_numpy() == "random")
        tables[split] = pd.DataFrame.from_dict({
            state: dict(**label_metrics(scores[state][rows], labels["is_source"][rows], labels["weight"][rows],
                                        threshold),
                        detections_in_coadd=int((peaks[state][PEAK_COLUMN] >= threshold).sum()))
            for state, threshold in thresholds.items()}, orient="index")
    return tables


def combine_mock_summaries(summaries):
    """{"before": summary, "after": summary} -> one table with the main columns of each side by side."""
    columns = dict(n_detections="detections", purity="purity", completeness_above_limit="complete_above",
                   completeness_below_limit="complete_below", extended_completeness="extended",
                   star_completeness="stars")
    return pd.concat({state: summary.set_index("coadd")[[c for c in columns if c in summary]].rename(columns=columns)
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
    old_threshold = old["threshold"]
    labels = load_and_split_labels(label_paths, coadd, cfg)
    peaks = predict_aux_peaks(model, normalisation, model_config, coadd, calib_peaks)
    labels[PEAK_COLUMN] = label_scores(labels, peaks, cfg["aux_match_radius_pix"])

    train = labels[(labels["split"] == "train") & (labels["how"] == "random")]
    threshold, stats, status = choose_weighted_threshold(train[PEAK_COLUMN], train["is_source"], train["weight"],
                                                         cfg["aux_purity"], cfg["wilson_z"])
    if status == "met":
        print(f"\nNew threshold {threshold:.5f} (was {old_threshold:.5f}). On the random auxiliary training labels: "
              f"weighted purity {stats['purity']:.4f}, lower bound {stats['purity_lower']:.4f} from n_eff "
              f"{stats['n_eff']:.1f} labels; goal aux_purity = {cfg['aux_purity']}")
    else:
        threshold = float(old_threshold)
        print(f"\nNo threshold reaches aux_purity = {cfg['aux_purity']} on the random auxiliary training labels (best: "
              f"{stats['best_threshold']:.5f}, lower bound {stats['purity_lower']:.4f} from n_eff {stats['n_eff']:.1f}"
              f" labels), so the threshold stays at {threshold:.5f}. More random labels narrow the bound.")

    thresholds = dict(before=old_threshold, after=threshold)
    label_tables = compare_on_labels(labels, {state: labels[PEAK_COLUMN].to_numpy() for state in thresholds},
                                     thresholds, {state: peaks for state in thresholds})

    print_table("Random auxiliary training labels, before and after (recall and purity use the label weights)",
                label_tables["train"])
    print_table("Random auxiliary test labels (held out), before and after", label_tables["test"])

    coadds = mock_coadds or [cfg["reference_coadd"]]
    store, mock_peaks = score_test(catalogue_dir, image_dir, model_dir, coadds, cfg, test_name, stem, tile_cap)
    mock_table = combine_mock_summaries({state: summarise(store, mock_peaks, value, store.cfg)[0]
                                         for state, value in thresholds.items()})
    print_table("Mock test coadds, before and after", mock_table)

    out.mkdir(parents=True)
    for artefact in ("weights", "normalisation", "model_config", "history", "calib_peaks"):
        if (model_dir / ARTEFACTS[artefact]).exists():
            shutil.copy2(model_dir / ARTEFACTS[artefact], out / ARTEFACTS[artefact])
    old.update(threshold=threshold, status=status, purity_goal=cfg["aux_purity"], chosen_on="auxiliary training labels",
               aux_coadd=str(aux_coadd), aux_labels=[str(p) for p in label_paths], previous_threshold=old_threshold,
               previous_model=str(model_dir), **stats)
    (out / ARTEFACTS["threshold"]).write_text(json.dumps(old, indent=2))
    labels.to_csv(out / "mep_aux_labels.csv", index=False)
    write_report(out, label_tables, mock_table,
                 dict(threshold=threshold, previous_threshold=old_threshold, status=status))
    print(f"\nSaved the model with the new threshold -> {out} (the original in {model_dir} is unchanged)")
    return out


class AuxTrainingTiles:
    """Training examples cut from the auxiliary coadd, each holding a training label at a random position.

    Only the neighbourhood of each label is known, so the loss is restricted to it ("partial labels"): within
    label_radius_pix of a label, the detection map is taught a source centre (source labels) or background (spurious
    ones), with loss weight update_aux_weight, times update_miss_weight for sources the model missed; everywhere else
    the loss is zero. Labels with a reason (REASONS) also teach the head that reason names (REASON_HEADS): "yes"
    within REASON_RADIUS_PIX of the label, for example tidal_map on a detection that is a tidal feature; a spurious
    label tells galaxy_heatmap and star_heatmap there is no centre. Pixels near a test label are always excluded so
    the test labels stay unseen. Tiles are rotated and flipped at random.
    """

    def __init__(self, coadd, labels, normalisation, cfg, heads):
        self.coadd, self.normalisation, self.cfg, self.heads = coadd, normalisation, cfg, list(heads)
        self.train = labels[labels["split"] == "train"].reset_index(drop=True)
        self.test_xy = labels.loc[labels["split"] == "test", ["x", "y"]].to_numpy(float)
        # Row numbers of source and spurious training labels; each tile picks one of the two kinds with equal odds.
        self.rows_by_kind = [np.flatnonzero(self.train["is_source"].to_numpy() == kind) for kind in (True, False)]
        self.rows_by_kind = [rows for rows in self.rows_by_kind if len(rows)]
        full = cfg["tile_size"] + 2 * cfg["tile_halo"]
        self.pixel_y, self.pixel_x = np.mgrid[0:full, 0:full]

    def near(self, xy, x0, y0, radius=None):
        """(mask of tile pixels within radius (default label_radius_pix) of any of the positions xy, the positions in
        tile pixels, which of the rows of xy those are)."""
        halo, radius = self.cfg["tile_halo"], self.cfg["label_radius_pix"] if radius is None else radius
        full = self.cfg["tile_size"] + 2 * halo
        in_tile = np.asarray(xy, float).reshape(-1, 2) - [x0 - halo, y0 - halo]
        rows = np.flatnonzero((in_tile >= -radius).all(axis=1) & (in_tile < full + radius).all(axis=1))
        mask = np.zeros((full, full), bool)
        for x, y in in_tile[rows]:
            mask |= (self.pixel_x - x) ** 2 + (self.pixel_y - y) ** 2 <= radius ** 2
        return mask, in_tile[rows], rows

    def targets(self, x0, y0):
        """Training targets for the tile whose corner (without halo) is at (x0, y0), one per head, in the layout of
        TargetMaker.tile_targets: [target, positive weight, loss weight] for maps, [values, weight] for regressions."""
        cfg, train = self.cfg, self.train
        size, halo = cfg["tile_size"], cfg["tile_halo"]
        full = size + 2 * halo
        zeros = lambda *shape: np.zeros((full, full, *shape), np.float32)
        inside_tile = np.zeros((full, full), bool)  # the halo and anything beyond the image get no loss
        inside_tile[halo:halo + min(size, self.coadd.ny - y0), halo:halo + min(size, self.coadd.nx - x0)] = True
        allowed = inside_tile & ~self.near(self.test_xy, x0, y0)[0]

        def loss_weight(rows):
            """Loss weight near the given training labels (the highest of the labels a pixel is near)."""
            weight = np.zeros((full, full), np.float32)
            for row in rows:
                mask = self.near(train.loc[[row], ["x", "y"]].to_numpy(float), x0, y0)[0]
                weight[mask] = np.maximum(weight[mask], train.at[row, "loss_weight"])
            return weight * allowed

        def centres(rows):
            """Heatmap with a Gaussian at each label, and its centre pixels."""
            heatmap, positive = zeros(), zeros()
            for row in rows:
                x, y = train.at[row, "x"] - x0 + halo, train.at[row, "y"] - y0 + halo
                centre = paint_gaussian(heatmap, x, y, cfg["target_sigma_pix"])
                if centre is not None:
                    positive[centre[1], centre[0]] = 1.0
            return heatmap, positive

        _, _, here = self.near(train[["x", "y"]].to_numpy(float), x0, y0)
        sources = [row for row in here if train.at[row, "is_source"]]
        spurious = [row for row in here if not train.at[row, "is_source"]]
        stars = [row for row in sources if train.at[row, "reason"] == "star"]
        map_reasons = {head: reason for reason, head in REASON_HEADS.items() if head != "star_heatmap"}
        targets = {"centroid_offset": zeros(3), "source_structure": zeros(5)}
        for head in self.heads:
            target, positive, weight = zeros(), zeros(), zeros()
            if head == "detection_heatmap":
                target, positive = centres(sources)
                weight = loss_weight(here)
            elif head == "galaxy_heatmap":  # an untagged source may be a galaxy or a star, so it teaches nothing here
                weight = loss_weight(spurious + stars)
            elif head == "star_heatmap":
                target, positive = centres(stars)
                weight = loss_weight(spurious + stars)
            elif head in map_reasons:  # only labels with this map's reason say anything about it
                rows = [row for row in here if train.at[row, "reason"] == map_reasons[head]]
                if rows:
                    mask = self.near(train.loc[rows, ["x", "y"]].to_numpy(float), x0, y0, REASON_RADIUS_PIX)[0]
                    target = positive = mask.astype(np.float32)
                    weight = loss_weight(rows)
            targets[head] = np.dstack([target, positive, weight])
        return targets

    def sample(self, rng):
        """One training example (network input planes, PSF stamps, targets): a tile containing a randomly chosen
        training label at a random position, randomly rotated and flipped."""
        size = self.cfg["tile_size"]
        rows = self.rows_by_kind[rng.integers(len(self.rows_by_kind))]
        label = self.train.iloc[int(rng.choice(rows))]
        x0 = int(np.clip(label["x"] - rng.uniform(0, size), 0, max(self.coadd.nx - size, 0)))
        y0 = int(np.clip(label["y"] - rng.uniform(0, size), 0, max(self.coadd.ny - size, 0)))
        signal, variance = self.coadd.halo_pair(self.coadd.signal, self.coadd.variance, x0, y0)
        planes = encode_planes(signal, variance, self.normalisation)
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
    old_cfg = model_config["cfg"]
    old_threshold = read_threshold(model_dir)

    coadd = AuxCoadd(aux_coadd, old_cfg)
    labels = load_and_split_labels(label_paths, coadd, old_cfg)
    peaks = dict(before=predict_aux_peaks(model, normalisation, model_config, coadd,
                                          pd.read_parquet(model_dir / ARTEFACTS["calib_peaks"])))
    scores = dict(before=label_scores(labels, peaks["before"], old_cfg["aux_match_radius_pix"]))

    # The updated model gets "no data" edges and those of the current heads that something here can teach: a head
    # with no truth in the mocks and no labels would otherwise drift into an arbitrary feature of the detection head
    # instead of the map its name promises.
    train_store = CoaddStore(train_name, catalogue_dir, image_dir, old_cfg, stem)
    valid_store = CoaddStore(valid_name, catalogue_dir, image_dir, old_cfg, stem)
    taught_by_labels = {REASON_HEADS[reason] for reason in labels["reason"] if reason in REASON_HEADS}
    unteachable = set(unteachable_heads(train_store, CONFIG["heads"])) - taught_by_labels
    not_added = unteachable - set(old_cfg["heads"])
    heads = tuple(user_cfg.get("heads", [h for h in CONFIG["heads"] if h not in not_added]))
    cfg = {**old_cfg, "heads": heads, "edge_padding": user_cfg.get("edge_padding", CONFIG["edge_padding"])}
    if not_added:
        print(f"Not adding {', '.join(sorted(not_added))}: neither the mocks nor the labels hold their truth")
    if set(heads) != set(old_cfg["heads"]):
        print(f"Model heads {sorted(old_cfg['heads'])} -> {sorted(heads)}; detections are unchanged until it is "
              "trained")
        model = transfer_weights(model, cfg)
    frozen = sorted(unteachable & set(heads))  # heads it already has but nothing here can teach: kept as they are
    for layer in model.layers:
        if any(layer.name in (head, f"{head}_pre") for head in frozen):
            layer.trainable = False
    if frozen:
        print(f"Keeping {', '.join(frozen)} unchanged: nothing here holds their truth")
    for data in (coadd, train_store, valid_store):
        data.cfg = cfg

    missed = labels["is_source"].to_numpy() & ((labels["how"].to_numpy() == "missed")
                                               | (scores["before"] < old_threshold))
    labels["loss_weight"] = cfg["update_aux_weight"] * np.where(missed, cfg["update_miss_weight"], 1.0)
    print(f"{int(missed.sum())} source labels the model missed count {cfg['update_miss_weight']:g}x more")

    target_maker = TargetMaker(train_store, cfg)
    target_maker.log_re_mean, target_maker.log_re_std = log_re_scaling(model_config)  # keep the model's size scale
    for store in (train_store, valid_store):
        target_maker.prepare(store)
    mock_tiles = CoaddTileSequence(train_store, None, normalisation, cfg, target_maker, shuffle=True,
                                   fresh_noise=True, resample=True, coadds_per_tile=cfg["train_coadds_per_tile"],
                                   seed=cfg["seed"], edge_augment_fraction=cfg["edge_augment_fraction"])
    valid_index = valid_store.sample_index()
    valid_index = valid_index.sample(n=min(cfg["valid_samples"], len(valid_index)),
                                     random_state=cfg["seed"]).reset_index(drop=True)
    valid_tiles = CoaddTileSequence(valid_store, valid_index, normalisation, cfg, target_maker,
                                    workers=cfg["data_workers"])
    batches = MixedBatches(mock_tiles, AuxTrainingTiles(coadd, labels, normalisation, cfg, cfg["heads"]), cfg,
                           cfg["data_workers"])
    print(f"Fine-tuning: {cfg['update_epochs']} epochs of {len(batches)} batches, each {batches.n_aux} auxiliary + "
          f"{batches.batch_size - batches.n_aux} mock tiles, learning rate {cfg['update_learning_rate']}; validation "
          f"loss on {len(valid_index)} mock valid tiles")

    out.mkdir(parents=True)
    compile_unet(model, {**cfg, "learning_rate": cfg["update_learning_rate"]})
    monitored = "detection_heatmap_loss" if "detection_heatmap" in cfg["heads"] else "galaxy_heatmap_loss"
    # verbose=0 (silent) is valid; Keras's type hints only list the string options
    losses_before = model.evaluate(valid_tiles, return_dict=True, verbose=0)  # pyright: ignore[reportArgumentType]
    loss_before = losses_before[monitored]
    print(f"Mock valid {monitored.replace('_loss', '')} loss before fine-tuning: {loss_before:.5f}")
    model.fit(batches, validation_data=valid_tiles, epochs=cfg["update_epochs"],
              callbacks=[keras.callbacks.CSVLogger(out / "mep_update_history.csv")])
    model.save_weights(out / ARTEFACTS["weights"])
    shutil.copy2(model_dir / ARTEFACTS["normalisation"], out / ARTEFACTS["normalisation"])
    update_settings = ("update_learning_rate", "update_epochs", "update_steps_per_epoch", "update_aux_fraction",
                       "update_aux_weight", "update_miss_weight", "label_radius_pix", "aux_test_fraction",
                       "aux_block_pix", "seed")
    saved_cfg = {**json.loads((model_dir / ARTEFACTS["model_config"]).read_text())["cfg"],
                 "heads": list(cfg["heads"]), "edge_padding": cfg["edge_padding"]}
    (out / ARTEFACTS["model_config"]).write_text(json.dumps(dict(
        {key: value for key, value in model_config.items() if key != "cfg"}, cfg=saved_cfg,
        updated_from=str(model_dir), aux_coadd=str(aux_coadd), aux_labels=[str(p) for p in label_paths],
        update={key: cfg[key] for key in update_settings}, mock_valid_loss_before=loss_before), indent=2))

    print("\nCalibrating the fine-tuned model on the mock calib catalogue...")
    new_threshold = calibrate_unet(catalogue_dir, image_dir, out, user_cfg, calib_name, stem)["threshold"]
    new_model, _, new_config = load_model(out, user_cfg)
    peaks["after"] = predict_aux_peaks(new_model, normalisation, new_config, coadd,
                                       pd.read_parquet(out / ARTEFACTS["calib_peaks"]))
    scores["after"] = label_scores(labels, peaks["after"], cfg["aux_match_radius_pix"])

    thresholds = dict(before=old_threshold, after=new_threshold)
    labels[f"{PEAK_COLUMN}_before"], labels[f"{PEAK_COLUMN}_after"] = scores["before"], scores["after"]
    label_tables = compare_on_labels(labels, scores, thresholds, peaks)
    print_table("Random auxiliary training labels, original model (before) and fine-tuned model (after); recall and "
                "purity use the label weights", label_tables["train"])
    print_table("Random auxiliary test labels (held out), before and after", label_tables["test"])

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
