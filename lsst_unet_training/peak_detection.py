"""Run the trained U-Net over whole coadds, turn its detection map into a peak list, and match peaks to the truth.

Detections are the peaks of detection_heatmap (or, for models without it, galaxy_heatmap). Each peak also carries the
galaxy and star heatmaps' values at its position (galaxy_score, star_score) when the model has them, which say what
kind of source it is.
"""

import gc
import time

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.spatial import KDTree

from .coadd_data import encode_planes


def detection_map_name(outputs):
    """The output detections are taken from."""
    return "detection_heatmap" if "detection_heatmap" in outputs else "galaxy_heatmap"


def tile_peaks(predictions, batch_rows, store, cfg, log_re_scaling):
    """Peaks in one batch of predictions: 3x3 local maxima of the detection map inside each tile (not its halo),
    above min_peak_score, at most max_peaks_per_tile per tile (highest first). Each peak's position is refined by the
    predicted centroid offset, and its size read from the structure head."""
    size, halo = cfg["tile_size"], cfg["tile_halo"]
    log_re_mean, log_re_std = log_re_scaling
    detection = detection_map_name(predictions)
    kinds = [name for name in ("galaxy_heatmap", "star_heatmap") if name in predictions and name != detection]
    peaks = []
    for i, row in enumerate(batch_rows):
        heatmap = np.asarray(predictions[detection][i, ..., 0])
        kind_maps = {name: np.asarray(predictions[name][i, ..., 0]) for name in kinds}
        offset = np.asarray(predictions["centroid_offset"][i])
        structure = np.asarray(predictions["source_structure"][i])
        in_tile = np.zeros(heatmap.shape, bool)
        in_tile[halo:halo + min(size, store.ny - int(row.y0)), halo:halo + min(size, store.nx - int(row.x0))] = True
        local_max = heatmap == maximum_filter(heatmap, size=3, mode="nearest")
        py, px = np.nonzero(local_max & in_tile & (heatmap >= cfg["min_peak_score"]))
        if len(px) > cfg["max_peaks_per_tile"]:
            keep = np.argpartition(heatmap[py, px], -cfg["max_peaks_per_tile"])[-cfg["max_peaks_per_tile"]:]
            px, py = px[keep], py[keep]
        for x, y in zip(px, py):
            dx, dy = np.clip(offset[y, x], -0.5, 0.5)
            re_pix = np.expm1(structure[y, x][0] * log_re_std + log_re_mean)
            peaks.append(dict(split=store.name, combo=row.combo, tile_id=int(row.tile_id),
                              x=float(row.x0 + x - halo + dx), y=float(row.y0 + y - halo + dy),
                              raw_score=float(heatmap[y, x]), predicted_re_pix=float(max(re_pix, 0.0)),
                              **{name.replace("_heatmap", "_score"): float(m[y, x]) for name, m in kind_maps.items()}))
    return peaks


def predict_peaks(model, store, normalisation, cfg, log_re_scaling, coadds=None, tile_cap=None):
    """Raw peaks for every tile of the given coadds (default: all). Coadds are processed one at a time, each loaded
    into RAM once and streamed tile by tile; tile_cap limits the number of (coadd, tile) pairs for a quick look."""
    index = store.sample_index(coadds)
    if tile_cap is not None and len(index) > tile_cap:
        index = index.sample(n=int(tile_cap), random_state=cfg["seed"])
    index = index.sort_values("combo").reset_index(drop=True)
    keys = list(dict.fromkeys(index["combo"]))
    print(f"{store.name}: predicting {len(index):,} tiles across {len(keys)} coadds...", flush=True)
    peaks, done, start = [], 0, time.time()
    for key in keys:
        rows = list(index[index["combo"] == key].itertuples(index=False))
        signal, variance = store.full_coadd(key, cfg["seed"])
        psf = store.psf_kernels(key)
        for first in range(0, len(rows), cfg["infer_batch"]):
            batch = rows[first:first + cfg["infer_batch"]]
            images = np.stack([encode_planes(*store.halo_pair(signal, variance, int(r.x0), int(r.y0)), normalisation)
                               for r in batch])
            predictions = model.predict_on_batch({"image_planes": images,
                                                  "psf_kernels": np.repeat(psf[None], len(batch), axis=0)})
            if isinstance(predictions, (list, tuple)):
                predictions = dict(zip(model.output_names, predictions))
            peaks += tile_peaks(predictions, batch, store, cfg, log_re_scaling)
            done += len(batch)
        del signal, variance
        gc.collect()
        seconds_per_tile = (time.time() - start) / max(done, 1)
        print(f"  {store.name} {key}: {done}/{len(index)} tiles, {len(peaks):,} peaks, "
              f"ETA {seconds_per_tile * (len(index) - done) / 60:.1f} min", flush=True)
    peaks = pd.DataFrame(peaks)
    if len(peaks):
        peaks = peaks[peaks["x"].between(0, store.nx - 1) & peaks["y"].between(0, store.ny - 1)].reset_index(drop=True)
    return peaks


def match_peaks(store, peaks, cfg, kinds=("galaxy", "star")):
    """Label each peak real if it is the highest-scoring peak within match_radius_pix of a truth source of the given
    kinds (per coadd, each source matches at most one peak). Adds label_real, truth_kind ("galaxy" / "star"),
    truth_index (row in store.truth or store.stars) and match_distance_pix."""
    peaks = peaks.copy().reset_index(drop=True)
    peaks["label_real"], peaks["truth_kind"], peaks["truth_index"] = False, "", -1
    peaks["match_distance_pix"] = np.nan
    galaxy_rows = np.flatnonzero(store.truth_inside) if "galaxy" in kinds else np.empty(0, int)
    star_rows = np.flatnonzero(store.star_inside) if "star" in kinds else np.empty(0, int)
    truth_rows = np.r_[galaxy_rows, star_rows]
    truth_kind = np.array(["galaxy"] * len(galaxy_rows) + ["star"] * len(star_rows), dtype=object)
    if len(peaks) == 0 or len(truth_rows) == 0:
        return peaks
    tree = KDTree(np.column_stack([np.r_[store.truth_x[galaxy_rows], store.star_x[star_rows]],
                                   np.r_[store.truth_y[galaxy_rows], store.star_y[star_rows]]]))
    for _, block in peaks.groupby("combo"):
        distance, nearest = tree.query(block[["x", "y"]].to_numpy(float), k=1)
        close = np.isfinite(distance) & (distance <= cfg["match_radius_pix"])
        pairs = pd.DataFrame(dict(peak_index=block.index.to_numpy()[close],
                                  truth_local=np.asarray(nearest)[close].astype(np.int64),
                                  distance=distance[close], score=block["raw_score"].to_numpy(float)[close]))
        winners = pairs.sort_values(["score", "distance"], ascending=[False, True]).drop_duplicates("truth_local")
        peaks.loc[winners["peak_index"], "label_real"] = True
        peaks.loc[winners["peak_index"], "truth_index"] = truth_rows[winners["truth_local"].to_numpy()]
        peaks.loc[winners["peak_index"], "truth_kind"] = truth_kind[winners["truth_local"].to_numpy()]
        peaks.loc[winners["peak_index"], "match_distance_pix"] = winners["distance"].to_numpy()
    return peaks
