"""Run the trained U-Net over whole coadds, turn its galaxy heatmap into a peak list, and match peaks to the truth."""

import gc
import time

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.spatial import KDTree

from .coadd_data import encode_planes


def tile_peaks(predictions, batch_rows, store, cfg, log_re_scaling):
    """Peaks in one batch of predictions: 3x3 local maxima of the galaxy heatmap inside each tile (not its halo),
    above min_peak_score, at most max_peaks_per_tile per tile (highest first). Each peak's position is refined by the
    predicted centroid offset, and its size read from the structure head."""
    size, halo = cfg["tile_size"], cfg["tile_halo"]
    log_re_mean, log_re_std = log_re_scaling
    peaks = []
    for i, row in enumerate(batch_rows):
        heatmap = np.asarray(predictions["galaxy_heatmap"][i, ..., 0])
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
                              raw_score=float(heatmap[y, x]), predicted_re_pix=float(max(re_pix, 0.0))))
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
            images = np.stack([encode_planes(store.extract_halo(signal, int(r.x0), int(r.y0)),
                                             store.extract_halo(variance, int(r.x0), int(r.y0)), normalisation)
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


def match_peaks(store, peaks, cfg):
    """Label each peak real if it is the highest-scoring peak within match_radius_pix of a truth galaxy (per coadd,
    each galaxy matches at most one peak). Adds label_real, truth_index and match_distance_pix."""
    peaks = peaks.copy().reset_index(drop=True)
    peaks["label_real"], peaks["truth_index"], peaks["match_distance_pix"] = False, -1, np.nan
    truth_rows = np.flatnonzero(store.truth_inside)
    if len(peaks) == 0 or len(truth_rows) == 0:
        return peaks
    tree = KDTree(np.column_stack([store.truth_x[truth_rows], store.truth_y[truth_rows]]))
    for _, block in peaks.groupby("combo"):
        distance, nearest = tree.query(block[["x", "y"]].to_numpy(float), k=1)
        close = np.isfinite(distance) & (distance <= cfg["match_radius_pix"])
        pairs = pd.DataFrame(dict(peak_index=block.index.to_numpy()[close],
                                  truth_local=np.asarray(nearest)[close].astype(np.int64),
                                  distance=distance[close], score=block["raw_score"].to_numpy(float)[close]))
        winners = pairs.sort_values(["score", "distance"], ascending=[False, True]).drop_duplicates("truth_local")
        peaks.loc[winners["peak_index"], "label_real"] = True
        peaks.loc[winners["peak_index"], "truth_index"] = truth_rows[winners["truth_local"].to_numpy()]
        peaks.loc[winners["peak_index"], "match_distance_pix"] = winners["distance"].to_numpy()
    return peaks
