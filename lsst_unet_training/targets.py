"""Training targets painted onto each tile from the truth catalogue.

Every galaxy centre becomes a small Gaussian peak in the galaxy heatmap; at its centre pixel the network also learns
the sub-pixel offset and the galaxy's structure (log size, axis ratio, orientation). Clumps and tidal blobs get their
own heatmaps. Galaxies are weighted so rare populations count as much as common ones: the train catalogue is binned
by surface brightness, stellar mass, redshift and sSFR, and each galaxy is weighted by 1 / (count in its bin).
"""

import numpy as np
import pandas as pd

POPULATION_PROPERTIES = ["truth_mu_r", "truth_logM", "truth_z", "truth_logssfr"]
N_QUANTILES = 6
HEATMAP_SIGMA_PIX = dict(galaxy=1.5, clump=1.0, tidal=2.0)


def quantile_edges(values, n_quantiles=N_QUANTILES):
    """Bin edges at the quantiles of the finite values, open-ended at both ends."""
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < n_quantiles:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_quantiles + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def bin_codes(values, edges):
    """Bin index of each value (-1 for non-finite)."""
    values = np.asarray(values, float)
    codes = np.digitize(values, np.asarray(edges)[1:-1]).astype(np.int16)
    codes[~np.isfinite(values)] = -1
    return codes


def group_by_tile(tile_ids):
    """{tile id: positions in tile_ids with that id}."""
    ids = np.asarray(tile_ids)
    if len(ids) == 0:
        return {}
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_ids))]
    ends = np.r_[starts[1:], len(order)]
    return {int(sorted_ids[start]): order[start:end] for start, end in zip(starts, ends)}


def paint_gaussian(heatmap, cx, cy, sigma):
    """Max-combine a unit-peak Gaussian at (cx, cy) into heatmap and set its nearest pixel to exactly 1. Returns that
    pixel (x, y), or None if it lies outside the heatmap."""
    radius = int(np.ceil(3.0 * sigma))
    xc, yc = int(np.rint(cx)), int(np.rint(cy))
    x0, x1 = max(0, xc - radius), min(heatmap.shape[1], xc + radius + 1)
    y0, y1 = max(0, yc - radius), min(heatmap.shape[0], yc + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    gaussian = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma ** 2)
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)
    if 0 <= xc < heatmap.shape[1] and 0 <= yc < heatmap.shape[0]:
        heatmap[yc, xc] = 1.0
        return xc, yc
    return None


class TargetMaker:
    """Population weights and structure scaling learned from the train catalogue, applied to any catalogue."""

    def __init__(self, train_store, cfg):
        self.cfg = cfg
        inside = train_store.truth_inside
        self.edges = {name: quantile_edges(getattr(train_store, name)[inside]) for name in POPULATION_PROPERTIES}
        self.train_counts = pd.Series(self.population_keys(train_store)[inside]).value_counts()
        log_re = np.log1p(np.maximum(train_store.truth_re_arcsec / cfg["pixscale"], 0.0))
        fit = inside & np.isfinite(log_re)
        self.log_re_mean = float(np.nanmean(log_re[fit]))  # structure target 0 = log(1 + Re/pixel) scaled
        self.log_re_std = max(float(np.nanstd(log_re[fit])), 1e-3)

    def population_keys(self, store):
        """'mu|logM|z|sSFR' bin label of every truth galaxy."""
        codes = [bin_codes(getattr(store, name), self.edges[name]) for name in POPULATION_PROPERTIES]
        return np.array(["|".join(map(str, row)) for row in zip(*codes)], dtype=object)

    def prepare(self, store):
        """Attach per-galaxy weights, structure targets and per-tile groupings of galaxies, clumps and tidal blobs."""
        size = self.cfg["tile_size"]
        weight = pd.Series(self.population_keys(store)).map(1.0 / self.train_counts).fillna(0.0).to_numpy(float)
        positive = weight[store.truth_inside & (weight > 0)]
        if len(positive):
            weight = weight / np.mean(positive)
        store.population_weight = np.clip(weight, 1.0 / 8.0, 8.0)

        log_re = np.log1p(np.maximum(store.truth_re_arcsec / self.cfg["pixscale"], 0.0))
        axis_ratio = np.clip(1.0 - np.nan_to_num(store.truth_ellipticity, nan=0.3), 0.05, 1.0)
        angle = np.deg2rad(np.nan_to_num(store.truth_pa, nan=0.0))
        store.structure_target = np.column_stack([
            (np.nan_to_num(log_re, nan=self.log_re_mean) - self.log_re_mean) / self.log_re_std,
            axis_ratio, np.sin(2 * angle), np.cos(2 * angle)]).astype(np.float32)

        def tile_of(x, y):
            return np.floor(x / size).astype(np.int64) + store.n_tile_x * np.floor(y / size).astype(np.int64)

        inside = np.flatnonzero(store.truth_inside)
        store.truth_by_tile = {tile: inside[rows] for tile, rows in
                               group_by_tile(tile_of(store.truth_x[inside], store.truth_y[inside])).items()}

        def blobs_by_tile(xy):
            if len(xy) == 0:
                return {}
            rows = np.flatnonzero((xy[:, 0] >= 0) & (xy[:, 0] < store.nx) & (xy[:, 1] >= 0) & (xy[:, 1] < store.ny))
            return {tile: rows[r] for tile, r in group_by_tile(tile_of(xy[rows, 0], xy[rows, 1])).items()}

        store.clump_by_tile, store.tidal_by_tile = blobs_by_tile(store.clump_xy), blobs_by_tile(store.tidal_xy)
        return store

    def tile_targets(self, store, tile_id, x0, y0):
        """{output name: target array} for one tile. Heatmap targets stack [heatmap, centre weight, valid mask];
        regression targets stack [values, weight], with weight set only at galaxy centre pixels."""
        size, halo = self.cfg["tile_size"], self.cfg["tile_halo"]
        full = size + 2 * halo
        valid = np.zeros((full, full), np.float32)  # 1 on the tile itself, 0 on the halo and beyond the image
        valid[halo:halo + min(size, store.ny - int(y0)), halo:halo + min(size, store.nx - int(x0))] = 1.0

        galaxy, galaxy_weight = np.zeros((full, full), np.float32), np.zeros((full, full), np.float32)
        offset, structure = np.zeros((full, full, 2), np.float32), np.zeros((full, full, 4), np.float32)
        offset_weight, structure_weight = np.zeros((full, full), np.float32), np.zeros((full, full), np.float32)
        clump, clump_weight = np.zeros((full, full), np.float32), np.zeros((full, full), np.float32)
        tidal, tidal_weight = np.zeros((full, full), np.float32), np.zeros((full, full), np.float32)

        for index in store.truth_by_tile.get(int(tile_id), np.empty(0, int)):
            x, y = store.truth_x[index] - x0 + halo, store.truth_y[index] - y0 + halo
            centre = paint_gaussian(galaxy, x, y, HEATMAP_SIGMA_PIX["galaxy"])
            if centre is None:
                continue
            cx, cy = centre
            weight = float(store.population_weight[index])
            if weight >= galaxy_weight[cy, cx]:  # two galaxies on one pixel: the rarer population wins
                galaxy_weight[cy, cx] = offset_weight[cy, cx] = structure_weight[cy, cx] = weight
                offset[cy, cx] = [x - cx, y - cy]
                structure[cy, cx] = store.structure_target[index]

        blob_heads = [(self.cfg["clump_head"], store.clump_by_tile, store.clump_xy, clump, clump_weight, "clump"),
                      (self.cfg["tidal_head"], store.tidal_by_tile, store.tidal_xy, tidal, tidal_weight, "tidal")]
        for enabled, by_tile, xy, heatmap, weights, kind in blob_heads:
            if not enabled:
                continue
            for index in by_tile.get(int(tile_id), np.empty(0, int)):
                centre = paint_gaussian(heatmap, xy[index, 0] - x0 + halo, xy[index, 1] - y0 + halo,
                                        HEATMAP_SIGMA_PIX[kind])
                if centre is not None:
                    weights[centre[1], centre[0]] = 1.0

        return {"galaxy_heatmap": np.dstack([galaxy, galaxy_weight, valid]),
                "centroid_offset": np.dstack([offset, offset_weight]),
                "source_structure": np.dstack([structure, structure_weight]),
                "clump_heatmap": np.dstack([clump, clump_weight, valid]),
                "tidal_heatmap": np.dstack([tidal, tidal_weight, valid])}
