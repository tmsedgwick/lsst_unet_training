"""Training targets painted onto each tile from the truth.

Centre heads (*_heatmap) are taught a Gaussian peak at each object's centre, whose centre pixel is the positive:
  galaxy_heatmap      galaxies; the Gaussian's width grows with the galaxy's size (target_sigma_per_re x Re, between
                      target_sigma_pix and target_sigma_max_pix), so an extended galaxy has a broad centre
  star_heatmap        stars (narrow Gaussians)
  detection_heatmap   both: the centres of all sources
At a galaxy's centre pixel the network also learns the sub-pixel offset and the galaxy's structure (log size, axis
ratio, orientation).

Map heads (*_map) are taught where a phenomenon's own light is detectable in the coadd. The mock generator saves the
light of star-forming clumps (sfregion_map), tidal features (tidal_map) and diffraction spikes (spike_map) alone; a
pixel is on the phenomenon where that light, smoothed by a Gaussian of truth_map_filter_pix and combined over bands
by inverse variance, has S/N >= truth_map_snr in that coadd's noise. Models made before the maps had clump_heatmap
and tidal_heatmap: centre heatmaps of the catalogued clump and tidal blob positions.

Where a head's truth is unknown (mocks made without stars, or without the phenomenon's light saved) its loss weight
is zero, so nothing wrong is learned.

Rare sources count more. A galaxy's weight is the product of its rarity in population (1 / count in its bin of
surface brightness, stellar mass, redshift and sSFR) and in appearance (1 / count in its bin of r magnitude and
size), each first scaled to average 1; a star's is its rarity in r magnitude. Weights are scaled to average 1 and
kept between 1/8 and max_population_weight. A tile's weight, how much more often training draws it, is the weight of
its rarest source, between 1 and max_tile_oversampling. The bins are quantiles of the train catalogue.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

POPULATION_PROPERTIES = ["truth_mu_r", "truth_logM", "truth_z", "truth_logssfr"]
N_QUANTILES = 6
N_APPEARANCE_QUANTILES = 8
BLOB_SIGMA_PIX = dict(clump=1.0, tidal=2.0)  # widths of the clump_heatmap / tidal_heatmap centre peaks
# Each map head and the light it maps (the components coadd_data reads).
MAP_COMPONENTS = dict(sfregion_map="sfregions", tidal_map="tidal", spike_map="spikes")
MIN_WEIGHT = 1.0 / 8.0


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


def bin_keys(columns, edges):
    """'a|b|...' bin label of each row, from per-column values and edges."""
    codes = [bin_codes(values, column_edges) for values, column_edges in zip(columns, edges)]
    return np.array(["|".join(map(str, row)) for row in zip(*codes)], dtype=object) if codes else np.empty(0, object)


def rarity(keys, counts):
    """1 / (train count in each key's bin), 0 for bins the train catalogue never had."""
    return pd.Series(keys, dtype=object).map(1.0 / counts).fillna(0.0).to_numpy(float)


def scaled_to_mean_one(weight, selection):
    positive = weight[selection & (weight > 0)]
    return weight / np.mean(positive) if len(positive) else weight


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


def truth_map(light, noise_sigma, cfg):
    """(H, W) 0 / 1 map of where a phenomenon's (band, H, W) noise-free light is detectable in noise of noise_sigma
    per band: smoothed by a Gaussian of truth_map_filter_pix and combined over bands by inverse variance, its S/N is
    at least truth_map_snr."""
    sigma, inverse_variance = cfg["truth_map_filter_pix"], 1.0 / np.asarray(noise_sigma, float) ** 2
    combined = gaussian_filter(np.tensordot(inverse_variance, light, axes=1), sigma)
    noise = np.sqrt(inverse_variance.sum() / (4.0 * np.pi * sigma ** 2))  # std of the smoothed, combined noise
    return (combined / noise >= cfg["truth_map_snr"]).astype(np.float32)


def appearance_columns(store):
    """r magnitude and log size (pixels) of the store's galaxies."""
    return [store.truth_mag_r, np.log10(np.maximum(store.truth_re_arcsec, 1e-3))]


class TargetMaker:
    """Weights, target widths and structure scaling learned from the train catalogue, applied to any catalogue."""

    def __init__(self, train_store, cfg):
        self.cfg = cfg
        inside = train_store.truth_inside
        self.population_edges = [quantile_edges(getattr(train_store, name)[inside]) for name in POPULATION_PROPERTIES]
        self.appearance_edges = [quantile_edges(values[inside], N_APPEARANCE_QUANTILES)
                                 for values in appearance_columns(train_store)]
        self.population_counts = pd.Series(self.population_keys(train_store)[inside]).value_counts()
        self.appearance_counts = pd.Series(self.appearance_keys(train_store)[inside]).value_counts()
        stars_inside = train_store.star_inside
        self.star_edges = [quantile_edges(train_store.star_mag_r[stars_inside], N_APPEARANCE_QUANTILES)]
        self.star_counts = pd.Series(self.star_keys(train_store)[stars_inside]).value_counts()
        log_re = np.log1p(np.maximum(train_store.truth_re_arcsec / cfg["pixscale"], 0.0))
        fit = inside & np.isfinite(log_re)
        self.log_re_mean = float(np.nanmean(log_re[fit]))  # structure target 0 = log(1 + Re/pixel) scaled
        self.log_re_std = max(float(np.nanstd(log_re[fit])), 1e-3)

    def population_keys(self, store):
        return bin_keys([getattr(store, name) for name in POPULATION_PROPERTIES], self.population_edges)

    def appearance_keys(self, store):
        return bin_keys(appearance_columns(store), self.appearance_edges)

    def star_keys(self, store):
        return bin_keys([store.star_mag_r], self.star_edges)

    def prepare(self, store):
        """Attach per-source weights, target widths, structure targets, per-tile groupings of galaxies, stars, clumps
        and tidal blobs, and tile weights."""
        cfg, size, inside = self.cfg, self.cfg["tile_size"], store.truth_inside
        weight = (scaled_to_mean_one(rarity(self.population_keys(store), self.population_counts), inside)
                  * scaled_to_mean_one(rarity(self.appearance_keys(store), self.appearance_counts), inside))
        store.population_weight = np.clip(scaled_to_mean_one(weight, inside), MIN_WEIGHT, cfg["max_population_weight"])
        star_weight = scaled_to_mean_one(rarity(self.star_keys(store), self.star_counts), store.star_inside)
        store.star_weight = np.clip(star_weight, MIN_WEIGHT, cfg["max_population_weight"])

        re_pix = store.truth_re_arcsec / cfg["pixscale"]
        store.target_sigma = np.clip(np.nan_to_num(cfg["target_sigma_per_re"] * re_pix, nan=0.0),
                                     cfg["target_sigma_pix"], cfg["target_sigma_max_pix"])
        log_re = np.log1p(np.maximum(re_pix, 0.0))
        axis_ratio = np.clip(1.0 - np.nan_to_num(store.truth_ellipticity, nan=0.3), 0.05, 1.0)
        angle = np.deg2rad(np.nan_to_num(store.truth_pa, nan=0.0))
        store.structure_target = np.column_stack([
            (np.nan_to_num(log_re, nan=self.log_re_mean) - self.log_re_mean) / self.log_re_std,
            axis_ratio, np.sin(2 * angle), np.cos(2 * angle)]).astype(np.float32)

        def tile_of(x, y):
            return np.floor(x / size).astype(np.int64) + store.n_tile_x * np.floor(y / size).astype(np.int64)

        def by_tile(x, y, selection):
            rows = np.flatnonzero(selection)
            return {tile: rows[r] for tile, r in group_by_tile(tile_of(x[rows], y[rows])).items()}

        store.truth_by_tile = by_tile(store.truth_x, store.truth_y, inside)
        store.star_by_tile = by_tile(store.star_x, store.star_y, store.star_inside)
        for name, xy in (("clump_by_tile", store.clump_xy), ("tidal_by_tile", store.tidal_xy)):
            if len(xy) == 0:
                setattr(store, name, {})
                continue
            setattr(store, name, by_tile(xy[:, 0], xy[:, 1], (xy[:, 0] >= 0) & (xy[:, 0] < store.nx)
                                         & (xy[:, 1] >= 0) & (xy[:, 1] < store.ny)))

        tile_weight = np.ones(len(store.tile_grid()))
        sources = ((store.truth_by_tile, store.population_weight), (store.star_by_tile, store.star_weight))
        for groups, weights in sources:
            for tile, rows in groups.items():
                tile_weight[tile] = max(tile_weight[tile], float(weights[rows].max()))
        store.tile_weight = np.clip(tile_weight, 1.0, cfg["max_tile_oversampling"])
        return store

    def tile_targets(self, store, tile_id, x0, y0, key, heads=None):
        """{output name: target array} for one tile of one coadd, for the heads in heads (default cfg["heads"]) plus
        centroid_offset and source_structure. Heatmap and map targets stack [target, positive weight, loss weight];
        regression targets stack [values, weight], with weight set only at galaxy centre pixels."""
        cfg, heads = self.cfg, set(heads or self.cfg["heads"])
        size, halo = cfg["tile_size"], cfg["tile_halo"]
        full = size + 2 * halo
        valid = np.zeros((full, full), np.float32)  # 1 on the tile itself, 0 on the halo and beyond the image
        valid[halo:halo + min(size, store.ny - int(y0)), halo:halo + min(size, store.nx - int(x0))] = 1.0
        zeros = lambda *shape: np.zeros((full, full, *shape), np.float32)

        galaxy, galaxy_weight = zeros(), zeros()
        offset, structure, centre_weight = zeros(2), zeros(4), zeros()
        for index in store.truth_by_tile.get(int(tile_id), np.empty(0, int)):
            x, y = store.truth_x[index] - x0 + halo, store.truth_y[index] - y0 + halo
            centre = paint_gaussian(galaxy, x, y, float(store.target_sigma[index]))
            if centre is None:
                continue
            cx, cy = centre
            weight = float(store.population_weight[index])
            if weight >= galaxy_weight[cy, cx]:  # two galaxies on one pixel: the rarer one wins
                galaxy_weight[cy, cx] = centre_weight[cy, cx] = weight
                offset[cy, cx] = [x - cx, y - cy]
                structure[cy, cx] = store.structure_target[index]

        star, star_weight = zeros(), zeros()
        for index in store.star_by_tile.get(int(tile_id), np.empty(0, int)):
            centre = paint_gaussian(star, store.star_x[index] - x0 + halo, store.star_y[index] - y0 + halo,
                                    cfg["target_sigma_pix"])
            if centre is not None:
                star_weight[centre[1], centre[0]] = max(star_weight[centre[1], centre[0]], store.star_weight[index])

        targets = {"galaxy_heatmap": np.dstack([galaxy, galaxy_weight, valid]),
                   "centroid_offset": np.dstack([offset, centre_weight]),
                   "source_structure": np.dstack([structure, centre_weight])}
        star_valid = valid if store.has_stars else zeros()
        if "star_heatmap" in heads:
            targets["star_heatmap"] = np.dstack([star, star_weight, star_valid])
        if "detection_heatmap" in heads:
            targets["detection_heatmap"] = np.dstack([np.maximum(galaxy, star),
                                                      np.maximum(galaxy_weight, star_weight), valid])
        for head, component in MAP_COMPONENTS.items():
            if head in heads:
                light = store.component_tile(component, key, x0, y0)
                if light is None:
                    targets[head] = np.dstack([zeros(), zeros(), zeros()])
                else:
                    mask = truth_map(light, store.noise_sigma(key), cfg) if light.any() else zeros()
                    targets[head] = np.dstack([mask, mask, valid])
        for head, groups, xy, kind in (("clump_heatmap", store.clump_by_tile, store.clump_xy, "clump"),
                                       ("tidal_heatmap", store.tidal_by_tile, store.tidal_xy, "tidal")):
            if head in heads:
                blob, blob_weight = zeros(), zeros()
                for index in groups.get(int(tile_id), np.empty(0, int)):
                    centre = paint_gaussian(blob, xy[index, 0] - x0 + halo, xy[index, 1] - y0 + halo,
                                            BLOB_SIGMA_PIX[kind])
                    if centre is not None:
                        blob_weight[centre[1], centre[0]] = 1.0
                targets[head] = np.dstack([blob, blob_weight, valid])
        return targets
