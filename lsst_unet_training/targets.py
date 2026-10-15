"""Training targets painted onto each tile from the truth.

Centre heads (*_heatmap) are taught a Gaussian peak at each object's centre, whose centre pixel is the positive:
  galaxy_heatmap      galaxies; the Gaussian's width grows with the galaxy's size (target_sigma_per_re x Re, between
                      target_sigma_pix and target_sigma_max_pix), so an extended galaxy has a broad centre
  star_heatmap        stars (narrow Gaussians)
  detection_heatmap   both: the centres of all sources
At a galaxy's centre pixel the network also learns the sub-pixel offset and the galaxy's structure (log size, axis
ratio, orientation).

Map heads (*_map) are taught where a phenomenon's own light is detectable in the coadd. The mock generator saves the
light of star-forming regions (sfregion_map), tidal features (tidal_map) and diffraction spikes (spike_map) alone; a
pixel is on the phenomenon where that light, smoothed by a Gaussian of truth_map_filter_pix and combined over bands
by inverse variance, has S/N >= truth_map_snr in that coadd's noise. The optional sfregion_heatmap and tidal_heatmap
heads are taught centre peaks at the catalogued star-forming region and tidal blob positions.

A source is a centre only as far as it can be seen in that coadd, judged by its size-matched S/N (band_snr, combined
over the bands): at S/N >= detectable_snr it is a positive; below invisible_snr it is background, like empty sky;
in between it is neither, with no loss around it (whether it shows depends on the noise). In a tile with bands
missing (training the band adapter) a positive must be detectable in the bands present; a source that is not, but
would be visible with all bands, is neither. The maps' truth there only counts the light and noise of the bands
present. Where a head's truth is unknown (mocks made without stars, or without the phenomenon's light saved) its
loss weight is zero, so nothing wrong is learned.

Positives are weighted (cfg["weighting"]):
  balanced   the galaxy positives of the train catalogue in each coadd fall in cells of (class, S/N bin)
             (source_classes, weight_snr_edges); each cell is raised to the count of the largest class in its S/N bin
             (weight = largest count / own count, between 1 and max_population_weight), so at every S/N each class
             weighs as much as the largest. A source's weight depends on the coadd, through its S/N and whether it is
             resolved; stars keep weight 1. Tiles are drawn in proportion to their positives' total weight (in the
             reference coadd), between 1/4 and max_tile_oversampling times the average, and each draw's loss is scaled
             by the inverse (stratified sampling). With footprint_negatives, the pixels around an up-weighted galaxy
             count more too (see TargetMaker.tile_targets).
  rarity     a galaxy's weight is the product of its rarity in population (1 / count in its bin of surface
             brightness, stellar mass, redshift and sSFR) and in appearance (1 / count in its bin of r magnitude and
             size), each first scaled to average 1; a star's is its rarity in r magnitude. The bins are quantiles of
             the train catalogue. Weights average 1 and are kept between 1/8 and max_population_weight. A tile's
             weight, how much more often training draws it, is the weight of its most heavily weighted source, between
             1 and max_tile_oversampling.

The extended_heatmap head is taught the centres of the positives that are resolved in that coadd (Re at least its
r-band PSF FWHM), every resolved (class, S/N bin) cell carrying the same total weight, so every resolved class counts
the same at each S/N whatever the weighting of the other heads.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .coadd_data import FWHM_TO_SIGMA
from .config import ADDED_POPULATIONS, BANDS, HUBBLE_CLASSES

POPULATION_PROPERTIES = ["truth_mu_r", "truth_logM", "truth_z", "truth_logssfr"]
N_QUANTILES = 6
N_APPEARANCE_QUANTILES = 8
BLOB_SIGMA_PIX = dict(sfregion=1.0, tidal=2.0)  # widths of the sfregion_heatmap / tidal_heatmap centre peaks
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


def truth_map(light, noise_sigma, cfg, bands_present=None):
    """(H, W) 0 / 1 map of where a phenomenon's (band, H, W) noise-free light is detectable in noise of noise_sigma
    per band: smoothed by a Gaussian of truth_map_filter_pix and combined over bands by inverse variance, its S/N is
    at least truth_map_snr. Bands not in bands_present (a boolean per band; default all) do not count."""
    sigma, inverse_variance = cfg["truth_map_filter_pix"], 1.0 / np.asarray(noise_sigma, float) ** 2
    if bands_present is not None:
        inverse_variance = inverse_variance * np.asarray(bands_present, float)
    combined = gaussian_filter(np.tensordot(inverse_variance, light, axes=1), sigma)
    noise = np.sqrt(inverse_variance.sum() / (4.0 * np.pi * sigma ** 2))  # std of the smoothed, combined noise
    return (combined / noise >= cfg["truth_map_snr"]).astype(np.float32)


def band_snr(flux, re_pix, psf_sigma_pix, noise_sigma):
    """(n_source, n_band) S/N of sources of the given flux (n, n_band) and half-light radius (n, pixels) in noise
    of noise_sigma per band: flux over the noise in an aperture of the source's PSF-convolved size."""
    size_sq = psf_sigma_pix[None, :] ** 2 + (np.nan_to_num(np.asarray(re_pix, float), nan=0.0)[:, None] / 1.1774) ** 2
    return flux / (noise_sigma[None, :] * np.sqrt(4.0 * np.pi * size_sq))


def combined_snr(store, key, flux, re_pix, cfg):
    """(n,) size-matched S/N (band_snr) of sources of the given flux (n, n_band) and half-light radius (n, pixels) in
    one coadd, combined over the bands."""
    psf_sigma = np.array([store.coadd_settings(key)[1][band] for band in BANDS]) * FWHM_TO_SIGMA / cfg["pixscale"]
    return np.sqrt(np.nansum(band_snr(flux, re_pix, psf_sigma, store.noise_sigma(key)) ** 2, axis=1))


def source_classes(store, fwhm_r):
    """Weighting class of each of the store's galaxies in a coadd of r-band PSF FWHM fwhm_r (arcsec): its added
    population (ADDED_POPULATIONS) if it belongs to one, else "unresolved" if its Re is below the FWHM, else its Hubble
    class (HUBBLE_CLASSES), else "resolved, other"."""
    re_arcsec = np.nan_to_num(store.truth_re_arcsec, nan=0.0)
    resolved = re_arcsec >= fwhm_r
    classes = np.where(resolved, "resolved, other", "unresolved").astype(object)
    hubble_type = pd.Series(store.truth_hubble_type, dtype=str)
    for name, pattern in HUBBLE_CLASSES.items():
        classes[resolved & hubble_type.str.fullmatch(pattern).to_numpy(bool)] = name
    for population, name in ADDED_POPULATIONS.items():
        classes[store.truth_lsb_population == population] = name
    return classes


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
        if cfg["weighting"] not in ("balanced", "rarity"):
            raise ValueError(f"weighting {cfg['weighting']!r}: use 'balanced' or 'rarity'")
        # {coadd: weight of each (class, S/N bin) cell}: balanced weighting, and the extended_heatmap head's
        self.cell_weights, self.extended_cell_weights = {}, {}
        if "extended_heatmap" in cfg["heads"]:  # the resolved positives' cells only, in each coadd
            for key in train_store.coadds:
                cells = pd.Series(self.cells(train_store, key)[0][inside], dtype=object)
                cells = cells.loc[(cells != "") & ~cells.str.startswith("unresolved|")]
                counts = cells.value_counts()
                self.extended_cell_weights[key] = counts.sum() / (len(counts) * counts)
        if cfg["weighting"] == "balanced":  # galaxy classes only: each raised to its S/N bin's largest class
            for key in train_store.coadds:
                cells = pd.Series(self.cells(train_store, key)[0][inside], dtype=object)
                counts = cells.loc[cells != ""].value_counts()
                snr_bin = counts.index.to_series().str.rsplit("|", n=1).str[1]
                self.cell_weights[key] = counts.groupby(snr_bin).transform("max") / counts

    def cells(self, store, key):
        """('class|S/N bin' of each galaxy, the same for each star) in one coadd; "" for sources below
        detectable_snr, which are never positives."""
        cfg = self.cfg
        edges = np.array([cfg["detectable_snr"], *cfg["weight_snr_edges"]])
        cell_lists = []
        for classes, flux, re_pix in (
                (source_classes(store, store.coadd_settings(key)[1]["r"]), store.truth_flux,
                 store.truth_re_arcsec / cfg["pixscale"]),
                (np.full(len(store.stars), "star", object), store.star_flux, np.zeros(len(store.stars)))):
            snr_bin = np.digitize(combined_snr(store, key, flux, re_pix, cfg), edges)  # 0: below detectable_snr
            cell = np.array([f"{c}|{b}" for c, b in zip(classes, snr_bin)], dtype=object)
            cell[snr_bin == 0] = ""
            cell_lists.append(cell)
        return cell_lists[0], cell_lists[1]

    def extended_weights(self, store, key):
        """Weight of each of the store's galaxies in the extended_heatmap target of one coadd: the train catalogue's
        resolved-cell weights there; 0 for galaxies that are unresolved or below detectable_snr. Computed once per
        store and coadd."""
        cached = store.extended_weights.get(key)
        if cached is None:
            cap, table = self.cfg["max_population_weight"], self.extended_cell_weights[key]
            cell = self.cells(store, key)[0]
            resolved = (cell != "") & ~pd.Series(cell, dtype=object).str.startswith("unresolved|").to_numpy(bool)
            mapped = pd.Series(cell, dtype=object).map(table).fillna(cap).to_numpy(float)
            cached = np.where(resolved, np.clip(mapped, MIN_WEIGHT, cap), 0.0)
            store.extended_weights[key] = cached
        return cached

    def weights(self, store, key):
        """(galaxy weights, star weights) of the store's sources in one coadd. Balanced galaxy weights are the train
        catalogue's cell weights in that coadd (a cell the train catalogue lacks counts as the rarest), stars weigh 1;
        computed once per store and coadd."""
        if self.cfg["weighting"] == "rarity":
            return store.population_weight, store.star_weight
        cached = store.coadd_weights.get(key)
        if cached is None:
            cap = self.cfg["max_population_weight"]
            table = self.cell_weights[key]
            galaxy_cells, star_cells = self.cells(store, key)
            galaxy = np.where(galaxy_cells == "", 0.0, np.clip(pd.Series(galaxy_cells, dtype=object).map(table)
                                                               .fillna(cap).to_numpy(float), 1.0, cap))
            star = np.where(star_cells == "", 0.0, 1.0)  # stars are not a science case: natural weight
            cached = store.coadd_weights[key] = (galaxy, star)
        return cached

    def population_keys(self, store):
        return bin_keys([getattr(store, name) for name in POPULATION_PROPERTIES], self.population_edges)

    def appearance_keys(self, store):
        return bin_keys(appearance_columns(store), self.appearance_edges)

    def star_keys(self, store):
        return bin_keys([store.star_mag_r], self.star_edges)

    def prepare(self, store):
        """Attach per-source weights, target widths, structure targets, per-tile groupings of galaxies, stars,
        star-forming regions and tidal blobs, and tile weights."""
        cfg, size, inside = self.cfg, self.cfg["tile_size"], store.truth_inside
        if cfg["weighting"] == "rarity":
            weight = (scaled_to_mean_one(rarity(self.population_keys(store), self.population_counts), inside)
                      * scaled_to_mean_one(rarity(self.appearance_keys(store), self.appearance_counts), inside))
            store.population_weight = np.clip(scaled_to_mean_one(weight, inside), MIN_WEIGHT,
                                              cfg["max_population_weight"])
            star_weight = scaled_to_mean_one(rarity(self.star_keys(store), self.star_counts), store.star_inside)
            store.star_weight = np.clip(star_weight, MIN_WEIGHT, cfg["max_population_weight"])
        else:  # balanced: per coadd (weights); the reference coadd's decide which tiles are drawn more often
            store.coadd_weights = {}
            store.population_weight, store.star_weight = self.weights(store, cfg["reference_coadd"])

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
        for name, xy in (("sfregion_by_tile", store.sfregion_xy), ("tidal_by_tile", store.tidal_xy)):
            if len(xy) == 0:
                setattr(store, name, {})
                continue
            setattr(store, name, by_tile(xy[:, 0], xy[:, 1], (xy[:, 0] >= 0) & (xy[:, 0] < store.nx)
                                         & (xy[:, 1] >= 0) & (xy[:, 1] < store.ny)))

        sources = ((store.truth_by_tile, store.population_weight), (store.star_by_tile, store.star_weight))
        if cfg["weighting"] == "balanced":
            # Stratified sampling: tiles are drawn in proportion to the total weight of their positives (reference
            # coadd), between 1/4 and max_tile_oversampling times the average, and each draw's loss is scaled by the
            # inverse, so the expected loss is the class-balanced one while rare classes come up more often.
            total = np.zeros(len(store.tile_grid()))
            for groups, weights in sources:
                for tile, rows in groups.items():
                    total[tile] += float(weights[rows].sum())
            share = np.clip(total / max(total.mean(), 1e-12), 0.25, cfg["max_tile_oversampling"])
            store.tile_weight, store.tile_loss_scale = share, share.mean() / share
            return store
        tile_weight = np.ones(len(store.tile_grid()))
        for groups, weights in sources:
            for tile, rows in groups.items():
                tile_weight[tile] = max(tile_weight[tile], float(weights[rows].max()))
        store.tile_weight = np.clip(tile_weight, 1.0, cfg["max_tile_oversampling"])
        return store

    def visibility(self, store, key, rows, x, y, re_pix, flux, coverage):
        """(positive, hidden): which sources (positions x, y in tile pixels) are positives, S/N >= detectable_snr in the
        bands with data at their centre (coverage: (n_band, H, W) bool; None: all bands), and which are neither
        positive nor background (no loss around them): not positives, but S/N >= invisible_snr with all bands. The rest
        are background."""
        if len(rows) == 0:
            return np.zeros(0, bool), np.zeros(0, bool)
        psf_sigma = np.array([store.coadd_settings(key)[1][band] for band in BANDS]) * FWHM_TO_SIGMA
        snr_sq = band_snr(flux[rows], re_pix, psf_sigma / self.cfg["pixscale"], store.noise_sigma(key)) ** 2
        all_bands = np.sqrt(np.nansum(snr_sq, axis=1))
        if coverage is None:
            bands_left = all_bands
        else:
            xi = np.clip(np.rint(x).astype(int), 0, coverage.shape[2] - 1)
            yi = np.clip(np.rint(y).astype(int), 0, coverage.shape[1] - 1)
            bands_left = np.sqrt(np.nansum(snr_sq * coverage[:, yi, xi].T, axis=1))
        positive = bands_left >= self.cfg["detectable_snr"]
        return positive, ~positive & (all_bands >= self.cfg["invisible_snr"])

    def tile_targets(self, store, tile_id, x0, y0, key, heads=None, coverage=None):
        """{output name: target array} for one tile of one coadd, for the heads in heads (default cfg["heads"]) plus
        centroid_offset and source_structure. Heatmap and map targets stack [target, positive weight, loss weight];
        regression targets stack [values, weight], with weight set only at galaxy centre pixels. coverage ((n_band,
        H, W) bool, where each band has data) is given for tiles with bands missing (see the module docstring)."""
        cfg, heads = self.cfg, set(heads or self.cfg["heads"])
        size, halo = cfg["tile_size"], cfg["tile_halo"]
        full = size + 2 * halo
        valid = np.zeros((full, full), np.float32)  # 1 on the tile itself, 0 on the halo and beyond the image
        valid[halo:halo + min(size, store.ny - int(y0)), halo:halo + min(size, store.nx - int(x0))] = 1.0
        zeros = lambda *shape: np.zeros((full, full, *shape), np.float32)
        yy, xx = np.mgrid[0:full, 0:full]
        centre_valid = valid.copy()  # valid, except around sources the bands present cannot show

        def hide(x, y, sigma):
            centre_valid[(xx - x) ** 2 + (yy - y) ** 2 <= (3.0 * sigma) ** 2] = 0.0

        galaxy_weights, star_weights = self.weights(store, key)
        extended_weights = self.extended_weights(store, key) if "extended_heatmap" in heads else None
        extended, extended_weight = zeros(), zeros()
        galaxy, galaxy_weight = zeros(), zeros()
        offset, structure, centre_weight = zeros(2), zeros(4), zeros()
        rows = store.truth_by_tile.get(int(tile_id), np.empty(0, int))
        positive, hidden = self.visibility(store, key, rows, store.truth_x[rows] - x0 + halo,
                                           store.truth_y[rows] - y0 + halo,
                                           store.truth_re_arcsec[rows] / cfg["pixscale"], store.truth_flux, coverage)
        rows_galaxies, positive_galaxy = rows, dict(zip(rows.tolist(), positive.tolist()))
        for index, is_positive, is_hidden in zip(rows, positive, hidden):
            x, y = store.truth_x[index] - x0 + halo, store.truth_y[index] - y0 + halo
            if is_hidden:
                hide(x, y, float(store.target_sigma[index]))
            if not is_positive:
                continue
            centre = paint_gaussian(galaxy, x, y, float(store.target_sigma[index]))
            if centre is None:
                continue
            cx, cy = centre
            weight = float(galaxy_weights[index])
            if weight >= galaxy_weight[cy, cx]:  # two galaxies on one pixel: the rarer one wins
                galaxy_weight[cy, cx] = centre_weight[cy, cx] = weight
                offset[cy, cx] = [x - cx, y - cy]
                structure[cy, cx] = store.structure_target[index]
            if extended_weights is not None and extended_weights[index] > 0:
                paint_gaussian(extended, x, y, float(store.target_sigma[index]))
                extended_weight[cy, cx] = max(extended_weight[cy, cx], float(extended_weights[index]))

        star, star_weight = zeros(), zeros()
        rows = store.star_by_tile.get(int(tile_id), np.empty(0, int))
        positive, hidden = self.visibility(store, key, rows, store.star_x[rows] - x0 + halo,
                                           store.star_y[rows] - y0 + halo, np.zeros(len(rows)), store.star_flux,
                                           coverage)
        for index, is_positive, is_hidden in zip(rows, positive, hidden):
            if is_hidden:
                hide(store.star_x[index] - x0 + halo, store.star_y[index] - y0 + halo, cfg["target_sigma_pix"])
            if not is_positive:
                continue
            centre = paint_gaussian(star, store.star_x[index] - x0 + halo, store.star_y[index] - y0 + halo,
                                    cfg["target_sigma_pix"])
            if centre is not None:
                star_weight[centre[1], centre[0]] = max(star_weight[centre[1], centre[0]], star_weights[index])

        if cfg["footprint_negatives"]:
            emphasis = np.ones((full, full), np.float32)
            # Every pixel within footprint_re x Re of a galaxy weighted w > 1 counts e = w^footprint_power times in
            # the centre heads' loss (its loss weight is multiplied by e); a positive there counts max(its own weight,
            # e). With footprint_power 1, inside the footprint the balance of centres against substructure (star-
            # forming knots, the disc) is the natural one while the footprint as a whole, the boosted galaxy included,
            # counts w times against empty sky; with power p < 1 the galaxy's centre outweighs its own substructure
            # w^(1 - p) times.
            for index in rows_galaxies:
                w = float(galaxy_weights[index])
                if w > 1.0 and positive_galaxy[index]:
                    x, y = store.truth_x[index] - x0 + halo, store.truth_y[index] - y0 + halo
                    r = max(3.0 * float(store.target_sigma[index]),
                            cfg["footprint_re"] * float(np.nan_to_num(store.truth_re_arcsec[index])) / cfg["pixscale"])
                    disc = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
                    emphasis[disc] = np.maximum(emphasis[disc], w ** cfg["footprint_power"])
            centre_valid *= emphasis
            for weight in (galaxy_weight, star_weight, extended_weight):
                weight[:] = np.where(weight > 0, np.maximum(weight, emphasis) / emphasis, 0.0)

        targets = {"galaxy_heatmap": np.dstack([galaxy, galaxy_weight, centre_valid]),
                   "centroid_offset": np.dstack([offset, centre_weight]),
                   "source_structure": np.dstack([structure, centre_weight])}
        star_valid = centre_valid if store.has_stars else zeros()
        if "star_heatmap" in heads:
            targets["star_heatmap"] = np.dstack([star, star_weight, star_valid])
        if "detection_heatmap" in heads:
            targets["detection_heatmap"] = np.dstack([np.maximum(galaxy, star),
                                                      np.maximum(galaxy_weight, star_weight), centre_valid])
        if "extended_heatmap" in heads:
            targets["extended_heatmap"] = np.dstack([extended, extended_weight, centre_valid])
        for head, component in MAP_COMPONENTS.items():
            if head in heads:
                light = store.component_tile(component, key, x0, y0)
                if light is None:
                    targets[head] = np.dstack([zeros(), zeros(), zeros()])
                else:
                    present = None
                    if coverage is not None:
                        light, present = light * coverage, coverage.any(axis=(1, 2))
                    mask = truth_map(light, store.noise_sigma(key), cfg, present) if light.any() else zeros()
                    targets[head] = np.dstack([mask, mask, valid])
        for head, groups, xy, kind in (("sfregion_heatmap", store.sfregion_by_tile, store.sfregion_xy, "sfregion"),
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
