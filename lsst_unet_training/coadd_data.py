"""Read one mock catalogue's images and truth, and serve coadd tiles for any (depth, seeing) combination.

Inputs are the outputs of mock_lsst_image_generation: <image_dir>/<name>/ (base render, base_meta.json,
coadd_manifest.json and any saved coadds) and <catalogue_dir>/<stem>_<name>.csv (+ _clumps, _tidal). Saved coadds
are read from disk; the others (train/valid by default) are rebuilt tile by tile from the base render, by broadening
it to the coadd's PSF and adding noise for its number of visits, with the image generator's PSF and noise model.

Every tile is a tile_size square plus tile_halo pixels of context on each side. The network input per band is
[arcsinh(S/N / 3), normalised log variance], and the PSF enters separately as one unit-sum Gaussian stamp per band.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .config import BANDS, CATALOGUE_STEM

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def as_float(values):
    """Float array of a column, with anything non-numeric as NaN."""
    return np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)


def sky_sigma(band, n_visit, cfg):
    """Per-pixel sky noise of an n_visit coadd: the 10-year value (from the 5-sigma point-source depth over a
    nominal-PSF aperture) scaled by sqrt(10-year visits / n_visit)."""
    n_pix_psf = 1.13 * (cfg["nominal_fwhm"][band] / cfg["pixscale"]) ** 2
    sigma_10yr = 10 ** (-0.4 * (cfg["depth_10yr"][band] - cfg["zeropoint"])) / (5 * np.sqrt(n_pix_psf))
    return sigma_10yr * np.sqrt(cfg["visits_10yr"][band] / n_visit)


def point_source_depth(band, n_visit, cfg):
    """5-sigma point-source depth (AB) of an n_visit coadd at the nominal PSF."""
    return cfg["depth_10yr"][band] - 1.25 * np.log10(cfg["visits_10yr"][band] / max(int(n_visit), 1))


def broaden(image, psf_fwhm, base_fwhm, cfg):
    """Broaden a base-PSF (band, y, x) image to the per-band PSF FWHM (Gaussian, quadrature difference)."""
    broadened = np.empty_like(image)
    for i, band in enumerate(BANDS):
        sigma_pix = np.sqrt(max(float(psf_fwhm[band]) ** 2 - base_fwhm ** 2, 0.0)) * FWHM_TO_SIGMA / cfg["pixscale"]
        broadened[i] = image[i] if sigma_pix <= 0 else gaussian_filter(image[i], sigma=sigma_pix, mode="reflect",
                                                                       truncate=5.0)
    return broadened


def add_noise(image, n_visit, rng, cfg):
    """(signal, variance): sky Gaussian noise plus source Poisson noise averaged over n_visit visits."""
    signal, variance = image.copy(), np.empty_like(image)
    for i, band in enumerate(BANDS):
        visits = int(n_visit[band])
        sky = sky_sigma(band, visits, cfg)
        source = np.maximum(image[i], 0.0)
        signal[i] += rng.poisson(visits * source).astype(np.float32) / visits - source
        signal[i] += rng.normal(0.0, sky, image[i].shape).astype(np.float32)
        variance[i] = sky ** 2 + source / visits
    return signal.astype(np.float32), variance.astype(np.float32)


def gaussian_psf_kernel(fwhm_arcsec, cfg):
    """Unit-sum Gaussian PSF stamp; stands in for an LSST psf.computeKernelImage()."""
    sigma_pix = float(fwhm_arcsec) * FWHM_TO_SIGMA / cfg["pixscale"]
    half = cfg["psf_stamp"] // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_pix ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


def encode_planes(signal, variance, normalisation):
    """(band, H, W) signal and variance -> (H, W, 2 * n_band) network input: per band arcsinh(S/N / 3) and the
    normalised log variance, both clipped to +-8."""
    channels = []
    for i in range(len(BANDS)):
        var = np.maximum(np.nan_to_num(variance[i], nan=1e-12), 1e-12)
        sig = np.nan_to_num(signal[i], nan=0.0, posinf=0.0, neginf=0.0)
        snr = np.arcsinh((sig / np.sqrt(var)) / 3.0)
        log_var = (np.log(var) - normalisation["logvar_centre"][i]) / normalisation["logvar_scale"][i]
        channels += [np.clip(snr, -8, 8), np.clip(log_var, -8, 8)]
    return np.stack(channels, axis=-1).astype(np.float32)


class CoaddStore:
    """One catalogue's base render, coadd list and truth, with tiles served on demand."""

    def __init__(self, name, catalogue_dir, image_dir, cfg, stem=CATALOGUE_STEM):
        self.name, self.cfg = name, cfg
        self.dir = Path(image_dir) / name
        self.meta = json.loads((self.dir / "base_meta.json").read_text())
        self.manifest = json.loads((self.dir / "coadd_manifest.json").read_text())
        self.origin = tuple(int(v) for v in self.meta["origin"])  # frame pixel at image[:, 0, 0]
        _, self.ny, self.nx = (int(v) for v in self.meta["shape"])
        self.base_fwhm = float(self.meta["base_fwhm"])
        self.saved = bool(self.manifest["materialised"])  # coadds on disk, or rebuilt on the fly
        self.coadds = list(self.manifest["combos"])
        self.base_image = np.load(self.dir / "base_clean_signal.npy", mmap_mode="r")
        self._saved_coadds = {}
        self._load_truth(Path(catalogue_dir), stem)
        # Filled by targets.TargetMaker.prepare before training.
        self.population_weight = np.ones(len(self.truth))
        self.structure_target = np.zeros((len(self.truth), 4), np.float32)
        self.n_tile_x = int(np.ceil(self.nx / cfg["tile_size"]))
        self.truth_by_tile, self.clump_by_tile, self.tidal_by_tile = {}, {}, {}

    def _load_truth(self, catalogue_dir, stem):
        """Truth galaxies (and clump / tidal blob positions) in this image's pixel coordinates."""
        x0, y0 = self.origin
        truth = pd.read_csv(catalogue_dir / f"{stem}_{self.name}.csv").dropna(subset=["x_pix", "y_pix"])
        self.truth = truth.reset_index(drop=True)
        self.truth_x = self.truth["x_pix"].to_numpy(float) - x0
        self.truth_y = self.truth["y_pix"].to_numpy(float) - y0
        self.truth_inside = ((self.truth_x >= 0) & (self.truth_x < self.nx)
                             & (self.truth_y >= 0) & (self.truth_y < self.ny))
        column = lambda name, default=np.nan: (as_float(self.truth[name])
                                               if name in self.truth else np.full(len(self.truth), default))
        self.truth_mu_r, self.truth_mag_r = column("sb_r_total"), column("mag_r_total")
        self.truth_logM, self.truth_logssfr, self.truth_z = column("logM"), column("logsSFR"), column("z")
        self.truth_re_arcsec = column("re_total_arcsec")
        self.truth_ellipticity, self.truth_pa = column("ellipticity_total", 0.3), column("pa_deg", 0.0)
        self.clump_xy = self._blob_positions(catalogue_dir / f"{stem}_{self.name}_clumps.csv", "x_pix_clump",
                                             "y_pix_clump")
        self.tidal_xy = self._blob_positions(catalogue_dir / f"{stem}_{self.name}_tidal.csv", "x_pix_tidal",
                                             "y_pix_tidal")

    def _blob_positions(self, path, x_column, y_column):
        """(n, 2) finite pixel positions from a clump / tidal table, or empty if it is missing."""
        try:
            table = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return np.empty((0, 2), float)
        if x_column not in table or y_column not in table:
            return np.empty((0, 2), float)
        xy = np.column_stack([as_float(table[x_column]) - self.origin[0],
                              as_float(table[y_column]) - self.origin[1]])
        return xy[np.isfinite(xy).all(axis=1)]

    def coadd_settings(self, key):
        """(manifest entry, PSF FWHM per band, visits per band) of one coadd."""
        info = self.manifest["combos"][key]
        return (info, {band: float(info["fwhm_arcsec"][band]) for band in BANDS},
                {band: int(info["n_visit"][band]) for band in BANDS})

    def psf_kernels(self, key):
        """(psf_stamp, psf_stamp, n_band) PSF stamps of one coadd."""
        _, psf_fwhm, _ = self.coadd_settings(key)
        return np.stack([gaussian_psf_kernel(psf_fwhm[band], self.cfg) for band in BANDS], axis=-1)

    def extract_halo(self, cube, x0, y0):
        """The tile at (x0, y0) with its halo; beyond the image edge it is filled by reflection."""
        size, halo = self.cfg["tile_size"], self.cfg["tile_halo"]
        xs, ys, xe, ye = x0 - halo, y0 - halo, x0 + size + halo, y0 + size + halo
        sx0, sx1, sy0, sy1 = max(0, xs), min(self.nx, xe), max(0, ys), min(self.ny, ye)
        patch = np.asarray(cube[:, sy0:sy1, sx0:sx1], np.float32)
        pad = ((0, 0), (sy0 - ys, ye - sy1), (sx0 - xs, xe - sx1))
        if any(p for pair in pad for p in pair):
            patch = np.pad(patch, pad, mode="reflect" if min(patch.shape[1:]) > 1 else "edge")
        return patch

    def coadd_tile(self, key, x0, y0, rng=None):
        """(signal, variance) of one tile of one coadd. Saved coadds are read from disk; others are rebuilt from the
        base render. With rng=None the noise is seeded by (catalogue, coadd, tile), so it is the same every time;
        training passes a fresh rng for new noise each epoch."""
        if self.saved:
            if key not in self._saved_coadds:
                self._saved_coadds[key] = tuple(np.load(self.dir / f"{self.name}_{key}_{plane}.npy", mmap_mode="r")
                                                for plane in ("signal", "variance"))
            signal, variance = self._saved_coadds[key]
            return self.extract_halo(signal, x0, y0), self.extract_halo(variance, x0, y0)
        _, psf_fwhm, n_visit = self.coadd_settings(key)
        clean = broaden(self.extract_halo(self.base_image, x0, y0), psf_fwhm, self.base_fwhm, self.cfg)
        if rng is None:  # md5, not hash(): Python salts hash() per process
            digest = hashlib.md5(f"{self.name}|{key}|{int(x0)}|{int(y0)}".encode()).hexdigest()
            rng = np.random.default_rng(int(digest[:16], 16))
        return add_noise(clean, n_visit, rng, self.cfg)

    def full_coadd(self, key, seed):
        """(signal, variance) of a whole coadd in RAM: read from disk, or rebuilt with noise from seed."""
        if self.saved:
            return tuple(np.asarray(np.load(self.dir / f"{self.name}_{key}_{plane}.npy"), np.float32)
                         for plane in ("signal", "variance"))
        _, psf_fwhm, n_visit = self.coadd_settings(key)
        clean = broaden(np.asarray(self.base_image, np.float32), psf_fwhm, self.base_fwhm, self.cfg)
        return add_noise(clean, n_visit, np.random.default_rng(seed), self.cfg)

    def tile_grid(self):
        """Tile ids and lower-left corners covering the image."""
        size = self.cfg["tile_size"]
        n_x, n_y = int(np.ceil(self.nx / size)), int(np.ceil(self.ny / size))
        return pd.DataFrame([dict(tile_id=tx + n_x * ty, x0=tx * size, y0=ty * size)
                             for ty in range(n_y) for tx in range(n_x)])

    def sample_index(self, coadds=None):
        """Every (coadd, tile) pair, for the given coadds (default: all)."""
        tiles = self.tile_grid()
        return pd.concat([tiles.assign(combo=key) for key in (coadds or self.coadds)], ignore_index=True)


def log_variance_normalisation(store):
    """Per-band centre and scale for the log-variance input, from the sky variance of every coadd in the store,
    so the scaling is stable across the whole depth range (the S/N input needs none)."""
    log_sky_var = [[np.log(sky_sigma(band, store.coadd_settings(key)[2][band], store.cfg) ** 2) for key in store.coadds]
                   for band in BANDS]
    return dict(logvar_centre=[float(np.median(values)) for values in log_sky_var],
                logvar_scale=[float(max(np.std(values) + 0.5, 0.5)) for values in log_sky_var])
