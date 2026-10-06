"""Turning parts of a tile into "no data": artificial image edges and missing bands.

"No data" is zero signal and a huge variance (cfg["no_data_variance"]), which the network reads as missing, the same
as the area beyond a real image's edge or a masked pixel. Training uses it to teach the network about image edges
(every band missing beyond an artificial edge) and, for the band adapter, about missing bands (one or more bands
missing over the whole tile, or one band covering only part of it).
"""

import numpy as np

from .config import BANDS


def artificial_edge(full, rng):
    """(full, full) mask of the pixels beyond a random straight image edge (or two, making a corner) cutting the
    tile: what a tile at the edge of a real image would be missing."""
    beyond = np.zeros((full, full), bool)
    for _ in range(1 + int(rng.random() < 0.3)):
        side, cut = int(rng.integers(4)), int(rng.integers(full // 8, full - full // 8))
        if side == 0:
            beyond[:, :cut] = True
        elif side == 1:
            beyond[:, cut:] = True
        elif side == 2:
            beyond[:cut, :] = True
        else:
            beyond[cut:, :] = True
    return beyond


def cut_away(signal, variance, targets, beyond, no_data_variance):
    """Make the pixels in beyond "no data" in every band and drop them from every loss."""
    signal[:, beyond], variance[:, beyond] = 0.0, no_data_variance
    for value in targets.values():
        value[beyond, -1] = 0.0  # the last channel of every target is its loss weight
    return signal, variance, targets


def coverage(variance, cfg):
    """(n_band, H, W) True where a band has data."""
    return np.isfinite(variance) & (variance < 0.1 * cfg["no_data_variance"])


def drop_bands(signal, variance, psf_kernels, missing, cfg, region=None):
    """Copies of a tile's (band, H, W) signal and variance and (stamp, stamp, band) PSF stamps with the bands named in
    missing (e.g. "uy") turned to "no data": everywhere, or only inside region ((H, W) bool), for a band that covers
    part of the tile. A band missing everywhere gets the mean of the other bands' PSF stamps, so the PSF conditioning
    sees a plausible PSF."""
    signal, variance, psf_kernels = signal.copy(), variance.copy(), psf_kernels.copy()
    dropped = [BANDS.index(band) for band in missing]
    where = (slice(None), slice(None)) if region is None else region
    for b in dropped:
        signal[b][where], variance[b][where] = 0.0, cfg["no_data_variance"]
    if region is None and dropped:
        present = [b for b in range(len(BANDS)) if b not in dropped]
        mean_stamp = psf_kernels[..., present].mean(axis=-1)
        for b in dropped:
            psf_kernels[..., b] = mean_stamp / mean_stamp.sum()
    return signal, variance, psf_kernels


def sample_band_dropout(rng, full, cfg):
    """(missing bands, region) for one training tile: with probability band_partial_fraction one random band
    covering only part of the tile (region = the part it misses), otherwise whole bands drawn from
    band_dropout_patterns (region None)."""
    if rng.random() < cfg["band_partial_fraction"]:
        return BANDS[int(rng.integers(len(BANDS)))], artificial_edge(full, rng)
    patterns = list(cfg["band_dropout_patterns"])
    weights = np.array([cfg["band_dropout_patterns"][p] for p in patterns], float)
    return patterns[int(rng.choice(len(patterns), p=weights / weights.sum()))], None
