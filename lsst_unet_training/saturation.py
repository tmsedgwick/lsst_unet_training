"""Saturated star cores, applied to coadd tiles rebuilt during training so they match the saved coadds.

mock_lsst_image_generation saturates bright stars in every coadd it saves (stars.saturate_stars); the train and valid
coadds are rebuilt here instead, so this applies the same model: where a star's noise-free light exceeds its
saturation level, each row of the saturated region holds one flat value that scatters about the level from row to
row, with a slightly dimmed rim. The settings must match mock_lsst_image_generation's STAR_CONFIG.
"""

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates

SATURATION: dict[str, Any] = dict(
    saturation_ratio=dict(u=1.92, g=0.98, r=1.0, i=0.84, z=1.16, y=2.59),  # level per band / r-band level
    row_scatter=0.15, row_correlation=2.0, core_stretch=(1.0, 1.25), core_shift_px=1.5,
    max_half_width_pix=60,  # saturated cores are at most a few arcsec across
)


def core_half_width(mag_r):
    """Half-width (pixels) of the box searched for a star's saturated core."""
    return int(np.clip(40 + 30 * (18 - mag_r), 40, SATURATION["max_half_width_pix"]))


def smooth_series(n, rng, correlation):
    """n values of smooth, unit-variance random noise (Gaussian-smoothed white noise)."""
    kernel = np.exp(-0.5 * (np.arange(-4 * correlation, 4 * correlation + 1) / correlation) ** 2)
    series = np.convolve(rng.normal(size=n + len(kernel)), kernel, mode="same")[len(kernel) // 2:][:n]
    return (series - series.mean()) / max(series.std(), 1e-9)


def fill_saturated_core(image, clean, level, rng, stretch, shift):
    """One band of one star's box: rows of the region where `clean` exceeds `level` (stretched and shifted) become
    flat values scattered about the level."""
    ny, nx = clean.shape
    yc, xc = np.unravel_index(np.argmax(clean), clean.shape)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    saturated = map_coordinates(clean, [yc + (yy - yc - shift[0]) / stretch, xx - shift[1]], order=1) > level
    rows = np.flatnonzero(saturated.any(axis=1))
    if len(rows) == 0:
        return image
    values = level * np.exp(SATURATION["row_scatter"] * smooth_series(len(rows), rng, SATURATION["row_correlation"]))
    jitter = [np.rint(0.8 * smooth_series(len(rows), rng, 1.0)).astype(int) for _ in range(2)]
    filled = np.zeros_like(saturated)
    for row, value, left, right in zip(rows, values, *jitter):
        cols = np.flatnonzero(saturated[row])
        start, stop = max(cols[0] - left, 0), min(cols[-1] + right, nx - 1)
        filled[row, start:stop + 1] = True
        image[row, start:stop + 1] = value
    rim = binary_dilation(filled, iterations=2) & ~filled
    image[rim] *= 1 - 0.15 * rng.uniform()
    return image


def saturate_stars(signal, clean, stars_x, stars_y, mag_r, saturation_r, bands, rng):
    """Saturate the stars of a (band, ny, nx) noisy `signal`, given its noise-free version `clean`. Star positions are
    in the array's pixels; stars whose box does not fit inside the array are left alone."""
    ny, nx = signal.shape[1:]
    for x, y, mag, level_r in zip(np.rint(stars_x).astype(int), np.rint(stars_y).astype(int), mag_r, saturation_r):
        h = core_half_width(mag)
        if not (h <= x < nx - h and h <= y < ny - h) or not np.isfinite(level_r):
            continue
        box = (slice(y - h, y + h + 1), slice(x - h, x + h + 1))
        stretch, shift = rng.uniform(*SATURATION["core_stretch"]), rng.normal(0, SATURATION["core_shift_px"], 2)
        for i, band in enumerate(bands):
            level = level_r * SATURATION["saturation_ratio"][band]
            if clean[i][box].max() > level:
                signal[i][box] = fill_saturated_core(signal[i][box].copy(), clean[i][box], level, rng, stretch, shift)
    return signal
