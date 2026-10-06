"""Turn raw heatmap scores into calibrated probabilities and pick the detection threshold.

On the calib catalogue, isotonic regression maps each peak's raw score to p_detection_centroid, the fraction of peaks
with that score that are the centre of a real source (a galaxy or star within match_radius_pix). The threshold is the
lowest p_detection_centroid at which the purity of all peaks above it is still at least target_purity, judged by the
Wilson lower bound (so it holds with ~95% confidence, not just on average). It is fixed once, on the reference coadd
(10 years, nominal seeing); shallower or blurrier images then lose completeness naturally at the same threshold.

The same raw score means less with fewer bands, so this is done for each band set in cfg["calibration_band_sets"]
(e.g. "ugrizy", "griz"): the calib coadd is scored with the other bands missing, and that band set gets its own
calibration and threshold. A coadd is detected with the calibration of its own band set, or of the calibrated set
closest to it.
"""

import json
from pathlib import Path


import numpy as np
from sklearn.isotonic import IsotonicRegression

from .config import ARTEFACTS, BANDS

ALL_BANDS = "".join(BANDS)


def wilson_lower(successes, trials, z):
    """Wilson score lower bound on a binomial proportion."""
    if trials == 0:
        return 0.0
    p = successes / trials
    centre = p + z * z / (2 * trials)
    margin = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (centre - margin) / (1.0 + z * z / trials)


def choose_threshold(p_detection_centroid, is_real, target_purity, z):
    """(threshold, dict(purity, purity_lower, n), status): the lowest threshold whose Wilson lower-bound purity meets
    the target ('met'), or the one with the best lower bound if none does ('unmet')."""
    order = np.argsort(-np.asarray(p_detection_centroid, float))
    scores, labels = np.asarray(p_detection_centroid, float)[order], np.asarray(is_real, int)[order]
    true_positives, n_kept = np.cumsum(labels), np.arange(1, len(labels) + 1)
    lower = np.array([wilson_lower(int(tp), int(n), z) for tp, n in zip(true_positives, n_kept)])
    meets = np.flatnonzero(lower >= target_purity)
    i = int(meets[-1]) if len(meets) else int(np.argmax(lower))
    row = dict(purity=float(true_positives[i] / n_kept[i]), purity_lower=float(lower[i]), n=int(n_kept[i]))
    return float(scores[i]), row, "met" if len(meets) else "unmet"


def missing_bands(band_set):
    """The bands not in band_set, e.g. "uy" for "griz"."""
    return "".join(band for band in BANDS if band not in band_set)


def read_thresholds(model_dir):
    """{band set: dict(threshold, status, purity, purity_lower, n)} from a model's threshold file."""
    return json.loads((Path(model_dir) / ARTEFACTS["threshold"]).read_text())["band_sets"]


def nearest_band_set(band_set, calibrated):
    """band_set if it was calibrated, else the calibrated set sharing the most bands with it (then the one with the
    fewest bands it lacks)."""
    if band_set in calibrated:
        return band_set
    return max(calibrated, key=lambda other: (len(set(other) & set(band_set)), -len(set(other) - set(band_set))))


def calibrator_for(calib_peaks, band_set):
    """The p_detection_centroid calibration of one band set, from a model's calib peaks."""
    return fit_calibrator(calib_peaks[calib_peaks["band_set"] == band_set])


def fit_calibrator(peaks):
    """Isotonic map from raw_score to p_detection_centroid, fitted on matched calib peaks."""
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    return calibrator.fit(peaks["raw_score"], peaks["label_real"].astype(int))
