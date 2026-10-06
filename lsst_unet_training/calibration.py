"""Turn raw heatmap scores into calibrated probabilities and pick the detection threshold.

On the calib catalogue, isotonic regression maps each peak's raw score to p_detection_centroid, the fraction of peaks
with that score that are the centre of a real source (a galaxy or star within match_radius_pix). The threshold is the
lowest p_detection_centroid at which the purity of all peaks above it is still at least target_purity, judged by the
Wilson lower bound (so it holds with ~95% confidence, not just on average). It is fixed once, on the reference coadd
(10 years, nominal seeing); shallower or blurrier images then lose completeness naturally at the same threshold.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


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


def fit_calibrator(peaks):
    """Isotonic map from raw_score to p_detection_centroid, fitted on matched calib peaks."""
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    return calibrator.fit(peaks["raw_score"], peaks["label_real"].astype(int))
