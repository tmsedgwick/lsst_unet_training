"""Calibrate the trained detector on the calib catalogue, then score it on the sealed test catalogue.

Calibration maps each peak's raw score to p_detection_centroid, the probability that the peak is the centre of a
real source (a galaxy or a star within match_radius_pix), and chooses the threshold on it; it writes the matched
calib peaks and the threshold to model_dir, which with the weights and normalisation are everything the detector
needs at inference time. Evaluation applies that frozen calibration to test coadds and reports, per coadd, purity and
completeness relative to the image's own 5-sigma depth: for all galaxies, for extended ones (Re >= EXTENDED_RE_ARCSEC)
and for stars, plus completeness against galaxy size.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .calibration import (ALL_BANDS, calibrated_band_sets, calibrator_for, choose_threshold,  # noqa: E402
                          fit_calibrator, missing_bands, nearest_band_set, read_threshold)
from .masking import all_band_sets  # noqa: E402
from .coadd_data import CoaddStore, point_source_depth  # noqa: E402
from .config import ARTEFACTS, BANDS, CATALOGUE_STEM  # noqa: E402
from .peak_detection import match_peaks, predict_peaks  # noqa: E402
from .training import load_model, log_re_scaling  # noqa: E402

# Completeness is binned in r magnitude relative to each image's 5-sigma point-source depth (negative = brighter).
RELATIVE_MAG_EDGES = np.arange(-3.0, 2.001, 0.25)
RELATIVE_MAG_CENTRES = 0.5 * (RELATIVE_MAG_EDGES[:-1] + RELATIVE_MAG_EDGES[1:])
ABOVE_LIMIT = (-1.0, -0.25)  # "comfortably above the 5-sigma limit"
BELOW_LIMIT = (0.0, 1.0)
BRIGHTER_THAN_LIMIT = (-np.inf, -0.25)  # every source comfortably above the limit, however bright
EXTENDED_RE_ARCSEC = 2.0
RE_EDGES_ARCSEC = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0])  # galaxy size bins for completeness


SCORE_GRID = np.linspace(0.03, 1.0, 98)  # raw scores at which band sets' calibrations are compared


def calibrate_unet(catalogue_dir, image_dir, model_dir, cfg=None, calib_name="calib", stem=CATALOGUE_STEM):
    """Calibrate on the reference calib coadd: each band set (calibration.calibrated_band_sets) gets its own score ->
    p_detection_centroid calibration, from the coadd with the other bands missing; one threshold on
    p_detection_centroid is chosen with all bands. Also reports, per band set, the purity of its peaks at that
    threshold and how far its calibration is from the all-band one. Returns dict(threshold, status, ..., band_sets),
    also saved in the model's threshold file, with a calibration plot and table."""
    model, normalisation, model_config = load_model(model_dir, cfg)
    cfg, model_dir = model_config["cfg"], Path(model_dir)
    store = CoaddStore(calib_name, catalogue_dir, image_dir, cfg, stem)
    reference = cfg["reference_coadd"]
    if reference not in store.coadds:
        raise KeyError(f"reference coadd {reference!r} not in {calib_name}'s coadds: {store.coadds}")
    band_sets = calibrated_band_sets(cfg)
    if ALL_BANDS not in band_sets:
        raise ValueError(f"the calibrated band sets must include {ALL_BANDS}, which sets the threshold")
    tables, curves = {}, {}
    for band_set in band_sets:
        peaks = match_peaks(store, predict_peaks(model, store, normalisation, cfg, log_re_scaling(model_config),
                                                 coadds=[reference], missing=missing_bands(band_set)), cfg)
        if not bool(peaks["label_real"].any()):
            raise RuntimeError(f"the U-Net produced no peaks matching calib sources with bands {band_set}")
        calibrator = fit_calibrator(peaks)
        tables[band_set] = peaks.assign(band_set=band_set, p_detection_centroid=calibrator.predict(peaks["raw_score"]))
        curves[band_set] = calibrator.predict(SCORE_GRID)
    six = tables[ALL_BANDS]
    threshold, row, status = choose_threshold(six["p_detection_centroid"], six["label_real"], cfg["target_purity"],
                                              cfg["wilson_z"])
    per_set = {}
    for band_set, peaks in tables.items():
        kept = peaks["p_detection_centroid"].to_numpy(float) >= threshold
        real = peaks["label_real"].to_numpy(bool)
        per_set[band_set] = dict(n_bands=len(band_set), detections=int(kept.sum()),
                                 purity=float((kept & real).sum() / max(int(kept.sum()), 1)),
                                 calibration_difference=float(np.abs(curves[band_set] - curves[ALL_BANDS]).max()))
    pd.concat(tables.values(), ignore_index=True).to_parquet(model_dir / ARTEFACTS["calib_peaks"])
    (model_dir / ARTEFACTS["threshold"]).write_text(json.dumps(dict(
        threshold=threshold, target_purity=cfg["target_purity"], status=status, reference_combo=reference, **row,
        band_sets=per_set), indent=2))
    report = pd.DataFrame.from_dict(per_set, orient="index").rename_axis("band_set").reset_index()
    report.to_csv(model_dir / "mep_calibration_by_band_set.csv", index=False)
    plot_calibrations(curves, threshold, model_dir / "mep_calibration_by_band_set.png")
    print(f"p_detection_centroid threshold {threshold:.5f} on {reference} with all bands: purity {row['purity']:.4f} "
          f"(Wilson lower {row['purity_lower']:.4f}, target {cfg['target_purity']}, {status})")
    if status != "met":
        print("WARNING: the target purity could not be certified even on the reference image.")
    if len(band_sets) > 1:
        by_count = report.groupby("n_bands").agg(band_sets=("band_set", "size"), detections=("detections", "mean"),
                                                 purity_mean=("purity", "mean"), purity_worst=("purity", "min"),
                                                 calibration_difference_max=("calibration_difference", "max"))
        with pd.option_context("display.width", 160):
            print("At that threshold, by number of bands present (calibration difference: largest gap in "
                  "p_detection_centroid from the all-band calibration):")
            print(by_count.sort_index(ascending=False).round(4).to_string())
    return dict(threshold=threshold, status=status, **row, band_sets=per_set)


def plot_calibrations(curves, threshold, out_path):
    """Each band set's calibration (p_detection_centroid against raw score), coloured by its number of bands."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(BANDS)))
    for band_set, curve in sorted(curves.items(), key=lambda item: len(item[0])):
        ax.plot(SCORE_GRID, curve, color=colours[len(band_set) - 1], lw=2.5 if band_set == ALL_BANDS else 0.8,
                alpha=1.0 if band_set == ALL_BANDS else 0.6)
    ax.axhline(threshold, color="k", ls=":", label=f"threshold {threshold:.3f}")
    ax.legend(handles=[Line2D([0], [0], color=colours[k - 1], lw=2, label=f"{k} band{'s' if k > 1 else ''}")
                       for k in range(len(BANDS), 0, -1)] + [Line2D([0], [0], color="k", ls=":",
                                                                     label=f"threshold {threshold:.3f}")],
              fontsize=8, loc="lower right")
    ax.set(xlabel="raw score (detection map peak)", ylabel="p_detection_centroid",
           title="Calibration of each band set (the all-band set thick)", ylim=(-0.02, 1.02))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def score_test(catalogue_dir, image_dir, model_dir, coadds=None, cfg=None, test_name="test", stem=CATALOGUE_STEM,
               tile_cap=None, band_set=ALL_BANDS):
    """Matched test peaks with calibrated p_detection_centroid, for the given coadds (default: all), with only the
    bands of band_set (scored with its calibration, or that of the calibrated set closest to it)."""
    model, normalisation, model_config = load_model(model_dir, cfg)
    cfg, model_dir = model_config["cfg"], Path(model_dir)
    store = CoaddStore(test_name, catalogue_dir, image_dir, cfg, stem)
    peaks = match_peaks(store, predict_peaks(model, store, normalisation, cfg, log_re_scaling(model_config),
                                             coadds=coadds, tile_cap=tile_cap, missing=missing_bands(band_set)), cfg)
    calib_peaks = pd.read_parquet(model_dir / ARTEFACTS["calib_peaks"])
    calibrator = calibrator_for(calib_peaks, nearest_band_set(band_set, set(calib_peaks["band_set"])))
    peaks["p_detection_centroid"] = calibrator.predict(peaks["raw_score"]) if len(peaks) else []
    return store, peaks


def recovered_sources(peaks, kept, kind, n_sources):
    """Boolean array over a truth table (galaxies or stars) of the sources some kept peak matched."""
    recovered = np.zeros(n_sources, bool)
    of_kind = peaks["truth_kind"].to_numpy(object) == kind if "truth_kind" in peaks else np.full(len(peaks), True)
    hits = peaks["truth_index"].to_numpy(np.int64)[kept & peaks["label_real"].to_numpy(bool) & of_kind]
    recovered[hits[(hits >= 0) & (hits < n_sources)]] = True
    return recovered


def fraction(recovered, selected):
    return float((recovered & selected).sum()) / int(selected.sum()) if selected.any() else np.nan


def summarise(store, peaks, threshold, cfg):
    """Per-coadd purity and completeness above / below the image's 5-sigma limit (all galaxies, extended galaxies and
    stars), plus completeness curves against magnitude relative to the limit and against galaxy size."""
    kept_all = peaks["p_detection_centroid"].to_numpy(float) >= threshold
    measurable = store.truth_inside & np.isfinite(store.truth_mag_r)
    extended = store.truth_re_arcsec >= EXTENDED_RE_ARCSEC
    rows, curves, size_curves = [], {}, {}
    for key in [k for k in store.coadds if k in set(peaks["combo"])]:
        info, psf_fwhm, n_visit = store.coadd_settings(key)
        # Point-source depth, corrected for this coadd's seeing relative to nominal.
        depth = point_source_depth("r", n_visit["r"], cfg) + 2.5 * np.log10(cfg["nominal_fwhm"]["r"] / psf_fwhm["r"])
        relative_mag, star_relative_mag = store.truth_mag_r - depth, store.star_mag_r - depth
        in_coadd = (peaks["combo"] == key).to_numpy()
        coadd_peaks, kept = peaks[in_coadd], kept_all[in_coadd]
        galaxies = recovered_sources(coadd_peaks, kept, "galaxy", len(store.truth))
        stars = recovered_sources(coadd_peaks, kept, "star", len(store.stars))

        def in_range(values, low, high, base):
            return base & (values >= low) & (values < high)

        curves[key] = np.array([fraction(galaxies, in_range(relative_mag, low, high, measurable))
                                for low, high in zip(RELATIVE_MAG_EDGES[:-1], RELATIVE_MAG_EDGES[1:])])
        bright = in_range(relative_mag, *BRIGHTER_THAN_LIMIT, measurable)
        size_curves[key] = np.array([fraction(galaxies, in_range(store.truth_re_arcsec, low, high, bright))
                                     for low, high in zip(RE_EDGES_ARCSEC[:-1], RE_EDGES_ARCSEC[1:])])
        real = coadd_peaks["label_real"].to_numpy(bool)
        rows.append(dict(coadd=key, epoch=info["epoch"], fwhm_r=round(psf_fwhm["r"], 3), n_visit_r=n_visit["r"],
                         depth5_r=round(depth, 3), n_detections=int(kept.sum()),
                         purity=float((kept & real).sum()) / max(int(kept.sum()), 1),
                         completeness_above_limit=fraction(galaxies, in_range(relative_mag, *ABOVE_LIMIT, measurable)),
                         completeness_below_limit=fraction(galaxies, in_range(relative_mag, *BELOW_LIMIT, measurable)),
                         extended_completeness=fraction(galaxies, bright & extended),
                         star_completeness=fraction(stars, in_range(star_relative_mag, *BRIGHTER_THAN_LIMIT,
                                                                    store.star_inside))))
    return pd.DataFrame(rows), dict(magnitude=curves, size=size_curves)


def plot_results(summary, curves, target_purity, label, out_prefix):
    """Completeness vs magnitude relative to the limit and vs galaxy size (one line per coadd), and purity /
    completeness vs depth."""
    epochs = list(dict.fromkeys(summary["epoch"]))
    colours = dict(zip(epochs, plt.cm.viridis(np.linspace(0, 0.9, max(len(epochs), 2)))))
    styles = dict(zip(sorted(summary["fwhm_r"].unique()), ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 3))]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for row in summary.itertuples():
        ax.plot(RELATIVE_MAG_CENTRES, curves["magnitude"][row.coadd], color=colours[row.epoch],
                ls=styles.get(row.fwhm_r, "-"), lw=1.8)
    ax.axvline(0.0, color="k", lw=1.2, alpha=0.7)
    ax.set(xlabel="r magnitude relative to the image's 5σ depth (brighter ←)", ylabel="completeness",
           title=f"Test completeness vs 5σ limit: {label}\ncolour = survey epoch, line style = r-band PSF FWHM",
           ylim=(-0.02, 1.02))
    handles = [Line2D([0], [0], color=colours[e], lw=2, label=e) for e in epochs]
    handles += [Line2D([0], [0], color="0.4", lw=2, ls=s, label=f"FWHM$_r$ = {f}\"") for f, s in styles.items()]
    ax.legend(handles=handles, ncol=2, fontsize=8, loc="center left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_completeness_vs_limit.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    re_centres = np.sqrt(np.maximum(RE_EDGES_ARCSEC[:-1], 0.125) * RE_EDGES_ARCSEC[1:])
    for row in summary.itertuples():
        ax.plot(re_centres, curves["size"][row.coadd], color=colours[row.epoch], ls=styles.get(row.fwhm_r, "-"),
                lw=1.8, marker="o", ms=3)
    ax.set(xscale="log", xlabel="galaxy half-light radius Re (arcsec)", ylabel="completeness", ylim=(-0.02, 1.02),
           title=f"Completeness vs size, galaxies brighter than the 5σ limit: {label}")
    ax.legend(handles=handles, ncol=2, fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_completeness_vs_size.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for fwhm, group in summary.groupby("fwhm_r"):
        axes[0].plot(group["depth5_r"], group["completeness_above_limit"], marker="o", ls=styles.get(fwhm, "-"),
                     label=f"FWHM$_r$ = {fwhm}\"")
        axes[1].plot(group["depth5_r"], group["purity"], marker="s", ls=styles.get(fwhm, "-"),
                     label=f"FWHM$_r$ = {fwhm}\"")
    axes[1].axhline(target_purity, color="k", ls=":", alpha=0.7, label="target purity")
    axes[0].set(xlabel="5σ r depth (mag)", ylabel="completeness above limit", title="Completeness vs depth")
    axes[1].set(xlabel="5σ r depth (mag)", ylabel="purity", title="Purity vs depth")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_purity_completeness_vs_depth.png", dpi=180)
    plt.close(fig)


def evaluate_unet(catalogue_dir, image_dir, model_dir, coadds=None, cfg=None, test_name="test", stem=CATALOGUE_STEM,
                  label=None, tile_cap=None, band_sets=(ALL_BANDS,)):
    """Score the test coadds with the frozen calibration and threshold, for each band set (each with its own
    calibration; "all" for all 63); save peaks, a summary table (one row per band set and coadd) and plots. With
    several band sets, also a summary by number of bands: the mean and the worst set."""
    model_dir = Path(model_dir)
    threshold = read_threshold(model_dir)
    target_purity = json.loads((model_dir / ARTEFACTS["threshold"]).read_text())["target_purity"]
    band_sets = all_band_sets() if list(band_sets) == ["all"] else list(band_sets)
    name = "all" if coadds is None else "_".join(coadds)
    summaries, all_peaks = [], []
    for band_set in band_sets:
        store, peaks = score_test(catalogue_dir, image_dir, model_dir, coadds, cfg, test_name, stem, tile_cap,
                                  band_set)
        summary, curves = summarise(store, peaks, threshold, store.cfg)
        summary.insert(0, "n_bands", len(band_set))
        summary.insert(0, "band_set", band_set)
        if band_set == ALL_BANDS or len(band_sets) <= 8:
            plot_results(summary, curves, target_purity,
                         f"{label or ('all coadds' if coadds is None else ' & '.join(coadds))}, bands {band_set}",
                         model_dir / f"mep_{name}_{band_set}_test")
        summaries.append(summary)
        all_peaks.append(peaks.assign(band_set=band_set))
    summary = pd.concat(summaries, ignore_index=True)
    prefix = model_dir / f"mep_{name}_test"
    pd.concat(all_peaks, ignore_index=True).to_parquet(f"{prefix}_peaks.parquet")
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
        if len(band_sets) > 1:
            by_count = summarise_by_band_count(summary)
            by_count.to_csv(f"{prefix}_by_band_count.csv")
            plot_by_band_count(by_count, f"{prefix}_by_band_count.png")
            print("\nBy number of bands present (mean over band sets, and the worst set):")
            print(by_count.round(3).to_string())
    print(f"Saved test peaks, summary and plots with prefix {prefix}")
    return summary


def summarise_by_band_count(summary):
    """Per coadd and number of bands: mean and worst (minimum) completeness and purity over the band sets."""
    columns = ["completeness_above_limit", "extended_completeness", "purity"]
    grouped = summary.groupby(["coadd", "n_bands"])[columns]
    by_count = grouped.mean().add_suffix("_mean").join(grouped.min().add_suffix("_worst"))
    return by_count.sort_index(level=["coadd", "n_bands"], ascending=[True, False])


def plot_by_band_count(by_count, out_path):
    """Completeness above the limit and purity against the number of bands present (mean and worst band set)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for coadd, table in by_count.groupby(level="coadd"):
        n = table.index.get_level_values("n_bands")
        for ax, column in zip(axes, ["completeness_above_limit", "purity"]):
            line, = ax.plot(n, table[f"{column}_mean"], marker="o", label=f"{coadd} mean")
            ax.plot(n, table[f"{column}_worst"], marker="v", ls="--", color=line.get_color(), label=f"{coadd} worst")
    for ax, title in zip(axes, ["Completeness above the 5σ limit", "Purity"]):
        ax.set(xlabel="number of bands present", title=title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
