"""Calibrate the trained detector on the calib catalogue, then score it on the sealed test catalogue.

Calibration writes the matched calib peaks and the threshold to model_dir; together with the weights and
normalisation they are everything the detector needs at inference time. Evaluation applies that frozen calibration
to test coadds and reports, per coadd, purity and completeness relative to the image's own 5-sigma depth.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .calibration import choose_threshold, fit_calibrator  # noqa: E402
from .coadd_data import CoaddStore, point_source_depth  # noqa: E402
from .config import ARTEFACTS, CATALOGUE_STEM  # noqa: E402
from .peak_detection import match_peaks, predict_peaks  # noqa: E402
from .training import load_model, log_re_scaling  # noqa: E402

# Completeness is binned in r magnitude relative to each image's 5-sigma point-source depth (negative = brighter).
RELATIVE_MAG_EDGES = np.arange(-3.0, 2.001, 0.25)
RELATIVE_MAG_CENTRES = 0.5 * (RELATIVE_MAG_EDGES[:-1] + RELATIVE_MAG_EDGES[1:])
ABOVE_LIMIT = (-1.0, -0.25)  # "comfortably above the 5-sigma limit"
BELOW_LIMIT = (0.0, 1.0)


def calibrate_unet(catalogue_dir, image_dir, model_dir, cfg=None, calib_name="calib", stem=CATALOGUE_STEM):
    """Match peaks on the reference calib coadd, fit the score -> p_real calibration and choose the threshold."""
    model, normalisation, model_config = load_model(model_dir, cfg)
    cfg, model_dir = model_config["cfg"], Path(model_dir)
    store = CoaddStore(calib_name, catalogue_dir, image_dir, cfg, stem)
    reference = cfg["reference_coadd"]
    if reference not in store.coadds:
        raise KeyError(f"reference coadd {reference!r} not in {calib_name}'s coadds: {store.coadds}")
    peaks = match_peaks(store, predict_peaks(model, store, normalisation, cfg, log_re_scaling(model_config),
                                             coadds=[reference]), cfg)
    if not bool(peaks["label_real"].any()):
        raise RuntimeError("the U-Net produced no peaks matching calib galaxies")
    peaks["p_real"] = fit_calibrator(peaks).predict(peaks["raw_score"])
    threshold, row, status = choose_threshold(peaks["p_real"], peaks["label_real"], cfg["target_purity"],
                                              cfg["wilson_z"])
    peaks.to_parquet(model_dir / ARTEFACTS["calib_peaks"])
    (model_dir / ARTEFACTS["threshold"]).write_text(json.dumps(dict(
        threshold=threshold, target_purity=cfg["target_purity"], status=status, reference_combo=reference, **row),
        indent=2))
    print(f"p_real threshold {threshold:.5f} on {reference}: purity {row['purity']:.4f} "
          f"(Wilson lower {row['purity_lower']:.4f}, target {cfg['target_purity']}, {status})")
    if status != "met":
        print("WARNING: the target purity could not be certified even on the reference image.")
    return threshold


def score_test(catalogue_dir, image_dir, model_dir, coadds=None, cfg=None, test_name="test", stem=CATALOGUE_STEM,
               tile_cap=None):
    """Matched test peaks with calibrated p_real, for the given coadds (default: all)."""
    model, normalisation, model_config = load_model(model_dir, cfg)
    cfg, model_dir = model_config["cfg"], Path(model_dir)
    store = CoaddStore(test_name, catalogue_dir, image_dir, cfg, stem)
    peaks = match_peaks(store, predict_peaks(model, store, normalisation, cfg, log_re_scaling(model_config),
                                             coadds=coadds, tile_cap=tile_cap), cfg)
    peaks["p_real"] = fit_calibrator(pd.read_parquet(model_dir / ARTEFACTS["calib_peaks"])).predict(peaks["raw_score"])
    return store, peaks


def summarise(store, peaks, threshold, cfg):
    """Per-coadd purity and completeness above / below the image's 5-sigma limit, plus completeness curves."""
    kept_all = peaks["p_real"].to_numpy(float) >= threshold
    measurable = store.truth_inside & np.isfinite(store.truth_mag_r)
    rows, curves = [], {}
    for key in [k for k in store.coadds if k in set(peaks["combo"])]:
        info, psf_fwhm, n_visit = store.coadd_settings(key)
        # Point-source depth, corrected for this coadd's seeing relative to nominal.
        depth = point_source_depth("r", n_visit["r"], cfg) + 2.5 * np.log10(cfg["nominal_fwhm"]["r"] / psf_fwhm["r"])
        relative_mag = store.truth_mag_r - depth
        in_coadd = (peaks["combo"] == key).to_numpy()
        kept, real = kept_all[in_coadd], peaks["label_real"].to_numpy(bool)[in_coadd]
        recovered = np.zeros(len(store.truth_inside), bool)
        hits = peaks["truth_index"].to_numpy(np.int64)[in_coadd][kept & real]
        recovered[hits[(hits >= 0) & (hits < len(recovered))]] = True

        def completeness(low, high):
            selected = measurable & (relative_mag >= low) & (relative_mag < high)
            return float((recovered & selected).sum()) / max(int(selected.sum()), 1) if selected.any() else np.nan

        curves[key] = np.array([completeness(low, high)
                                for low, high in zip(RELATIVE_MAG_EDGES[:-1], RELATIVE_MAG_EDGES[1:])])
        rows.append(dict(coadd=key, epoch=info["epoch"], fwhm_r=round(psf_fwhm["r"], 3), n_visit_r=n_visit["r"],
                         depth5_r=round(depth, 3), n_detections=int(kept.sum()),
                         purity=float((kept & real).sum()) / max(int(kept.sum()), 1),
                         completeness_above_limit=completeness(*ABOVE_LIMIT),
                         completeness_below_limit=completeness(*BELOW_LIMIT)))
    return pd.DataFrame(rows), curves


def plot_results(summary, curves, target_purity, label, out_prefix):
    """Completeness vs magnitude relative to the limit (one line per coadd), and purity / completeness vs depth."""
    epochs = list(dict.fromkeys(summary["epoch"]))
    colours = dict(zip(epochs, plt.cm.viridis(np.linspace(0, 0.9, max(len(epochs), 2)))))
    styles = dict(zip(sorted(summary["fwhm_r"].unique()), ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 3))]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for row in summary.itertuples():
        ax.plot(RELATIVE_MAG_CENTRES, curves[row.coadd], color=colours[row.epoch], ls=styles.get(row.fwhm_r, "-"),
                lw=1.8)
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
                  label=None, tile_cap=None):
    """Score the test coadds with the frozen calibration and threshold; save peaks, a summary table and plots."""
    model_dir = Path(model_dir)
    store, peaks = score_test(catalogue_dir, image_dir, model_dir, coadds, cfg, test_name, stem, tile_cap)
    threshold_info = json.loads((model_dir / ARTEFACTS["threshold"]).read_text())
    summary, curves = summarise(store, peaks, threshold_info["threshold"], store.cfg)
    label = label or ("all coadds" if coadds is None else " & ".join(coadds))
    prefix = model_dir / f"mep_{'all' if coadds is None else '_'.join(coadds)}_test"
    peaks.to_parquet(f"{prefix}_peaks.parquet")
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    plot_results(summary, curves, threshold_info["target_purity"], label, prefix)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
    print(f"Saved test peaks, summary and plots with prefix {prefix}")
    return summary
