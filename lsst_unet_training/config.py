"""Default settings for training, calibrating and evaluating the U-Net detector.

Every function takes a cfg dict; pass a partial dict to override any key, e.g. ``dict(epochs=5)``.
"""

from typing import Any

BANDS = ["u", "g", "r", "i", "z", "y"]

CONFIG: dict[str, Any] = dict(
    # LSST imaging constants. These must match the IMAGE_CONFIG of mock_lsst_image_generation that made the images,
    # because train/valid coadds are rebuilt here with the same noise model.
    pixscale=0.2, zeropoint=31.4,
    nominal_fwhm=dict(u=1.16, g=1.11, r=1.05, i=1.01, z=0.97, y=0.95),  # 10-year median PSF FWHM (arcsec)
    depth_10yr=dict(u=26.2, g=27.4, r=27.6, i=26.9, z=26.1, y=24.8),  # 10-year 5-sigma point-source depth (AB)
    visits_10yr=dict(u=56, g=80, r=184, i=185, z=160, y=160),

    # Tiling: the network predicts a tile_size square and sees tile_halo pixels of context around it.
    tile_size=256, tile_halo=32, psf_stamp=25,

    # Network and optimiser.
    base_filters=24, learning_rate=2e-4, batch_size=4, epochs=40, patience=6, seed=20260724,
    clump_head=True, tidal_head=True,  # auxiliary heads that also learn clumps and tidal blobs
    loss_weights=dict(galaxy_heatmap=1.0, centroid_offset=0.25, source_structure=0.10, clump_heatmap=0.20,
                      tidal_heatmap=0.10),
    # Each epoch pairs every train tile with this many randomly chosen coadds (depth x seeing), with fresh noise.
    # Over many epochs the model sees the whole grid; this is the main cost lever.
    train_coadds_per_tile=1,
    valid_samples=300,  # fixed (tile, coadd) pairs used for early stopping
    data_workers=4,  # threads that build training batches while the model trains

    # Peak finding on the predicted galaxy heatmap.
    min_peak_score=0.03,  # local maxima below this are noise-floor bumps
    max_peaks_per_tile=512, match_radius_pix=3.0, infer_batch=16,

    # Calibration: one p_real threshold, chosen on this calib coadd (10 years, r-band FWHM ~1.1").
    reference_coadd="10y_fwhm110", target_purity=0.99, wilson_z=1.64,
)

# File names inside the model directory. They match what mep_unet_infer.py loads on the Rubin system.
ARTEFACTS = dict(
    weights="mep_unet_detector.weights.h5",
    normalisation="mep_unet_normalisation.json",
    model_config="mep_unet_model_config.json",
    history="mep_unet_training_history.csv",
    calib_peaks="mep_calib_peaks.parquet",
    threshold="mep_threshold.json",
)

CATALOGUE_STEM = "mock_catalogue"  # mock_lsst_image_generation's default catalogue file prefix
