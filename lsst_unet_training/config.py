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
    clump_head=True, tidal_head=True,  # sub-structure heads: star-forming clumps and tidal features
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

    # Updating a trained model with labelled detections on an extra ("auxiliary") coadd, see update.py.
    # The labels are split into training and test labels by holding out a random aux_test_fraction of the
    # aux_block_pix x aux_block_pix image blocks. A label counts as detected if a peak above the threshold lies within
    # aux_match_radius_pix of it.
    aux_test_fraction=0.3, aux_block_pix=400, aux_match_radius_pix=3.0,
    aux_purity=0.95,  # update_threshold_on_aux: purity the training labels above the new threshold must reach
    # update_weights_on_aux (fine-tuning): a fraction update_aux_fraction of each batch is auxiliary tiles, the rest
    # mock tiles. On auxiliary tiles only pixels within label_radius_pix of a label count towards the loss. An
    # auxiliary tile has only a few labels while a mock tile has ~100 galaxies, so each counted auxiliary pixel is
    # weighted update_aux_weight times a mock pixel.
    update_learning_rate=2e-5, update_epochs=5, update_steps_per_epoch=200, update_aux_fraction=0.5,
    update_aux_weight=10.0, label_radius_pix=8.0,
)

# File names inside the model directory.
ARTEFACTS = dict(
    weights="mep_unet_detector.weights.h5",
    normalisation="mep_unet_normalisation.json",
    model_config="mep_unet_model_config.json",
    history="mep_unet_training_history.csv",
    calib_peaks="mep_calib_peaks.parquet",
    threshold="mep_threshold.json",
)

CATALOGUE_STEM = "mock_catalogue"  # mock_lsst_image_generation's default catalogue file prefix
