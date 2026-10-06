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
    # The network's outputs ("heads", see unet_model.py). Every model also has centroid_offset and source_structure.
    # *_heatmap heads mark object centres: galaxy_heatmap (galaxies), star_heatmap (stars). *_map heads mark where a
    # phenomenon's light is: sfregion_map (star-forming regions), tidal_map (tidal streams and shells), spike_map
    # (diffraction spikes). detection_heatmap combines all of them into the final map of source (galaxy or star)
    # centres, which detections are taken from. Two more centre heads can be chosen: sfregion_heatmap and
    # tidal_heatmap, the centres of the catalogued star-forming regions and tidal blobs.
    heads=("galaxy_heatmap", "star_heatmap", "sfregion_map", "tidal_map", "spike_map", "detection_heatmap"),
    loss_weights=dict(detection_heatmap=1.0, galaxy_heatmap=0.5, star_heatmap=0.3, sfregion_map=0.2, tidal_map=0.2,
                      spike_map=0.2, centroid_offset=0.25, source_structure=0.10),

    # Training targets (targets.py). A galaxy's centre is a Gaussian whose width grows with its size, so an extended
    # galaxy is taught a broad peak rather than a pin-point one: sigma = target_sigma_per_re x Re (pixels), between
    # target_sigma_pix (also used for stars) and target_sigma_max_pix. A *_map head's truth is where that phenomenon's
    # own light is detectable in the coadd: its light, smoothed by a Gaussian of truth_map_filter_pix and combined
    # over bands by inverse variance, has S/N >= truth_map_snr.
    target_sigma_pix=1.5, target_sigma_per_re=0.25, target_sigma_max_pix=6.0, truth_map_snr=2.0,
    truth_map_filter_pix=1.5,
    # Rare sources count more: each galaxy is weighted by how rare it is in population (stellar mass, redshift, sSFR,
    # surface brightness) and in appearance (magnitude, size), each star by how rare its magnitude is, up to
    # max_population_weight times the average. Tiles holding rare sources are also drawn more often, up to
    # max_tile_oversampling times.
    max_population_weight=20.0, max_tile_oversampling=5.0,
    # Beyond the image edge the network sees "no data" (zero signal, a huge variance), as for masked pixels. In
    # training, edge_augment_fraction of the tiles get a random artificial image edge, so edges are well learned.
    # edge_padding="reflect" fills it with a mirror image of the image instead.
    edge_padding="no_data", edge_augment_fraction=0.3, no_data_variance=1e12,

    # Missing bands (see unet_model.py, training.train_band_adapter). A missing band is fed as "no data" like the area
    # beyond the image edge. A model with band_adapter=True has a small extra network that corrects the backbone's
    # features only where some, but not all, bands are missing, so its outputs with all six bands are exactly the
    # backbone's. The adapter is trained after the backbone, with the backbone frozen, on tiles with bands dropped:
    # whole bands, drawn from band_dropout_patterns ({bands removed: relative frequency}), or with probability
    # band_partial_fraction one band covering only part of the tile. A source that is detectable (combined S/N >=
    # detectable_snr) in all bands but not in those left is "unknown" in a dropped-band tile: no loss either way.
    # The adapter is also taught to reproduce the six-band model's maps (weight distillation_weight).
    band_adapter=False, adapter_filters=32,
    band_dropout_patterns={"u": 3, "y": 2, "g": 1, "r": 1, "i": 1, "z": 1, "uy": 3, "uzy": 2, "ugzy": 1, "uzgiy": 1,
                           "ugrzy": 1},
    band_partial_fraction=0.25, detectable_snr=5.0, distillation_weight=1.0,
    adapter_epochs=20, adapter_learning_rate=5e-4, adapter_patience=4,
    # Band sets calibrated (each has its own p_detection_centroid and threshold) and evaluated for adapter models.
    calibration_band_sets=("ugrizy", "grizy", "ugriz", "griz", "gri", "gr", "r"),
    # Each epoch pairs every train tile with this many randomly chosen coadds (depth x seeing), with fresh noise.
    # Over many epochs the model sees the whole grid; this is the main cost lever.
    train_coadds_per_tile=1,
    valid_samples=300,  # fixed (tile, coadd) pairs used for early stopping
    data_workers=4,  # threads that build training batches while the model trains

    # Peak finding on the predicted detection heatmap.
    min_peak_score=0.03,  # local maxima below this are noise-floor bumps
    max_peaks_per_tile=512, match_radius_pix=3.0, infer_batch=16,

    # Calibration: p_detection_centroid, the probability that a peak is the centre of a real source (galaxy or star),
    # and one threshold on it, chosen on this calib coadd (10 years, r-band FWHM ~1.1").
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
    # Labels of sources the model missed (real sources it gave p_detection_centroid below its threshold, and sources
    # the reviewer marked as missed) count this many times more again, so the update concentrates on its failures.
    update_miss_weight=3.0,
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
