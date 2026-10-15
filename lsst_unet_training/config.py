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
    # Input stretches: the network sees, per band, arcsinh(S/N / s) for each s in snr_stretches, plus the normalised
    # log variance. The first stretch (3) shows faint sources; the gentler one (100) keeps the shape of bright
    # sources, whose cores the first flattens (arcsinh is logarithmic at high S/N). mep_unet_v1 has (3,).
    snr_stretches=(3.0, 100.0),

    # Network and optimiser.
    base_filters=24, learning_rate=2e-4, batch_size=4, epochs=40, patience=6, seed=20260724,
    # The network's outputs ("heads", see unet_model.py). Every model also has centroid_offset and source_structure.
    # *_heatmap heads mark object centres: galaxy_heatmap (galaxies), star_heatmap (stars). *_map heads mark where a
    # phenomenon's light is: sfregion_map (star-forming regions), tidal_map (tidal streams and shells), spike_map
    # (diffraction spikes). detection_heatmap combines all of them into the final map of source (galaxy or star)
    # centres, which detections are taken from. Two more centre heads can be chosen: sfregion_heatmap and
    # tidal_heatmap, the centres of the catalogued star-forming regions and tidal blobs. extended_heatmap marks the
    # centres of resolved galaxies only, every resolved class carrying the same total weight at each S/N; it does not
    # feed detection_heatmap, but training it shapes the shared features towards extended galaxies.
    heads=("galaxy_heatmap", "star_heatmap", "sfregion_map", "tidal_map", "spike_map", "detection_heatmap",
           "extended_heatmap"),
    loss_weights=dict(detection_heatmap=1.0, galaxy_heatmap=0.5, star_heatmap=0.3, sfregion_map=0.2, tidal_map=0.2,
                      spike_map=0.2, extended_heatmap=1.0, centroid_offset=0.25, source_structure=0.10),

    # Training targets (targets.py). A galaxy's centre is a Gaussian whose width grows with its size, so an extended
    # galaxy is taught a broad peak rather than a pin-point one: sigma = target_sigma_per_re x Re (pixels), between
    # target_sigma_pix (also used for stars) and target_sigma_max_pix. A *_map head's truth is where that phenomenon's
    # own light is detectable in the coadd: its light, smoothed by a Gaussian of truth_map_filter_pix and combined
    # over bands by inverse variance, has S/N >= truth_map_snr.
    target_sigma_pix=1.5, target_sigma_per_re=0.25, target_sigma_max_pix=6.0, truth_map_snr=2.0,
    truth_map_filter_pix=1.5,
    # Which sources are taught as centres: a source counts as far as it can be seen in that coadd. Its size-matched S/N
    # (flux over the noise of a filter matched to its PSF-convolved size, combined over the bands; targets.band_snr)
    # decides. At S/N >= detectable_snr it is a positive. Below invisible_snr nothing of it shows in the noise, so it
    # is background, like empty sky. In between, whether it shows depends on the noise: it is neither, with no loss
    # around it. Its light stays in the image either way.
    detectable_snr=5.0, invisible_snr=2.0,
    # How the positives are weighted. weighting="balanced" (class-balanced): the galaxy positives of the train
    # catalogue in each coadd fall in cells of (class, S/N bin). The classes are unresolved galaxies, each added
    # population (BCGs, UDGs, extended dIrrs, almost-dark galaxies, whatever their size) and the resolved galaxies of
    # the main population by Hubble class (HUBBLE_CLASSES, the rest as "resolved, other"); the S/N bins run from
    # detectable_snr to the first of weight_snr_edges, between the edges, and above the last. Each cell is raised to
    # the count of the largest class in its S/N bin (weight = largest count / own count, at least 1, at most
    # max_population_weight), so at every S/N each class weighs as much as the largest, the S/N bins keep their
    # natural proportions and stars keep weight 1. Tiles are drawn in proportion to their positives' total weight (up
    # to max_tile_oversampling times the average) and each draw's loss is scaled by the inverse (stratified
    # sampling), so rare classes come up often without changing the expected loss. weighting="rarity" (the scheme of
    # mep_unet_v1): a galaxy's weight is its rarity in population (stellar mass, redshift, sSFR, surface brightness)
    # times its rarity in appearance (magnitude, size), a star's the rarity of its magnitude; weights average 1 and are
    # kept between 1/8 and max_population_weight, and tiles holding heavily weighted sources are drawn more often, up
    # to max_tile_oversampling times.
    weighting="balanced", weight_snr_edges=(10.0, 20.0, 50.0), max_population_weight=200.0, max_tile_oversampling=10.0,
    # Fixed focal-loss normalisers: before training, each centre head's average number of positives per batch is
    # measured on focal_normaliser_batches training batches and its loss divided by that constant instead of each
    # batch's summed positive weight, so a weight means the same in every batch (with a per-batch sum, a batch full
    # of up-weighted sources would be divided back down). 0: each batch's own sum.
    focal_normaliser_batches=60,
    # footprint_negatives: within footprint_re x Re of each galaxy weighted w > 1, every pixel of the centre heads'
    # loss counts w times and positives there at least w (see targets.TargetMaker.tile_targets), so boosting a class
    # does not also boost its look-alikes (its own star-forming knots, its disc) relative to its centre; with
    # footprint_power p the footprint counts w^p instead (p = 0: no footprint weighting). p = 0.5 keeps both the
    # boosted classes' completeness and the purity against their own substructure.
    footprint_negatives=True, footprint_re=2.0, footprint_power=0.5,
    # Beyond the image edge the network sees "no data" (zero signal, a huge variance), as for masked pixels. In
    # training, edge_augment_fraction of the tiles get a random artificial image edge, so edges are well learned.
    # edge_padding="reflect" fills it with a mirror image of the image instead.
    edge_padding="no_data", edge_augment_fraction=0.3, no_data_variance=1e12,

    # Missing bands (see unet_model.py, training.train_band_adapter). A missing band is fed as "no data" like the area
    # beyond the image edge. A model with band_adapter=True has a small extra network that corrects the backbone's
    # features only where some, but not all, bands are missing, so its outputs with all six bands are exactly the
    # backbone's. The adapter is trained after the backbone, with the backbone frozen, on tiles with bands dropped:
    # how many bands survive is drawn from band_keep_weights ({bands kept: relative frequency}), then which ones,
    # uniformly, so all 63 combinations are seen and the common ones (5 or 4 bands left) most often; or, with
    # probability band_partial_fraction, one band covers only part of the tile. In a dropped-band tile a source is a
    # positive only if detectable (S/N >= detectable_snr) in the bands left; one that is not, but would be visible
    # (S/N >= invisible_snr) with all bands, is "unknown": no loss either way. The adapter is also taught to
    # reproduce the six-band model's maps (weight distillation_weight).
    band_adapter=False, adapter_filters=32,
    band_keep_weights={5: 0.40, 4: 0.25, 3: 0.17, 2: 0.11, 1: 0.07},
    band_partial_fraction=0.25, distillation_weight=1.0,
    adapter_epochs=20, adapter_learning_rate=5e-4, adapter_patience=4,
    # Each band set gets its own score -> p_detection_centroid calibration (the network's scores need not mean the
    # same with fewer bands); one threshold on p_detection_centroid, chosen with all bands, applies to every set. The
    # sets calibrated: None means all 63 for a model with a band adapter and ugrizy alone for one without.
    calibration_band_sets=None,
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

# Source classes, for weighting (weighting="balanced") and for completeness by class. The populations the image
# generator adds to the main one ({lsb_population value: name}) and the Hubble classes of the main population
# ({name: Hubble types matched in full}).
ADDED_POPULATIONS = dict(bcg="BCGs", udg="UDGs", extended_dirr="extended dIrrs", almost_dark="almost-dark galaxies")
HUBBLE_CLASSES = {"resolved ellipticals": r"E\d", "resolved lenticulars": r"SB?0", "resolved spirals": r"SB?[abc]",
                  "resolved irregulars": r"Irr"}

CATALOGUE_STEM = "mock_catalogue"  # mock_lsst_image_generation's default catalogue file prefix
