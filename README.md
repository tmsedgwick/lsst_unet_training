# lsst_unet_training

Train, calibrate and evaluate a noise- and PSF-aware U-Net that detects galaxies and stars in LSST-like coadds, and maps
star-forming regions, tidal features and diffraction spikes, using the mock catalogues and images made by [mock_lsst_image_generation](https://github.com/tmsedgwick/mock_lsst_image_generation).
One network covers every survey depth from 1 month to 10 years and every seeing from 0.7″ to 2.0″, because it is told
each image's noise and PSF instead of having to guess them.

```bash
python scripts/train_unet.py     --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet
python scripts/calibrate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet
python scripts/evaluate_unet.py  --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet
```

## How the four catalogues are used

| Catalogue | Role |
|---|---|
| `train` | fits the network weights; every epoch each tile gets a random depth and seeing, and fresh noise |
| `valid` | early stopping and learning-rate schedule (a fixed set of 300 tiles across depths and seeings) |
| `calib` | turns raw scores into probabilities and fixes the detection threshold |
| `test` | sealed until the final evaluation |

Train and valid coadds are rebuilt tile by tile on the fly from each catalogue's noise-free render, with the image
generator's PSF, noise and saturated-star models, so they need no disk space; calib and test coadds are read from
disk. Beyond the image edge a tile holds "no data" (zero signal, a huge variance), as a masked region would.

## Model architecture

**Input.** The image is cut into 256 × 256 pixel tiles, each read with a 32 pixel halo of context (320 × 320 in
total). For every band (ugrizy) the network sees two planes: `arcsinh(S/N / 3)`, where S/N is signal / √variance,
and the log variance, normalised per band. Working in S/N rather than flux makes a 1-month and a 10-year image look
alike to the network. The PSF is given separately, as one 25 × 25 unit-sum PSF stamp per band.

**Backbone.** A U-Net: an encoder of four residual blocks (24, 48, 96 and 192 filters, each halving the resolution),
a dilated bottleneck (288 filters, spatial dropout 0.15), and a decoder that upsamples back to full resolution,
concatenating the matching encoder features at each scale. Each residual block is two 3 × 3 convolutions with group
normalisation and swish activations, plus a shortcut connection.

**PSF conditioning (FiLM).** A small convolutional encoder turns the six PSF stamps into a 64-number embedding. At
every encoder level and the bottleneck, that embedding predicts a per-channel scale γ and shift β, and the features
become `features × (1 + γ) + β`. The same weights therefore adapt to sharp or blurry images; on real LSST data the
stamps come from `psf.computeKernelImage()` with no change to the network.

**Heads.** Outputs at full resolution. `*_heatmap` heads mark object centres; `*_map` heads mark where a
phenomenon's light is, so running the model on a real image gives, for example, a map of its tidal features.

| Head | Output | Loss (weight) |
|---|---|---|
| `detection_heatmap` | probability of a source (galaxy or star) centre; its peaks are the detections | focal (1.0) |
| `galaxy_heatmap` | probability of a galaxy centre | focal (0.5) |
| `star_heatmap` | probability of a star centre | focal (0.3) |
| `clump_map` | probability that a pixel holds detectable light of a star-forming region | focal (0.2) |
| `tidal_map` | probability that a pixel holds detectable light of a tidal stream or shell | focal (0.2) |
| `spike_map` | probability that a pixel lies on a diffraction spike | focal (0.2) |
| `centroid_offset` | sub-pixel offset from the peak pixel to the true centre | Huber at centres (0.25) |
| `source_structure` | log size, axis ratio, sin 2PA, cos 2PA | Huber at centres (0.10) |

`detection_heatmap` reads the decoder features and every map above, so it can add stars to the galaxies and reject
peaks the maps say are a star-forming region, tidal feature or spike. It starts as an exact copy of the galaxy map
and learns its correction, so the heads can also be added to an already trained model without changing its
detections (see "Update a trained model" below).

**Targets.** The centre-heatmap loss is the CenterNet-style focal loss: each true centre is a positive, pixels near a
centre are down-weighted negatives, and the halo is ignored. A galaxy's centre peak widens with its size (σ = 0.25 Re,
1.5 to 6 pixels), so an extended galaxy is taught a broad centre rather than a pin-point one. A map head's truth is
where its phenomenon's own light (saved alone by the image generator) is detectable in that coadd: smoothed over 1.5
pixels and combined over bands, at S/N ≥ 2. Heads whose truth the mocks lack (mocks made without stars, or without
these light images) get no loss, rather than being taught that the thing never occurs.

**Rare sources count more.** A galaxy's weight is its rarity in population (surface brightness, stellar mass,
redshift, sSFR) times its rarity in appearance (r magnitude, size); a star's is its rarity in magnitude. Weights are
scaled to average 1 and capped at 20, and tiles holding rare sources (bright, extended galaxies, bright stars) are
also drawn up to 5 times as often. The model has 3.6 million parameters.

**Training.** Adam (learning rate 2 × 10⁻⁴, gradient clipping), batch size 4, up to 40 epochs. The best weights
(lowest validation loss of the detection map) are kept; training stops after 6 epochs without improvement, and the learning rate
halves after 2. A third of the training tiles get an artificial image edge (one side or a corner turned to
"no data"), so sources near real image edges are learned too.

**Detection and calibration.** Detections are the 3 × 3 local maxima of the detection heatmap above 0.03, refined by
the predicted offset; each also carries the galaxy and star heatmaps' values (`galaxy_score`, `star_score`), which say
what kind of source it is. On the calib catalogue, peaks within 3 pixels of a true galaxy or star are labelled real,
and isotonic regression maps each raw score to `p_detection_centroid`, the chance that a peak with that score is the
centre of a real source. The threshold is the lowest `p_detection_centroid` at which the purity of everything above
it is at least 99%, taken on the Wilson lower bound (≈95% confidence). It is fixed once, on the 10-year, nominal-seeing calib coadd; shallower or blurrier images then lose
completeness naturally at the same threshold.

**Evaluation.** On the test catalogue, for each coadd: purity, and completeness as a function of r magnitude relative
to that image's own 5σ point-source depth, so a 1-month and a 10-year image are compared on equal terms; completeness
of extended galaxies (Re ≥ 2″) and of stars brighter than the limit; and completeness against galaxy size.

**Older models.** A model's config lists its heads and edge padding. Models saved before these settings had
`galaxy_heatmap`, `clump_heatmap` and `tidal_heatmap` (centres of clumps and tidal blobs) and mirrored the image
beyond its edge; they still load and run that way, and `update_weights_on_aux.py` gives them the current heads.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions the code was verified with, including `tensorflow-metal` for the GPU on
Apple-silicon Macs (skipped automatically elsewhere). On a Linux machine with an NVIDIA GPU, install
`tensorflow[and-cuda]` instead of `tensorflow`. Training on CPU works but is very slow.

## Example commands

The folders are examples: `--catalogue-dir` and `--image-dir` are the output folders of mock_lsst_image_generation,
and `--model-dir` is where the model goes. Keep them outside the repo.

**Full workflow**

```bash
# 1. Train (train + valid catalogues). Writes the weights, normalisation, model config, history and a loss plot.
python scripts/train_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet

# 2. Calibrate on the calib catalogue: p_detection_centroid calibration and the 99%-purity threshold.
python scripts/calibrate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet

# 3a. Quick test look: 1-year and 10-year coadds at nominal seeing (a few minutes).
python scripts/evaluate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --coadds 1y_fwhm110 10y_fwhm110

# 3b. Full test evaluation over every coadd (slow: all 42 depth x seeing combinations).
python scripts/evaluate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet
```

**Training options**

```bash
# A short trial run
python scripts/train_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet_trial --epochs 3

# Larger batches (needs more GPU memory)
python scripts/train_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --batch-size 8

# Retrain into a folder that already holds a model
python scripts/train_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --overwrite

# Catalogues generated with --numbered: choose which play train and valid
python scripts/train_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --train 1 --valid 2
```

**Calibration and evaluation options**

```bash
# A stricter purity target
python scripts/calibrate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --aux-purity 0.995

# Fix the threshold on a different calib coadd, e.g. 10 years at 1.3" seeing
python scripts/calibrate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --reference-coadd 10y_fwhm130

# Score only the worst seeing at three depths
python scripts/evaluate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --coadds 1y_fwhm200 5y_fwhm200 10y_fwhm200

# Fast preview on a random 200 (coadd, tile) pairs
python scripts/evaluate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --tile-cap 200
```

**Update a trained model with labelled detections on an extra coadd**

The model is trained and calibrated on mocks. If you have another coadd with labelled detections, the "auxiliary
coadd", you can use it to update the model. It can be a real coadd inspected by eye, a mock made with different
settings, or anything else. It is given as an `.npz` with `signal` and `variance` (band, y, x), `psf_kernels` and
`bands`.

Labels are pixel positions on the auxiliary coadd marked as a source or as spurious. They can come from the review
JSON files of lsst_unet_detection's review tool, one per candidate category, holding the candidates marked "real" or
"spurious" (unsure ones are ignored), detections picked by clicking, and the sources marked as missed (format in
`lsst_unet_training/update.py`). They can also come from a CSV with columns `x`, `y`, `label` ("real" or "spurious")
and optionally `weight` and `reason`. A label can carry a reason: for a spurious detection `spike`, `bridge`,
`sf_region`, `tidal`, `bad_centroid` or `hallucination`; for a real one `star`. Reasons that name a phenomenon also
teach its map when fine-tuning (e.g. `tidal` teaches `tidal_map`, `star` teaches `star_heatmap`).

Only labels that came up in the random review order are an unbiased sample, so only they go into the statistics
(the threshold, purity, recall); clicked and missed labels are used for fine-tuning only. If you reviewed only a
sample of a category (e.g. 157 of 11,324 U-Net-only candidates), each random label is weighted by
n_candidates / n_reviewed, so that category counts in proportion to its size.

Within each category the labels are split into training and test labels by 400-pixel image blocks, fixed by the seed,
so every category has test labels and both commands hold out the same ones. If no threshold reaches the purity goal,
`update_threshold_on_aux.py` keeps the current one. `update_weights_on_aux.py` gives the model's misses 3 times the
weight of other labels, draws mock tiles with rare sources more often, adds artificial image edges, and gives a model
made before the current heads those it can learn here (the mocks or the labels must hold their truth). Neither command changes the model you start from. Each writes a new folder
`<model-dir>_<suffix>` and refuses a suffix that is already taken. Both print, before and after the update:
- how many labelled sources and spurious detections are detected, with weighted recall and purity;
- the number of detections over the whole auxiliary coadd;
- purity and completeness on the mock test coadds, so you can check the mock performance has not got worse
  (default: the reference coadd; `--mock-coadds` for more, `--tile-cap` for a quick look).

The labels only describe the candidates that were reviewed: a lower threshold also lets through unreviewed peaks
elsewhere, which shows up in the whole-coadd detection count.

```bash
# Threshold only: the lowest p_detection_centroid at which the random training labels reach --aux-purity (0.95).
python scripts/update_threshold_on_aux.py --model-dir ~/mocks/unet --suffix thr1 --aux-coadd deep_coadd_cutout.npz --aux-labels feedback_unet_only.json feedback_peakfinder_only.json --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images

# Fine-tune the network: each batch is half auxiliary tiles (loss only within 8 px of a label) and half mock
# training tiles, at learning rate 2e-5. The result is recalibrated on the mock calib catalogue.
python scripts/update_weights_on_aux.py --model-dir ~/mocks/unet --suffix w1 --aux-coadd deep_coadd_cutout.npz --aux-labels feedback_unet_only.json feedback_peakfinder_only.json --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images

# Then choose the fine-tuned model's threshold from the same labels
python scripts/update_threshold_on_aux.py --model-dir ~/mocks/unet_w1 --suffix thr1 --aux-coadd deep_coadd_cutout.npz --aux-labels feedback_unet_only.json feedback_peakfinder_only.json --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images
```

Each new folder also holds `mep_aux_labels.csv` (every label with its split and p_detection_centroid),
`mep_update_report.json` (the before / after tables) and `mep_update_mock_test_summary.csv`. A fine-tuned folder also has
`mep_update_history.csv` (losses per epoch).

**Help and tests**

```bash
python scripts/train_unet.py --help
pytest -q
```

## Model folder

| File | Contents |
|---|---|
| `mep_unet_detector.weights.h5` | best weights |
| `mep_unet_normalisation.json` | per-band log-variance centre and scale for the input encoding |
| `mep_unet_model_config.json` | tiling and network settings, and the size scaling for the structure head |
| `mep_unet_training_history.csv`, `mep_unet_training_curve.png` | losses per epoch |
| `mep_calib_peaks.parquet` | matched calib peaks; the p_detection_centroid calibration is refitted from these |
| `mep_threshold.json` | the p_detection_centroid threshold, its purity and whether the target was met |
| `mep_<coadds>_test_*` | test peaks, per-coadd summary table and plots |

The weights, normalisation, model config, calib peaks and threshold are everything the detector needs at inference
time.

## Repository layout

| Path | Contents |
|---|---|
| `lsst_unet_training/config.py` | all settings (`CONFIG`) and model file names |
| `lsst_unet_training/coadd_data.py` | reading images and truth; rebuilding coadd tiles; input encoding |
| `lsst_unet_training/targets.py` | population weights and the training targets painted on each tile |
| `lsst_unet_training/tile_sequence.py` | the Keras batch loader |
| `lsst_unet_training/unet_model.py` | the network and its losses |
| `lsst_unet_training/training.py` | training, and loading a trained model |
| `lsst_unet_training/peak_detection.py` | tiled inference, peak finding and truth matching |
| `lsst_unet_training/calibration.py` | p_detection_centroid calibration and the Wilson-bound threshold |
| `lsst_unet_training/evaluation.py` | calibration and test-evaluation drivers, summary tables and plots |
| `lsst_unet_training/update.py` | updating a trained model with labelled detections on an extra coadd |
| `scripts/` | the command-line entry points above |
| `tests/` | pytest suite on a tiny synthetic dataset, run by GitHub Actions on every push |

## Licence

MIT (see `LICENSE`).
