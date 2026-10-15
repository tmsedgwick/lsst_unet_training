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
total). For every band (ugrizy) the network sees three planes: `arcsinh(S/N / 3)`, where S/N is signal / √variance,
the log variance, normalised per band, and `arcsinh(S/N / 100)`. Working in S/N rather than flux makes a 1-month and a
10-year image look alike to the network. The first stretch shows faint sources; the second, gentler one keeps the
shape of bright, extended galaxies, whose cores the first flattens (`snr_stretches`; `mep_unet_v1` has only the
first). The PSF is given separately, as one 25 × 25 unit-sum PSF stamp per band.

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
| `sfregion_map` | probability that a pixel holds detectable light of a star-forming region | focal (0.2) |
| `tidal_map` | probability that a pixel holds detectable light of a tidal stream or shell | focal (0.2) |
| `spike_map` | probability that a pixel lies on a diffraction spike | focal (0.2) |
| `extended_heatmap` | probability of the centre of a resolved galaxy; every resolved class weighs the same at each S/N | focal (1.0) |
| `centroid_offset` | sub-pixel offset from the peak pixel to the true centre | Huber at centres (0.25) |
| `source_structure` | log size, axis ratio, sin 2PA, cos 2PA | Huber at centres (0.10) |

`detection_heatmap` reads the decoder features and every map above except `extended_heatmap`, so it can add stars to the galaxies and reject
peaks the maps say are a star-forming region, tidal feature or spike. It starts as an exact copy of the galaxy map
and learns its correction, so the heads can also be added to an already trained model without changing its
detections (see "Update a trained model" below).

**Targets.** The centre-heatmap loss is the CenterNet-style focal loss: each true centre is a positive, pixels near a
centre are down-weighted negatives, and the halo is ignored. A galaxy's centre peak widens with its size (σ = 0.25 Re,
1.5 to 6 pixels), so an extended galaxy is taught a broad centre rather than a pin-point one. A map head's truth is
where its phenomenon's own light (saved alone by the image generator) is detectable in that coadd: smoothed over 1.5
pixels and combined over bands, at S/N ≥ 2. Heads whose truth the mocks lack (mocks made without stars, or without
these light images) get no loss, rather than being taught that the thing never occurs.

**Only what can be seen is a centre.** Whether a source can be seen in a coadd is judged by its size-matched S/N: its
flux over the noise of a filter matched to its PSF-convolved size, combined over the bands. This is the image's
surface-brightness limit at the source's own size. At S/N ≥ 5 a source is a positive. Below S/N 2 nothing of it shows
in the noise, so it is background, like empty sky: teaching it as a centre would teach the network to fire on noise.
Between 2 and 5 whether it shows depends on the noise, so it is neither, with no loss around it. Its light stays in
the image either way.

**Weights: class-balanced, so completeness follows S/N.** The aim is a detector that behaves like a more intelligent
peak finder: a source's chance of being found should depend on its signal-to-noise, not on what kind of galaxy it is.
With `weighting="balanced"` (the default), the galaxy positives of the train catalogue in each coadd fall in cells of
(class, S/N bin). The classes are unresolved galaxies, each added population (BCGs, UDGs, extended dIrrs, almost-dark
galaxies) and resolved galaxies by Hubble class; the S/N bins are 5–10, 10–20, 20–50 and above 50. Within each S/N bin
every class is raised to the count of the largest class (weight = largest count / own count, at most 200), so at every
S/N each class weighs as much as the commonest, while the S/N bins keep their natural proportions and stars keep
weight 1. Three more things make the weights work as intended:
- *Stratified sampling.* Tiles are drawn in proportion to their positives' total weight (up to 10 times the average)
  and each draw's loss is scaled down by the same factor, so rare classes come up often without changing the expected
  loss.
- *Fixed focal-loss normalisers.* The focal loss is usually divided by each batch's summed positive weight, which
  would divide an up-weighted batch straight back down. Instead each centre head's loss is divided by a constant, its
  average number of positives per batch, measured on 60 batches before training (`focal_normaliser_batches`).
- *Footprint weighting.* Raising a class's weight also raises how much it pays to fire on its look-alikes (its own
  star-forming knots and disc). So every pixel within 2 Re of a galaxy of weight w counts √w times in the centre
  heads' loss (`footprint_negatives`, `footprint_power=0.5`), and its centre still counts w in all.

`weighting="rarity"`, the scheme of earlier models, weights a galaxy by its rarity in population (surface brightness,
stellar mass, redshift, sSFR) times its rarity in appearance (r magnitude, size). The model has 3.6 million
parameters.

**Training.** Adam (learning rate 2 × 10⁻⁴, gradient clipping), batch size 4, up to 40 epochs. The best weights
(lowest validation loss of the detection map) are kept; training stops after 6 epochs without improvement, and the learning rate
halves after 2. Training can start from a trained model instead of from scratch (`--init-from`): layers are matched
by name, an input convolution that gained planes (a new stretch) keeps its old weights with zeros for the new ones,
and a new `extended_heatmap` head starts as a copy of the galaxy head, so the warm-started model first computes
exactly what the old one did. Training tiles are read at random from the (memory-mapped) mock images, so keep large mocks on a fast
local disk: a 14000 × 14000 pixel training frame (19 GB with its component images) trained about ten times slower
from a USB drive than from the internal disk. A third of the training tiles get an artificial image edge (one side or a corner turned to
"no data"), so sources near real image edges are learned too.

**Detection and calibration.** Detections are the 3 × 3 local maxima of the detection heatmap above 0.03, refined by
the predicted offset; each also carries the galaxy and star heatmaps' values (`galaxy_score`, `star_score`), which say
what kind of source it is. On the calib catalogue, peaks within 3 pixels of a true galaxy or star are labelled real,
and isotonic regression maps each raw score to `p_detection_centroid`, the chance that a peak with that score is the
centre of a real source. The threshold is the lowest `p_detection_centroid` at which the purity of everything above
it is at least 99%, taken on the Wilson lower bound (≈95% confidence). It is fixed once, on the 10-year,
nominal-seeing calib coadd; shallower or blurrier images then lose completeness naturally at the same threshold. The
network's raw scores need not mean the same with fewer bands, so each band set gets its own calibration (all 63 for a
model with a band adapter), from the calib coadd with the other bands removed. The threshold is one cut on
`p_detection_centroid`, chosen with all bands, for every band set: with fewer bands the model is less sure, its peaks
get lower `p_detection_centroid`, and fewer pass. Calibration also writes `mep_calibration_by_band_set.png` and
`.csv`: each set's calibration curve, its purity at the threshold, and its largest difference from the all-band
calibration.

**Evaluation.** On the test catalogue, for each coadd: purity, and completeness as a function of r magnitude relative
to that image's own 5σ point-source depth, so a 1-month and a 10-year image are compared on equal terms; completeness
of extended galaxies (Re ≥ 2″) and of stars brighter than the limit; and completeness against galaxy size. It also
gives the completeness of each class of source brighter than the limit, as a table and a bar chart (one bar per coadd,
drawn for up to 8 coadds): stars; unresolved galaxies and resolved ones by size (resolved means Re reaches the
coadd's r-band PSF FWHM); BCGs, UDGs, extended dIrrs and almost-dark galaxies; and the resolved galaxies of the main
population by Hubble type (ellipticals, lenticulars, spirals, irregulars).

**Model config.** A model's config gives its heads and its edge padding, which decide how the network is built and
fed. Besides the heads above, `sfregion_heatmap` and `tidal_heatmap` (centres of the catalogued star-forming regions
and tidal blobs) can be chosen, and `edge_padding="reflect"` fills the area beyond the image edge with a mirror image
instead of "no data". `update_weights_on_aux.py` gives a model the current heads it lacks.

**Missing bands.** A band without data is fed as "no data", like the area beyond an image edge. A model with a band
adapter (`band_adapter=True`) detects well without some bands while giving exactly the same results as without the
adapter when all six are present:
- The adapter is a small network (about 90 thousand parameters) that reads which bands have data at each pixel and
  corrects the backbone's features after every FiLM layer by `features + m × (features × γ + β)`. Here m is the
  fraction of bands missing where some, but not all, have data, so m = 0 wherever all bands are present (and beyond
  an image edge, where none are) and the correction vanishes exactly.
- It is trained after the backbone (`scripts/train_band_adapter.py`), with the backbone frozen, on tiles with bands
  dropped. Every one of the 63 band combinations is shown: the number of bands kept is drawn first (5 most often,
  then 4, 3, 2, 1), then which ones, uniformly. A quarter of the tiles instead have one band covering only part of
  the tile. A positive must be detectable (S/N ≥ 5) in the bands left; a source that is not, but would be visible
  with all bands, is neither a positive nor a negative. The maps' truth only counts the bands present, and the
  adapter is also taught to reproduce the six-band maps.
- With all six bands the results are exactly the backbone's. With fewer, completeness should fall gradually: a little
  with 5 bands, more with 4, and so on. `evaluate_unet.py --band-sets all` scores every combination and summarises by
  number of bands (the mean and the worst combination).

## Published model (`mep_unet_v3`)

`mep_unet_v3`, published with [lsst_unet_detection](https://github.com/tmsedgwick/lsst_unet_detection), was trained
with this repository's defaults by successive warm starts (`--init-from`): from a model trained on the standard mock
frames, first adding the second input stretch and the extended head, then switching to the class-balanced weights
(20 epochs each at learning rate 10⁻⁴), and finally 3 epochs on a 14000 × 14000 pixel frame made with
mock_lsst_image_generation's large-frame script, which holds about 500 UDGs, 480 extended dIrrs, 260 almost-dark
galaxies and 27 BCGs at realistic densities. Completeness above 5σ at 99% purity on a held-out 10-year mock coadd
(number of sources in brackets):

| | `mep_unet_v1` | `mep_unet_v3` |
|---|---|---|
| all galaxies (39,688) | 0.62 | 0.66 |
| unresolved galaxies (37,224) | 0.64 | 0.68 |
| resolved galaxies (2,464) | 0.26 | 0.37 |
| resolved, S/N > 50 (629) | 0.48 | 0.72 |
| UDGs (59) | 0.08 | 0.46 |
| extended dIrrs (74) | 0.12 | 0.58 |
| resolved spirals (170) | 0.56 | 0.68 |
| stars (445) | 0.70 | 0.92 |
| mean gap between a class and all sources at the same S/N | 0.35 | 0.17 |

Trained from scratch with the class-balanced weights, the network evens out the classes further but finds fewer
compact galaxies in the same training time; warm-starting from a model first trained without them gives the better
balance. Almost-dark galaxies are not found by either model.

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

# Warm start: continue from a trained model (here the published one) on your own mocks, at a lower learning rate
python scripts/train_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet_warm --init-from ~/unet_models/mep_unet_v3 --learning-rate 1e-4 --epochs 20

# The earlier rarity weighting instead of the class-balanced one, with another random seed
python scripts/train_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --weighting rarity --seed 1

# Retrain into a folder that already holds a model
python scripts/train_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --overwrite

# Catalogues generated with --numbered: choose which play train and valid
python scripts/train_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --train 1 --valid 2
```

**Calibration and evaluation options**

```bash
# A stricter purity target
python scripts/calibrate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --target-purity 0.995

# Fix the threshold on a different calib coadd, e.g. 10 years at 1.3" seeing
python scripts/calibrate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --reference-coadd 10y_fwhm130

# Completeness by class as a bar chart: a few coadds, so the bars stay readable
python scripts/evaluate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --coadds 1y_fwhm110 3y_fwhm110 10y_fwhm110 10y_fwhm150

# Score only the worst seeing at three depths
python scripts/evaluate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --coadds 1y_fwhm200 5y_fwhm200 10y_fwhm200

# Fast preview on a random 200 (coadd, tile) pairs
python scripts/evaluate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --tile-cap 200
```

**Missing bands**

```bash
# Give a trained model a band adapter and train it (the model itself is frozen); saved to ~/mocks/unet_bands
python scripts/train_band_adapter.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --suffix bands

# Calibrate every band set (63), then score every combination on the 10-year test coadd
python scripts/calibrate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet_bands
python scripts/evaluate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet_bands --coadds 10y_fwhm110 --band-sets all
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
`sfregion`, `tidal`, `bad_centroid` or `hallucination`; for a real one `star`. Reasons that name a phenomenon also
teach its map when fine-tuning (e.g. `tidal` teaches `tidal_map`, `star` teaches `star_heatmap`).

Only labels that came up in the random review order are an unbiased sample, so only they go into the statistics
(the threshold, purity, recall); clicked and missed labels are used for fine-tuning only. If you reviewed only a
sample of a category (e.g. 157 of 11,324 U-Net-only candidates), each random label is weighted by
n_candidates / n_reviewed, so that category counts in proportion to its size.

Within each category the labels are split into training and test labels by 400-pixel image blocks, fixed by the seed,
so every category has test labels and both commands hold out the same ones. If no threshold reaches the purity goal,
`update_threshold_on_aux.py` keeps the current one. `update_weights_on_aux.py` gives the model's misses 3 times the
weight of other labels, draws mock tiles with rare sources more often, adds artificial image edges, and gives the
model those of the current heads it lacks and can learn here (the mocks or the labels must hold their truth). Neither
command changes the model you start from. Each writes a new folder `<model-dir>_<suffix>` and refuses a suffix that is
already taken. Both print, before and after the update:
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

A model with a band adapter can have its threshold updated, but not its weights: the adapter was trained for the
weights the model has. Fine-tune the model the adapter was added to, then give the result a new adapter:

```bash
python scripts/update_weights_on_aux.py --model-dir ~/mocks/unet --suffix w1 --aux-coadd deep_coadd_cutout.npz --aux-labels feedback_unet_only.json feedback_peakfinder_only.json --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images
python scripts/train_band_adapter.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet_w1 --suffix bands
python scripts/calibrate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet_w1_bands
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
| `focal_normalisers.json` | the fixed focal-loss normaliser of each centre head (its positives per batch) |
| `mep_calib_peaks.parquet` | matched calib peaks of every band set (column `band_set`); the p_detection_centroid calibration is refitted from these |
| `mep_threshold.json` | the p_detection_centroid threshold, its purity and whether the target was met; per band set, its purity at that threshold |
| `mep_calibration_by_band_set.png`, `.csv` | each band set's calibration curve, purity at the threshold and difference from the all-band calibration |
| `mep_band_adapter_history.csv` | losses per epoch of the band adapter's training (models with an adapter) |
| `mep_<coadds>_test_*` | test peaks, per-coadd summary table and plots |
| `mep_<coadds>_test_by_class.csv`, `mep_<coadds>_<bands>_test_completeness_by_class.png` | completeness of each class of source, per band set and coadd, and its bar chart |

The weights, normalisation, model config, calib peaks and threshold are everything the detector needs at inference
time.

## Repository layout

| Path | Contents |
|---|---|
| `lsst_unet_training/config.py` | all settings (`CONFIG`) and model file names |
| `lsst_unet_training/coadd_data.py` | reading images and truth; rebuilding coadd tiles; input encoding |
| `lsst_unet_training/targets.py` | population weights and the training targets painted on each tile |
| `lsst_unet_training/tile_sequence.py` | the Keras batch loader |
| `lsst_unet_training/masking.py` | "no data": artificial image edges and missing bands |
| `lsst_unet_training/unet_model.py` | the network and its losses |
| `lsst_unet_training/training.py` | training (and training a band adapter), and loading a trained model |
| `lsst_unet_training/peak_detection.py` | tiled inference, peak finding and truth matching |
| `lsst_unet_training/calibration.py` | p_detection_centroid calibration and the Wilson-bound threshold |
| `lsst_unet_training/evaluation.py` | calibration and test-evaluation drivers, summary tables and plots |
| `lsst_unet_training/update.py` | updating a trained model with labelled detections on an extra coadd |
| `scripts/` | the command-line entry points above |
| `tests/` | pytest suite on a tiny synthetic dataset, run by GitHub Actions on every push |

## Licence

MIT (see `LICENSE`).
