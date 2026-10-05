# lsst_unet_training

Train, calibrate and evaluate a noise- and PSF-aware U-Net that detects galaxies in LSST-like coadds, using the mock
catalogues and images made by [mock_lsst_image_generation](https://github.com/tmsedgwick/mock_lsst_image_generation).
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
generator's PSF and noise model, so they need no disk space; calib and test coadds are read from disk.

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

**Heads.** Five 1 × 1 convolution outputs at full resolution:

| Head | Output | Loss (weight) |
|---|---|---|
| `galaxy_heatmap` | probability of a galaxy centre at each pixel; its peaks are the detections | focal (1.0) |
| `centroid_offset` | sub-pixel offset from the peak pixel to the true centre | Huber at centres (0.25) |
| `source_structure` | log size, axis ratio, sin 2PA, cos 2PA | Huber at centres (0.10) |
| `clump_heatmap` | star-forming clump centres (auxiliary) | focal (0.20) |
| `tidal_heatmap` | tidal blob centres (auxiliary) | focal (0.10) |

The heatmap loss is the CenterNet-style focal loss: each true centre is a positive, pixels near a centre are
down-weighted negatives, and the halo is ignored. Galaxies are weighted by the inverse of how common their population
is (binned in surface brightness, stellar mass, redshift and sSFR), so rare faint or high-redshift galaxies count as
much as common ones. The model has 3.6 million parameters.

**Training.** Adam (learning rate 2 × 10⁻⁴, gradient clipping), batch size 4, up to 40 epochs. The best weights
(lowest validation heatmap loss) are kept; training stops after 6 epochs without improvement, and the learning rate
halves after 2.

**Detection and calibration.** Detections are the 3 × 3 local maxima of the galaxy heatmap above 0.03, refined by the
predicted offset. On the calib catalogue, peaks within 3 pixels of a true galaxy are labelled real, and isotonic
regression maps each raw score to `p_real`, the chance a peak with that score is real. The threshold is the lowest
`p_real` at which the purity of everything above it is at least 99%, taken on the Wilson lower bound (≈95%
confidence). It is fixed once, on the 10-year, nominal-seeing calib coadd; shallower or blurrier images then lose
completeness naturally at the same threshold.

**Evaluation.** On the test catalogue, for each coadd: purity, and completeness as a function of r magnitude relative
to that image's own 5σ point-source depth, so a 1-month and a 10-year image are compared on equal terms.

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

# 2. Calibrate on the calib catalogue: p_real calibration and the 99%-purity threshold.
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

**Reuse the model trained in the original notebook**

```bash
# Copies its weights, normalisation, calib peaks and threshold, and writes the missing model config
python scripts/import_notebook_model.py --notebook-model-dir /path/to/mock_outputs/mep_unet --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet_notebook
```

**Update a trained model from auxiliary data**

The auxiliary coadd is any extra coadd with labelled detections: a real coadd inspected by eye, a different mock,
anything.
The review tool in `RunOnCoadd.ipynb` writes `feedback_<category>.json`: candidates you called real or spurious
(unsure ones are skipped) and galaxies you marked as missed, in the pixels of the coadd you reviewed
(`deep_coadd_cutout.npz`). A CSV with `x`, `y`, `label` (real / spurious) and optional `weight` works too. Each
reviewed candidate is weighted by n / n_reviewed of its category, so a sampled category (e.g. 57 of 11,324 U-Net-only
candidates) counts as much as it should against a fully reviewed one. The labels are split into train and test by
400-pixel image blocks, held fixed by the seed, so both commands use the same test labels.

Neither command changes the model you start from: each writes a new folder `<model-dir>_<suffix>` and refuses a
suffix that is already taken. Both print, before and after, the auxiliary train and test labels found
(weighted recall and purity), the number of detections over the whole auxiliary coadd, and purity / completeness on
the mock test coadds (default: the reference coadd; `--mock-coadds` for more, `--tile-cap` for a quick look). The labels only describe
the reviewed candidates: a lower threshold also lets through unreviewed peaks elsewhere, which the whole-image
detection count shows.

```bash
# Threshold only: the lowest p_real whose weighted purity on the auxiliary train labels is certified at --aux-purity.
python scripts/update_threshold_on_aux.py --model-dir ~/mocks/unet --suffix thr1 --aux-coadd deep_coadd_cutout.npz --aux-labels feedback_unet_only.json feedback_peakfinder_only.json --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images

# Weights: batches half auxiliary tiles (loss only within 8 px of a label) and half mock train tiles, at learning
# rate 2e-5; then calibrate on the mock calib catalogue and compare with the original model.
python scripts/update_weights_on_aux.py --model-dir ~/mocks/unet --suffix w1 --aux-coadd deep_coadd_cutout.npz --aux-labels feedback_unet_only.json feedback_peakfinder_only.json --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images

# Then tune the updated model's threshold on the same auxiliary labels
python scripts/update_threshold_on_aux.py --model-dir ~/mocks/unet_w1 --suffix thr1 --aux-coadd deep_coadd_cutout.npz --aux-labels feedback_unet_only.json feedback_peakfinder_only.json --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images
```

Each new folder also holds `mep_aux_labels.csv` (every label with its split and p_real), `mep_update_report.json`
and `mep_update_mock_test_summary.csv`; a weight update adds `mep_update_history.csv`.

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
| `mep_calib_peaks.parquet` | matched calib peaks; the p_real calibration is refitted from these |
| `mep_threshold.json` | the p_real threshold, its purity and whether the target was met |
| `mep_<coadds>_test_*` | test peaks, per-coadd summary table and plots |

The weights, normalisation, calib peaks and threshold are everything the detector needs at inference time, under
the same names the existing `mep_unet_infer.py` loads.

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
| `lsst_unet_training/calibration.py` | p_real calibration and the Wilson-bound threshold |
| `lsst_unet_training/evaluation.py` | calibration and test-evaluation drivers, summary tables and plots |
| `lsst_unet_training/update.py` | updating a trained model from auxiliary data: threshold and weights |
| `scripts/` | the command-line entry points above |
| `tests/` | pytest suite on a tiny synthetic dataset, run by GitHub Actions on every push |

The code reproduces the original notebook: loading its trained weights gives the same calibration threshold and the
same test purity and completeness.

## Licence

MIT (see `LICENSE`).
