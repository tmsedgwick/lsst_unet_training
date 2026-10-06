"""Choose a new detection threshold from labelled detections on an auxiliary coadd; the network is unchanged.

    python scripts/update_threshold_on_aux.py --model-dir ~/mocks/unet --suffix thr1 \
        --aux-coadd deep_coadd_cutout.npz --aux-labels feedback_unet_only.json feedback_peakfinder_only.json \
        --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images

The auxiliary coadd is any extra image with labelled detections (a real coadd inspected by eye, a different mock,
...), stored as an .npz with signal, variance, psf_kernels and bands. --aux-labels takes review JSON files
(feedback_<category>.json, format in lsst_unet_training/update.py) and / or CSVs with columns x, y, label ("real" for
a genuine source, "spurious" for an artefact) and optionally weight, in the pixel coordinates of --aux-coadd.

The labels are split into a training and a test half by image blocks, within each category. The new threshold is the
lowest p_detection_centroid at which the random-order training labels above it reach --aux-purity; if none does, the
old threshold is kept. The old and new thresholds are then compared on the test labels and on the mock test coadds
(default: the reference coadd). The model is copied, with the new threshold, to <model-dir>_<suffix>; the original
folder is never changed.
"""

import argparse
from pathlib import Path

from lsst_unet_training import CATALOGUE_STEM, CONFIG, update_threshold_on_aux


def add_aux_arguments(parser):
    """Arguments shared with update_weights_on_aux.py. Using the same split settings in both holds out the same test
    labels."""
    parser.add_argument("--model-dir", type=Path, required=True, help="trained and calibrated model folder")
    parser.add_argument("--suffix", required=True, help="the updated model is saved to <model-dir>_<suffix>")
    parser.add_argument("--aux-coadd", type=Path, required=True,
                        help=".npz with signal, variance, psf_kernels and bands (e.g. deep_coadd_cutout.npz)")
    parser.add_argument("--aux-labels", type=Path, nargs="+", required=True,
                        help="review JSON files and / or CSVs of labelled detections on --aux-coadd")
    parser.add_argument("--catalogue-dir", type=Path, required=True, help="mock catalogue folder")
    parser.add_argument("--image-dir", type=Path, required=True, help="mock image folder")
    parser.add_argument("--catalogue-stem", default=CATALOGUE_STEM, help="default: %(default)s")
    parser.add_argument("--test", default="test", help="mock test catalogue name (default: %(default)s)")
    parser.add_argument("--mock-coadds", nargs="+", metavar="COADD",
                        help="mock test coadds to compare on (default: the reference coadd)")
    parser.add_argument("--tile-cap", type=int, help="score at most this many mock (coadd, tile) pairs, for speed")
    parser.add_argument("--test-fraction", type=float, default=CONFIG["aux_test_fraction"],
                        help="fraction of image blocks whose labels are held out for testing (default: %(default)s)")
    parser.add_argument("--block-pix", type=int, default=CONFIG["aux_block_pix"],
                        help="side of the image blocks used for the train / test split, in pixels "
                             "(default: %(default)s)")
    parser.add_argument("--match-radius", type=float, default=CONFIG["aux_match_radius_pix"],
                        help="a label counts as detected if a peak above the threshold lies within this many pixels "
                             "(default: %(default)s)")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="seed of the split (default: %(default)s)")


def aux_cfg(args):
    """The config overrides given by the shared arguments."""
    return dict(aux_test_fraction=args.test_fraction, aux_block_pix=args.block_pix,
                aux_match_radius_pix=args.match_radius, seed=args.seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_aux_arguments(parser)
    parser.add_argument("--aux-purity", type=float, default=CONFIG["aux_purity"],
                        help="purity the training labels above the new threshold must reach, with ~95%% confidence "
                             "(default: %(default)s)")
    args = parser.parse_args()
    update_threshold_on_aux(args.model_dir, args.aux_coadd, args.aux_labels, args.catalogue_dir, args.image_dir,
                            args.suffix, dict(aux_cfg(args), aux_purity=args.aux_purity), args.catalogue_stem,
                            args.test, args.mock_coadds, args.tile_cap)


if __name__ == "__main__":
    main()
