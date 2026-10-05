"""Update the detection threshold from labelled detections on an auxiliary coadd (real, another mock, anything).

    python scripts/update_threshold_on_aux.py --model-dir ~/mocks/unet --suffix thr1 \
        --aux-coadd deep_coadd_cutout.npz --aux-labels feedback_unet_only.json feedback_peakfinder_only.json \
        --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images

--aux-labels takes review-tool feedback files (feedback_<category>.json from RunOnCoadd.ipynb) and / or CSVs with x, y,
label (real / spurious) and optionally weight, in the pixels of --aux-coadd. The labels are split into train and test
by image blocks; the threshold is chosen on the train labels, and the old and new thresholds are compared on the
auxiliary test labels and on the mock test coadds (default: the reference coadd). The model is copied, with the new
threshold, to <model-dir>_<suffix>; the original folder is never changed.
"""

import argparse
from pathlib import Path

from lsst_unet_training import CATALOGUE_STEM, CONFIG, update_threshold_on_aux


def add_aux_arguments(parser):
    """Arguments shared with update_weights_on_aux.py (the same split settings give the same train / test labels)."""
    parser.add_argument("--model-dir", type=Path, required=True, help="trained and calibrated model folder")
    parser.add_argument("--suffix", required=True, help="the new model goes to <model-dir>_<suffix>")
    parser.add_argument("--aux-coadd", type=Path, required=True,
                        help=".npz with signal, variance, psf_kernels and bands (e.g. deep_coadd_cutout.npz)")
    parser.add_argument("--aux-labels", type=Path, nargs="+", required=True,
                        help="feedback JSON files and / or CSVs")
    parser.add_argument("--catalogue-dir", type=Path, required=True, help="mock catalogue folder")
    parser.add_argument("--image-dir", type=Path, required=True, help="mock image folder")
    parser.add_argument("--catalogue-stem", default=CATALOGUE_STEM, help="default: %(default)s")
    parser.add_argument("--test", default="test", help="mock test catalogue name (default: %(default)s)")
    parser.add_argument("--mock-coadds", nargs="+", metavar="COADD",
                        help="mock test coadds to compare on (default: the reference coadd)")
    parser.add_argument("--tile-cap", type=int, help="score at most this many mock (coadd, tile) pairs")
    parser.add_argument("--test-fraction", type=float, default=CONFIG["aux_test_fraction"],
                        help="fraction of image blocks whose labels are held out (default: %(default)s)")
    parser.add_argument("--block-pix", type=int, default=CONFIG["aux_block_pix"],
                        help="block size of the train / test split in pixels (default: %(default)s)")
    parser.add_argument("--match-radius", type=float, default=CONFIG["aux_match_radius_pix"],
                        help="a label is detected by a peak within this many pixels (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="split seed (default: %(default)s)")


def aux_cfg(args):
    return dict(aux_test_fraction=args.test_fraction, aux_block_pix=args.block_pix,
                aux_match_radius_pix=args.match_radius, seed=args.seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_aux_arguments(parser)
    parser.add_argument("--aux-purity", type=float, default=CONFIG["aux_purity"],
                        help="weighted purity to certify on the auxiliary train labels (default: %(default)s)")
    args = parser.parse_args()
    update_threshold_on_aux(args.model_dir, args.aux_coadd, args.aux_labels, args.catalogue_dir, args.image_dir,
                     args.suffix, dict(aux_cfg(args), aux_purity=args.aux_purity),
                     args.catalogue_stem, args.test, args.mock_coadds, args.tile_cap)


if __name__ == "__main__":
    main()
