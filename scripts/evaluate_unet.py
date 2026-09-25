"""Score a trained, calibrated U-Net on the sealed test catalogue.

    # quick look: two coadds at nominal seeing
    python scripts/evaluate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --coadds 1y_fwhm110 10y_fwhm110
    # every test coadd (slow)
    python scripts/evaluate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet

Writes the matched test peaks, a per-coadd summary table and plots to the model folder.
"""

import argparse
from pathlib import Path

from lsst_unet_training import CATALOGUE_STEM, evaluate_unet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalogue-dir", type=Path, required=True, help="mock catalogue folder")
    parser.add_argument("--image-dir", type=Path, required=True, help="mock image folder")
    parser.add_argument("--model-dir", type=Path, required=True, help="trained and calibrated model folder")
    parser.add_argument("--coadds", nargs="+", metavar="COADD",
                        help="test coadds to score, e.g. 10y_fwhm110 (default: all in the test manifest)")
    parser.add_argument("--tile-cap", type=int, help="score at most this many (coadd, tile) pairs, for a fast preview")
    parser.add_argument("--test", default="test", help="test catalogue name (default: %(default)s)")
    parser.add_argument("--catalogue-stem", default=CATALOGUE_STEM, help="default: %(default)s")
    args = parser.parse_args()
    evaluate_unet(args.catalogue_dir, args.image_dir, args.model_dir, args.coadds, test_name=args.test,
                  stem=args.catalogue_stem, tile_cap=args.tile_cap)


if __name__ == "__main__":
    main()
