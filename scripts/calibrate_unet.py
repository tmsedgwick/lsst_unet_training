"""Calibrate a trained U-Net: map raw scores to p_real and choose the detection threshold on the calib catalogue.

    python scripts/calibrate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet

The threshold is the lowest p_real whose purity (Wilson lower bound) meets --target-purity on the reference coadd.
"""

import argparse
from pathlib import Path

from lsst_unet_training import CATALOGUE_STEM, CONFIG, calibrate_unet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalogue-dir", type=Path, required=True, help="mock catalogue folder")
    parser.add_argument("--image-dir", type=Path, required=True, help="mock image folder")
    parser.add_argument("--model-dir", type=Path, required=True, help="trained model folder")
    parser.add_argument("--target-purity", type=float, default=CONFIG["target_purity"], help="default: %(default)s")
    parser.add_argument("--reference-coadd", default=CONFIG["reference_coadd"],
                        help="calib coadd the threshold is fixed on (default: %(default)s)")
    parser.add_argument("--calib", default="calib", help="calibration catalogue name (default: %(default)s)")
    parser.add_argument("--catalogue-stem", default=CATALOGUE_STEM, help="default: %(default)s")
    args = parser.parse_args()
    calibrate_unet(args.catalogue_dir, args.image_dir, args.model_dir,
                   dict(target_purity=args.target_purity, reference_coadd=args.reference_coadd), args.calib,
                   args.catalogue_stem)


if __name__ == "__main__":
    main()
