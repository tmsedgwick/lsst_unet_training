"""Give a trained model a band adapter, so it detects well with bands missing, and train it.

    python scripts/train_band_adapter.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet --suffix bands
    python scripts/calibrate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet_bands
    python scripts/evaluate_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet_bands --coadds 10y_fwhm110 --band-sets ugrizy grizy griz gri r

The model itself is frozen: only the adapter is trained, on tiles with bands dropped, so the new model's results with
all six bands are exactly the original's. It is saved to <model-dir>_<suffix>; the original folder is never changed.
Calibration then gives each band set in the config (ugrizy, grizy, ugriz, griz, gri, gr, r) its own threshold.
"""

import argparse
from pathlib import Path

from lsst_unet_training import CATALOGUE_STEM, CONFIG, train_band_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalogue-dir", type=Path, required=True, help="mock catalogue folder")
    parser.add_argument("--image-dir", type=Path, required=True, help="mock image folder")
    parser.add_argument("--model-dir", type=Path, required=True, help="trained model folder (without an adapter)")
    parser.add_argument("--suffix", required=True, help="the model with its adapter is saved to <model-dir>_<suffix>")
    parser.add_argument("--epochs", type=int, default=CONFIG["adapter_epochs"], help="default: %(default)s")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="default: %(default)s")
    parser.add_argument("--train", default="train", help="training catalogue name (default: %(default)s)")
    parser.add_argument("--valid", default="valid", help="validation catalogue name (default: %(default)s)")
    parser.add_argument("--catalogue-stem", default=CATALOGUE_STEM, help="default: %(default)s")
    args = parser.parse_args()
    train_band_adapter(args.catalogue_dir, args.image_dir, args.model_dir, args.suffix,
                       dict(adapter_epochs=args.epochs, batch_size=args.batch_size), args.train, args.valid,
                       args.catalogue_stem)


if __name__ == "__main__":
    main()
