"""Train the U-Net on the train catalogue's coadds, early-stopping on the valid catalogue.

    python scripts/train_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images --model-dir ~/mocks/unet

--catalogue-dir and --image-dir are the output folders of mock_lsst_image_generation. The model directory receives
the best weights, input normalisation, model config, focal-loss normalisers, training history and a loss plot.

--init-from starts from a trained model's weights and input normalisation instead of from scratch (a warm start),
e.g. to retrain a model with the class-balanced weighting or on new mocks, usually with a lower learning rate:

    python scripts/train_unet.py --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images \
        --model-dir ~/unet_models/mep_unet_warm --init-from ~/unet_models/mep_unet_v3 --learning-rate 1e-4 --epochs 20
"""

import argparse
from pathlib import Path

from lsst_unet_training import ARTEFACTS, CATALOGUE_STEM, CONFIG, train_unet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalogue-dir", type=Path, required=True, help="mock catalogue folder")
    parser.add_argument("--image-dir", type=Path, required=True, help="mock image folder")
    parser.add_argument("--model-dir", type=Path, required=True, help="where to save the trained model")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="maximum epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="default: %(default)s")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="random seed (default: %(default)s)")
    parser.add_argument("--learning-rate", type=float, default=CONFIG["learning_rate"], help="default: %(default)s")
    parser.add_argument("--weighting", choices=("balanced", "rarity"), default=CONFIG["weighting"],
                        help="how positives are weighted, see config.py (default: %(default)s)")
    parser.add_argument("--init-from", type=Path, help="a trained model folder to start from (default: from scratch)")
    parser.add_argument("--train", default="train", help="training catalogue name (default: %(default)s)")
    parser.add_argument("--valid", default="valid", help="validation catalogue name (default: %(default)s)")
    parser.add_argument("--catalogue-stem", default=CATALOGUE_STEM,
                        help="catalogue file prefix, <stem>_<name>.csv (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true", help="replace a model already in --model-dir")
    args = parser.parse_args()
    if (args.model_dir / ARTEFACTS["weights"]).exists() and not args.overwrite:
        parser.error(f"{args.model_dir} already holds a trained model; pass --overwrite to replace it")
    cfg = dict(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, learning_rate=args.learning_rate,
               weighting=args.weighting)
    train_unet(args.catalogue_dir, args.image_dir, args.model_dir, cfg, args.train, args.valid, args.catalogue_stem,
               args.init_from)


if __name__ == "__main__":
    main()
