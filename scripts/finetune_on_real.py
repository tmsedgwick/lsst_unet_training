"""Fine-tune a trained U-Net on visually inspected detections in a real coadd, mixed with mock training tiles.

    python scripts/finetune_on_real.py --model-dir ~/mocks/unet --suffix real-ft1 \
        --real-coadd deep_coadd_cutout.npz --labels feedback_unet_only.json feedback_peakfinder_only.json \
        --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images

Each batch is part real tiles around train labels (loss only near a label: a galaxy centre where the reviewer said
real or marked a miss, background where they said spurious) and part mock train tiles, at a low learning rate. The
result is calibrated on the mock calib catalogue, then compared with the original model on the real test labels and
the mock test coadds. It is saved to <model-dir>_<suffix>; the original folder is never changed. Use the same split
settings as recalibrate_on_real.py to keep the same test labels.
"""

import argparse

from recalibrate_on_real import add_real_arguments, real_cfg

from lsst_unet_training import CONFIG, finetune_on_real


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_real_arguments(parser)
    parser.add_argument("--epochs", type=int, default=CONFIG["finetune_epochs"], help="default: %(default)s")
    parser.add_argument("--steps-per-epoch", type=int, default=CONFIG["finetune_steps_per_epoch"],
                        help="default: %(default)s")
    parser.add_argument("--learning-rate", type=float, default=CONFIG["finetune_learning_rate"],
                        help="default: %(default)s")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="default: %(default)s")
    parser.add_argument("--real-fraction", type=float, default=CONFIG["finetune_real_fraction"],
                        help="share of each batch that is real tiles (default: %(default)s)")
    parser.add_argument("--real-weight", type=float, default=CONFIG["finetune_real_weight"],
                        help="loss weight of a real-label pixel relative to a mock pixel (default: %(default)s)")
    parser.add_argument("--label-radius", type=float, default=CONFIG["label_radius_pix"],
                        help="loss is counted within this many pixels of a label (default: %(default)s)")
    parser.add_argument("--train", default="train", help="mock train catalogue name (default: %(default)s)")
    parser.add_argument("--valid", default="valid", help="mock valid catalogue name (default: %(default)s)")
    parser.add_argument("--calib", default="calib", help="mock calib catalogue name (default: %(default)s)")
    args = parser.parse_args()
    cfg = dict(real_cfg(args), finetune_epochs=args.epochs, finetune_steps_per_epoch=args.steps_per_epoch,
               finetune_learning_rate=args.learning_rate, batch_size=args.batch_size,
               finetune_real_fraction=args.real_fraction, finetune_real_weight=args.real_weight,
               label_radius_pix=args.label_radius)
    finetune_on_real(args.model_dir, args.real_coadd, args.labels, args.catalogue_dir, args.image_dir, args.suffix, cfg,
                     args.catalogue_stem, args.train, args.valid, args.calib, args.test, args.mock_coadds,
                     args.tile_cap)


if __name__ == "__main__":
    main()
