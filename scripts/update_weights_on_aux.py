"""Update (fine-tune) the weights on labelled detections in an auxiliary coadd, mixed with mock training tiles.

    python scripts/update_weights_on_aux.py --model-dir ~/mocks/unet --suffix w1 \
        --aux-coadd deep_coadd_cutout.npz --aux-labels feedback_unet_only.json feedback_peakfinder_only.json \
        --catalogue-dir ~/mocks/catalogues --image-dir ~/mocks/images

Each batch is part auxiliary tiles around train labels (loss only near a label: a galaxy centre where the reviewer said
real or marked a miss, background where they said spurious) and part mock train tiles, at a low learning rate. The
result is calibrated on the mock calib catalogue, then compared with the original model on the auxiliary test labels
and the mock test coadds. It is saved to <model-dir>_<suffix>; the original folder is never changed. Use the same split
settings as update_threshold_on_aux.py to keep the same test labels.
"""

import argparse

from update_threshold_on_aux import add_aux_arguments, aux_cfg

from lsst_unet_training import CONFIG, update_weights_on_aux


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_aux_arguments(parser)
    parser.add_argument("--epochs", type=int, default=CONFIG["update_epochs"], help="default: %(default)s")
    parser.add_argument("--steps-per-epoch", type=int, default=CONFIG["update_steps_per_epoch"],
                        help="default: %(default)s")
    parser.add_argument("--learning-rate", type=float, default=CONFIG["update_learning_rate"],
                        help="default: %(default)s")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="default: %(default)s")
    parser.add_argument("--aux-fraction", type=float, default=CONFIG["update_aux_fraction"],
                        help="share of each batch that is auxiliary tiles (default: %(default)s)")
    parser.add_argument("--aux-weight", type=float, default=CONFIG["update_aux_weight"],
                        help="loss weight of an auxiliary-label pixel relative to a mock pixel (default: %(default)s)")
    parser.add_argument("--label-radius", type=float, default=CONFIG["label_radius_pix"],
                        help="loss is counted within this many pixels of a label (default: %(default)s)")
    parser.add_argument("--train", default="train", help="mock train catalogue name (default: %(default)s)")
    parser.add_argument("--valid", default="valid", help="mock valid catalogue name (default: %(default)s)")
    parser.add_argument("--calib", default="calib", help="mock calib catalogue name (default: %(default)s)")
    args = parser.parse_args()
    cfg = dict(aux_cfg(args), update_epochs=args.epochs, update_steps_per_epoch=args.steps_per_epoch,
               update_learning_rate=args.learning_rate, batch_size=args.batch_size,
               update_aux_fraction=args.aux_fraction, update_aux_weight=args.aux_weight,
               label_radius_pix=args.label_radius)
    update_weights_on_aux(args.model_dir, args.aux_coadd, args.aux_labels, args.catalogue_dir, args.image_dir,
                   args.suffix, cfg, args.catalogue_stem, args.train, args.valid, args.calib, args.test,
                   args.mock_coadds, args.tile_cap)


if __name__ == "__main__":
    main()
