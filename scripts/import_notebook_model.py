"""Copy a model trained in the original notebook (mock_outputs/mep_unet/) into a model folder these scripts can use.

The notebook saved weights, normalisation, calib peaks and threshold but no model config; this recreates it (the size
scaling comes from the train catalogue, exactly as training computes it) and copies the rest unchanged.

    python scripts/import_notebook_model.py --notebook-model-dir .../mock_outputs/mep_unet \\
        --catalogue-dir .../FullMockExperiments --image-dir .../mock_outputs --model-dir ~/mocks/unet_notebook \\
        --catalogue-stem forward_mock_restframe_empirical_clustered
"""

import argparse
import shutil
from pathlib import Path

from lsst_unet_training import ARTEFACTS, CATALOGUE_STEM, CONFIG
from lsst_unet_training.coadd_data import CoaddStore
from lsst_unet_training.targets import TargetMaker
from lsst_unet_training.training import write_model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook-model-dir", type=Path, required=True, help="the notebook's mep_unet folder")
    parser.add_argument("--catalogue-dir", type=Path, required=True, help="catalogues the model was trained on")
    parser.add_argument("--image-dir", type=Path, required=True, help="images the model was trained on")
    parser.add_argument("--model-dir", type=Path, required=True, help="new model folder")
    parser.add_argument("--train", default="train", help="training catalogue name (default: %(default)s)")
    parser.add_argument("--catalogue-stem", default=CATALOGUE_STEM, help="default: %(default)s")
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    for artefact in ("weights", "normalisation", "history", "calib_peaks", "threshold"):
        source = args.notebook_model_dir / ARTEFACTS[artefact]
        if source.exists():
            shutil.copy2(source, args.model_dir / ARTEFACTS[artefact])
            print(f"copied {source.name}")
    maker = TargetMaker(CoaddStore(args.train, args.catalogue_dir, args.image_dir, CONFIG, args.catalogue_stem), CONFIG)
    write_model_config(args.model_dir, CONFIG, maker, args.train)
    print(f"wrote {ARTEFACTS['model_config']} -> {args.model_dir}")


if __name__ == "__main__":
    main()
