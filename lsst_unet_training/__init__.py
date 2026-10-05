"""Train, calibrate and evaluate a noise- and PSF-aware U-Net galaxy detector on mock LSST coadds."""

from .config import ARTEFACTS, BANDS, CATALOGUE_STEM, CONFIG
from .evaluation import calibrate_unet, evaluate_unet
from .real_feedback import finetune_on_real, recalibrate_on_real
from .training import load_model, train_unet

__all__ = ["ARTEFACTS", "BANDS", "CATALOGUE_STEM", "CONFIG", "calibrate_unet", "evaluate_unet", "finetune_on_real",
           "load_model", "recalibrate_on_real", "train_unet"]
