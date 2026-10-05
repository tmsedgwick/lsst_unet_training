"""Train, calibrate and evaluate a noise- and PSF-aware U-Net galaxy detector on mock LSST coadds."""

from .config import ARTEFACTS, BANDS, CATALOGUE_STEM, CONFIG
from .evaluation import calibrate_unet, evaluate_unet
from .training import load_model, train_unet
from .update import update_threshold_on_aux, update_weights_on_aux

__all__ = ["ARTEFACTS", "BANDS", "CATALOGUE_STEM", "CONFIG", "calibrate_unet", "evaluate_unet", "load_model",
           "train_unet", "update_threshold_on_aux", "update_weights_on_aux"]
