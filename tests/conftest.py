"""A tiny synthetic dataset in the layout mock_lsst_image_generation writes: four 320 x 320 px catalogues of
Gaussian galaxies, each with two coadds (1 and 10 years at ~1.1" seeing); calib and test coadds are saved."""

import json

import numpy as np
import pandas as pd
import pytest

from lsst_unet_training import BANDS, CONFIG, calibrate_unet, train_unet
from lsst_unet_training.coadd_data import add_noise, broaden

SIZE, N_GALAXIES, BASE_FWHM = 320, 60, 0.45
TINY = dict(epochs=1, batch_size=2, base_filters=8, valid_samples=4, data_workers=1)
COADDS = {"1y_fwhm110": (0.1, 1.1), "10y_fwhm110": (1.0, 1.1)}  # key: (survey fraction, r-band FWHM)


def write_catalogue(name, catalogue_dir, image_dir, rng):
    x, y = rng.uniform(5, SIZE - 5, N_GALAXIES), rng.uniform(5, SIZE - 5, N_GALAXIES)
    flux = 10 ** rng.uniform(1.5, 3.5, N_GALAXIES)  # nJy
    mag = 31.4 - 2.5 * np.log10(flux)
    pd.DataFrame(dict(x_pix=x, y_pix=y, mag_r_total=mag, sb_r_total=mag + 2.0, logM=rng.uniform(8, 11, N_GALAXIES),
                      logsSFR=rng.uniform(-12, -9, N_GALAXIES), z=rng.uniform(0.1, 2, N_GALAXIES),
                      re_total_arcsec=rng.uniform(0.2, 1.0, N_GALAXIES),
                      ellipticity_total=rng.uniform(0, 0.6, N_GALAXIES), pa_deg=rng.uniform(0, 180, N_GALAXIES),
                      )).to_csv(catalogue_dir / f"mock_catalogue_{name}.csv", index=False)
    pd.DataFrame(dict(x_pix_clump=x[:10] + 1, y_pix_clump=y[:10])).to_csv(
        catalogue_dir / f"mock_catalogue_{name}_clumps.csv", index=False)
    pd.DataFrame(dict(x_pix_tidal=x[:3] + 4, y_pix_tidal=y[:3])).to_csv(
        catalogue_dir / f"mock_catalogue_{name}_tidal.csv", index=False)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    image = np.zeros((len(BANDS), SIZE, SIZE), np.float32)
    for gx, gy, f in zip(x, y, flux):
        gaussian = np.exp(-0.5 * ((xx - gx) ** 2 + (yy - gy) ** 2) / 1.5 ** 2) / (2 * np.pi * 1.5 ** 2)
        image += (f * gaussian).astype(np.float32)
    out = image_dir / name
    out.mkdir(parents=True)
    np.save(out / "base_clean_signal.npy", image)
    (out / "base_meta.json").write_text(json.dumps(dict(split=name, origin=[0, 0], bands=BANDS, base_fwhm=BASE_FWHM,
                                                        shape=list(image.shape), ps=0.2, zp=31.4)))
    saved = name in ("calib", "test")
    combos = {}
    for key, (fraction, fwhm_r) in COADDS.items():
        fwhm = {b: fwhm_r * CONFIG["nominal_fwhm"][b] / CONFIG["nominal_fwhm"]["r"] for b in BANDS}
        n_visit = {b: max(1, round(CONFIG["visits_10yr"][b] * fraction)) for b in BANDS}
        combos[key] = dict(epoch=key.split("_")[0], survey_fraction=fraction, f_r=fwhm_r, n_visit=n_visit,
                                       fwhm_arcsec=fwhm, seed_entropy=[0])
        if saved:
            signal, variance = add_noise(broaden(image, fwhm, BASE_FWHM, CONFIG), n_visit, rng, CONFIG)
            np.save(out / f"{name}_{key}_signal.npy", signal)
            np.save(out / f"{name}_{key}_variance.npy", variance)
    (out / "coadd_manifest.json").write_text(json.dumps(dict(split=name, base_fwhm=BASE_FWHM, bands=BANDS,
                                                             materialised=saved, combos=combos)))


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("mocks")
    catalogue_dir, image_dir = root / "catalogues", root / "images"
    catalogue_dir.mkdir()
    rng = np.random.default_rng(0)
    for name in ["train", "valid", "calib", "test"]:
        write_catalogue(name, catalogue_dir, image_dir, rng)
    return catalogue_dir, image_dir


@pytest.fixture(scope="session")
def trained_model(dataset, tmp_path_factory):
    """A one-epoch tiny model, calibrated on the 10-year calib coadd."""
    catalogue_dir, image_dir = dataset
    model_dir = tmp_path_factory.mktemp("models") / "unet"
    train_unet(catalogue_dir, image_dir, model_dir, TINY)
    calibrate_unet(catalogue_dir, image_dir, model_dir, dict(reference_coadd="10y_fwhm110"))
    return model_dir
