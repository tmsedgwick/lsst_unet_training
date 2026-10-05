"""A tiny synthetic dataset in the layout mock_lsst_image_generation writes: four 320 x 320 px catalogues of
Gaussian galaxies and a few stars (one bright enough to saturate, with diffraction spikes), clumps and tidal blobs,
each with two coadds (1 and 10 years at ~1.1" seeing); calib and test coadds are saved. The light of the clumps, tidal
blobs and spikes is also saved alone, as the image generator does."""

import json

import numpy as np
import pandas as pd
import pytest

from lsst_unet_training import BANDS, CONFIG, calibrate_unet, train_unet
from lsst_unet_training.coadd_data import add_noise, broaden
from lsst_unet_training.saturation import saturate_stars

SIZE, N_GALAXIES, BASE_FWHM = 320, 60, 0.45
STARS = dict(x=[60.5, 200.2, 250.7, 150.1], y=[250.3, 80.6, 200.4, 160.9], mag_r=[13.0, 18.0, 20.0, 21.5])
TINY = dict(epochs=1, batch_size=2, base_filters=8, valid_samples=4, data_workers=1)
COADDS = {"1y_fwhm110": (0.1, 1.1), "10y_fwhm110": (1.0, 1.1)}  # key: (survey fraction, r-band FWHM)


def write_catalogue(name, catalogue_dir, image_dir, rng):
    x, y = rng.uniform(5, SIZE - 5, N_GALAXIES), rng.uniform(5, SIZE - 5, N_GALAXIES)
    flux = 10 ** rng.uniform(1.5, 3.5, N_GALAXIES)  # nJy
    mag = 31.4 - 2.5 * np.log10(flux)
    galaxies = pd.DataFrame(dict(
        type="star_forming", x_pix=x, y_pix=y, mag_r_total=mag, sb_r_total=mag + 2.0,
        logM=rng.uniform(8, 11, N_GALAXIES), logsSFR=rng.uniform(-12, -9, N_GALAXIES),
        z=rng.uniform(0.1, 2, N_GALAXIES),
        re_total_arcsec=rng.uniform(0.2, 1.0, N_GALAXIES), ellipticity_total=rng.uniform(0, 0.6, N_GALAXIES),
        pa_deg=rng.uniform(0, 180, N_GALAXIES)))
    galaxies.loc[:2, "re_total_arcsec"] = 3.0  # a few extended galaxies
    stars = pd.DataFrame(dict(type="star", x_pix=STARS["x"], y_pix=STARS["y"], mag_r_total=STARS["mag_r"],
                              saturation_r=8000.0))
    pd.concat([galaxies, stars], ignore_index=True).to_csv(catalogue_dir / f"mock_catalogue_{name}.csv", index=False)
    pd.DataFrame(dict(x_pix_clump=x[:10] + 1, y_pix_clump=y[:10])).to_csv(
        catalogue_dir / f"mock_catalogue_{name}_clumps.csv", index=False)
    pd.DataFrame(dict(x_pix_tidal=x[:3] + 4, y_pix_tidal=y[:3])).to_csv(
        catalogue_dir / f"mock_catalogue_{name}_tidal.csv", index=False)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]

    def blob(cx, cy, total, sigma=1.5):
        return np.repeat((total * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma ** 2)
                          / (2 * np.pi * sigma ** 2))[None], len(BANDS), axis=0).astype(np.float32)

    image = sum(blob(gx, gy, f) for gx, gy, f in zip(x, y, flux))
    image += sum(blob(sx, sy, 10 ** (-0.4 * (m - 31.4)), 1.0) for sx, sy, m in zip(STARS["x"], STARS["y"],
                                                                                   STARS["mag_r"]))
    components = dict(sfregions=sum(blob(cx, cy, 300.0, 1.0) for cx, cy in zip(x[:10] + 1, y[:10])),
                      tidal=sum(blob(tx, ty, 3000.0, 4.0) for tx, ty in zip(x[:3] + 4, y[:3])),
                      spikes=np.zeros_like(image))
    bright_x, bright_y = int(STARS["x"][0]), int(STARS["y"][0])
    components["spikes"][:, bright_y - 1:bright_y + 2, :] = 50.0  # a horizontal and a vertical spike
    components["spikes"][:, :, bright_x - 1:bright_x + 2] = 50.0
    for light in components.values():
        image += light
    out = image_dir / name
    out.mkdir(parents=True)
    np.save(out / "base_clean_signal.npy", image)
    for component, light in components.items():
        np.save(out / f"base_{component}_signal.npy", light)
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
            clean = broaden(image, fwhm, BASE_FWHM, CONFIG)
            signal, variance = add_noise(clean, n_visit, rng, CONFIG)
            signal = saturate_stars(signal, clean, np.array(STARS["x"]), np.array(STARS["y"]),
                                    np.array(STARS["mag_r"]), np.full(4, 8000.0), BANDS, rng)
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
