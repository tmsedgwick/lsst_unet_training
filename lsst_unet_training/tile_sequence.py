"""Keras data loader over (coadd, tile) pairs of one catalogue."""

import hashlib

import numpy as np
import pandas as pd
import keras

from .coadd_data import encode_planes
from .masking import artificial_edge, coverage, cut_away, drop_bands, sample_band_dropout


class CoaddTileSequence(keras.utils.PyDataset):
    """Batches of network inputs (and optionally targets) for a list of (coadd, tile) pairs.

    For training, pass index=None and resample=True: every epoch draws coadds_per_tile x (number of tiles) tiles,
    each with probability proportional to its tile_weight (so tiles holding rare sources come up more often), pairs
    each with a random coadd and gives it fresh noise; edge_augment_fraction of them get an artificial image edge.
    For validation, pass a fixed index and no augmentation for a stable early-stopping signal.

    With band_dropout (training the band adapter), every tile loses bands (masking.sample_band_dropout; the same ones
    each time for a fixed index) and the inputs also hold the tile with all its bands, full_planes and full_psf, from
    which the frozen backbone gives the six-band answer; the targets are made for the bands that are left.
    """

    def __init__(self, store, index, normalisation, cfg, target_maker=None, shuffle=False, fresh_noise=False,
                 resample=False, coadds_per_tile=1, seed=0, workers=1, edge_augment_fraction=0.0,
                 band_dropout=False):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=16)
        self.store, self.normalisation, self.target_maker, self.cfg = store, normalisation, target_maker, cfg
        self.batch_size, self.shuffle, self.fresh_noise = int(cfg["batch_size"]), shuffle, fresh_noise
        self.resample, self.coadds_per_tile = resample, int(coadds_per_tile)
        self.edge_augment_fraction, self.band_dropout = float(edge_augment_fraction), band_dropout
        self.rng = np.random.default_rng(seed)
        self.tiles = store.tile_grid()
        self.index = self.random_coadds() if index is None else index.reset_index(drop=True)
        self._psf_kernels = {}
        self.on_epoch_end()

    def random_coadds(self):
        """coadds_per_tile x (number of tiles) tiles drawn in proportion to their tile_weight, each with a random
        coadd."""
        n = len(self.tiles) * self.coadds_per_tile
        weight = np.asarray(self.store.tile_weight, float)[self.tiles["tile_id"].to_numpy()]
        picks = self.rng.choice(len(self.tiles), size=n, p=weight / weight.sum())
        coadds = self.store.coadds
        return self.tiles.iloc[picks].assign(combo=[coadds[i] for i in self.rng.integers(0, len(coadds), n)])

    def on_epoch_end(self):
        if self.resample:  # new random tiles and coadds (and, via fresh_noise, new noise) every epoch
            self.index = self.random_coadds().reset_index(drop=True)
        self.order = np.arange(len(self.index))
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.order) / self.batch_size))

    def psf_kernels(self, key):
        if key not in self._psf_kernels:
            self._psf_kernels[key] = self.store.psf_kernels(key)
        return self._psf_kernels[key]

    def augmentation_rng(self, row, noise_rng):
        """Random numbers for a tile's augmentation: fresh in training, fixed per (tile, coadd) otherwise."""
        if noise_rng is not None:
            return noise_rng
        digest = hashlib.md5(f"{self.store.name}|{row.combo}|{int(row.tile_id)}|augment".encode()).hexdigest()
        return np.random.default_rng(int(digest[:16], 16))

    def __getitem__(self, batch_number):
        start = batch_number * self.batch_size
        rows = list(self.index.iloc[self.order[start:start + self.batch_size]].itertuples(index=False))
        batch = dict(image_planes=[], psf_kernels=[], full_planes=[], full_psf=[])
        targets = []
        for row in rows:
            noise_rng = np.random.default_rng() if self.fresh_noise else None  # None = the tile's fixed noise
            signal, variance = self.store.coadd_tile(row.combo, int(row.x0), int(row.y0), rng=noise_rng)
            psf = self.psf_kernels(row.combo)
            rng = self.augmentation_rng(row, noise_rng)
            beyond = (artificial_edge(signal.shape[1], rng) if self.target_maker is not None
                      and rng.random() < self.edge_augment_fraction else None)
            if beyond is not None:
                signal, variance, _ = cut_away(signal, variance, {}, beyond, self.cfg["no_data_variance"])
            if self.band_dropout:
                batch["full_planes"].append(encode_planes(signal, variance, self.normalisation))
                batch["full_psf"].append(psf)
                missing, region = sample_band_dropout(rng, signal.shape[1], self.cfg)
                signal, variance, psf = drop_bands(signal, variance, psf, missing, self.cfg, region=region)
            if self.target_maker is not None:
                tile_targets = self.target_maker.tile_targets(
                    self.store, int(row.tile_id), int(row.x0), int(row.y0), row.combo,
                    coverage=coverage(variance, self.cfg) if self.band_dropout else None)
                if beyond is not None:
                    _, _, tile_targets = cut_away(signal, variance, tile_targets, beyond, self.cfg["no_data_variance"])
                targets.append(tile_targets)
            batch["image_planes"].append(encode_planes(signal, variance, self.normalisation))
            batch["psf_kernels"].append(psf)
        inputs = {name: np.stack(values) for name, values in batch.items() if values}
        if self.target_maker is None:
            return inputs
        return inputs, {name: np.stack([t[name] for t in targets]) for name in targets[0]}
