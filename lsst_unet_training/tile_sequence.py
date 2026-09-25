"""Keras data loader over (coadd, tile) pairs of one catalogue."""

import numpy as np
import pandas as pd
import keras

from .coadd_data import encode_planes


class CoaddTileSequence(keras.utils.PyDataset):
    """Batches of network inputs (and optionally targets) for a list of (coadd, tile) pairs.

    For training, pass index=None and resample=True: every epoch each tile is paired with coadds_per_tile randomly
    chosen coadds and gets fresh noise, so one epoch is about one pass over the tiles rather than the whole
    depth x seeing grid. For validation, pass a fixed index and no augmentation for a stable early-stopping signal.
    """

    def __init__(self, store, index, normalisation, cfg, target_maker=None, shuffle=False, fresh_noise=False,
                 resample=False, coadds_per_tile=1, seed=0, workers=1):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=16)
        self.store, self.normalisation, self.target_maker = store, normalisation, target_maker
        self.batch_size, self.shuffle, self.fresh_noise = int(cfg["batch_size"]), shuffle, fresh_noise
        self.resample, self.coadds_per_tile = resample, int(coadds_per_tile)
        self.rng = np.random.default_rng(seed)
        self.tiles = store.tile_grid()
        self.index = self.random_coadds() if index is None else index.reset_index(drop=True)
        self._psf_kernels = {}
        self.on_epoch_end()

    def random_coadds(self):
        """Every tile paired with coadds_per_tile randomly chosen coadds."""
        coadds = self.store.coadds
        picks = [self.rng.integers(0, len(coadds), len(self.tiles)) for _ in range(self.coadds_per_tile)]
        return pd.concat([self.tiles.assign(combo=[coadds[i] for i in pick]) for pick in picks], ignore_index=True)

    def on_epoch_end(self):
        if self.resample:  # new random coadds (and, via fresh_noise, new noise) every epoch
            self.index = self.random_coadds()
        self.order = np.arange(len(self.index))
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.order) / self.batch_size))

    def psf_kernels(self, key):
        if key not in self._psf_kernels:
            self._psf_kernels[key] = self.store.psf_kernels(key)
        return self._psf_kernels[key]

    def __getitem__(self, batch_number):
        start = batch_number * self.batch_size
        rows = list(self.index.iloc[self.order[start:start + self.batch_size]].itertuples(index=False))
        images = []
        for row in rows:
            rng = np.random.default_rng() if self.fresh_noise else None  # None = the tile's fixed noise
            signal, variance = self.store.coadd_tile(row.combo, int(row.x0), int(row.y0), rng=rng)
            images.append(encode_planes(signal, variance, self.normalisation))
        inputs = {"image_planes": np.stack(images), "psf_kernels": np.stack([self.psf_kernels(r.combo) for r in rows])}
        if self.target_maker is None:
            return inputs
        targets = [self.target_maker.tile_targets(self.store, int(r.tile_id), int(r.x0), int(r.y0)) for r in rows]
        return inputs, {name: np.stack([t[name] for t in targets]) for name in targets[0]}
