"""Keras data loader over (coadd, tile) pairs of one catalogue."""

import numpy as np
import pandas as pd
import keras

from .coadd_data import encode_planes


def artificial_edge(full, rng):
    """(full, full) mask of the pixels beyond a random straight image edge (or two, making a corner) cutting the
    tile: what a tile at the edge of a real image would be missing."""
    beyond = np.zeros((full, full), bool)
    for _ in range(1 + int(rng.random() < 0.3)):
        side, cut = int(rng.integers(4)), int(rng.integers(full // 8, full - full // 8))
        if side == 0:
            beyond[:, :cut] = True
        elif side == 1:
            beyond[:, cut:] = True
        elif side == 2:
            beyond[:cut, :] = True
        else:
            beyond[cut:, :] = True
    return beyond


def cut_away(signal, variance, targets, beyond, no_data_variance):
    """Make the pixels in beyond "no data" (zero signal, huge variance) and drop them from every loss."""
    signal[:, beyond], variance[:, beyond] = 0.0, no_data_variance
    for value in targets.values():
        value[beyond, -1] = 0.0  # the last channel of every target is its loss weight
    return signal, variance, targets


class CoaddTileSequence(keras.utils.PyDataset):
    """Batches of network inputs (and optionally targets) for a list of (coadd, tile) pairs.

    For training, pass index=None and resample=True: every epoch draws coadds_per_tile x (number of tiles) tiles,
    each with probability proportional to its tile_weight (so tiles holding rare sources come up more often), pairs
    each with a random coadd and gives it fresh noise; edge_augment_fraction of them get an artificial image edge.
    For validation, pass a fixed index and no augmentation for a stable early-stopping signal.
    """

    def __init__(self, store, index, normalisation, cfg, target_maker=None, shuffle=False, fresh_noise=False,
                 resample=False, coadds_per_tile=1, seed=0, workers=1, edge_augment_fraction=0.0):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=16)
        self.store, self.normalisation, self.target_maker, self.cfg = store, normalisation, target_maker, cfg
        self.batch_size, self.shuffle, self.fresh_noise = int(cfg["batch_size"]), shuffle, fresh_noise
        self.resample, self.coadds_per_tile = resample, int(coadds_per_tile)
        self.edge_augment_fraction = float(edge_augment_fraction)
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

    def __getitem__(self, batch_number):
        start = batch_number * self.batch_size
        rows = list(self.index.iloc[self.order[start:start + self.batch_size]].itertuples(index=False))
        images, targets = [], []
        for row in rows:
            rng = np.random.default_rng() if self.fresh_noise else None  # None = the tile's fixed noise
            signal, variance = self.store.coadd_tile(row.combo, int(row.x0), int(row.y0), rng=rng)
            if self.target_maker is not None:
                tile_targets = self.target_maker.tile_targets(self.store, int(row.tile_id), int(row.x0), int(row.y0),
                                                              row.combo)
                augment_rng = rng or np.random.default_rng()
                if augment_rng.random() < self.edge_augment_fraction:
                    beyond = artificial_edge(signal.shape[1], augment_rng)
                    signal, variance, tile_targets = cut_away(signal, variance, tile_targets, beyond,
                                                              self.cfg["no_data_variance"])
                targets.append(tile_targets)
            images.append(encode_planes(signal, variance, self.normalisation))
        inputs = {"image_planes": np.stack(images), "psf_kernels": np.stack([self.psf_kernels(r.combo) for r in rows])}
        if self.target_maker is None:
            return inputs
        return inputs, {name: np.stack([t[name] for t in targets]) for name in targets[0]}
