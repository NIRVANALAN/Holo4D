# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

from torch.utils.data.distributed import DistributedSampler
from pdb import set_trace as st
import numpy as np
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.base.batched_sampler import (
    BatchedRandomSampler,
    CustomRandomSampler,
)
import torch


class EasyDataset:
    """a dataset that you can easily resize and combine.
    Examples:
    ---------
        2 * dataset ==> duplicate each element 2x

        10 @ dataset ==> set the size to 10 (random sampling, duplicates if necessary)

        dataset1 + dataset2 ==> concatenate datasets
    """

    def __add__(self, other):
        return CatDataset([self, other])

    def __rmul__(self, factor):
        return MulDataset(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset(factor, self)

    def set_epoch(self, epoch):
        pass  # nothing to do by default

    def make_sampler(self,
                     batch_size,
                     shuffle=True,
                     drop_last=True,
                     world_size=1,
                     rank=0,
                     fixed_length=False):
        if not (shuffle):
            raise NotImplementedError()  # cannot deal yet
        num_of_aspect_ratios = len(self._resolutions)
        num_of_views = self.num_views
        # st()

        sampler = CustomRandomSampler(
            self,
            batch_size,
            num_of_aspect_ratios,
            4 if not fixed_length else num_of_views,
            # 2 if not fixed_length else num_of_views,
            # 5 if not fixed_length else num_of_views,
            # 4 if not fixed_length else num_of_views,
            # 3 if not fixed_length else num_of_views,
            num_of_views,
            world_size,
            warmup=1,
            drop_last=drop_last,
        )

        if world_size > 1:  # modified according to gpt
            # print('world size: ', world_size, 'global rank: ', rank)

            # 1️⃣ Create DistributedSampler
            distributed_sampler = DistributedSampler(
                self,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,  # This ensures dataset is shuffled across GPUs
                drop_last=drop_last)

            # 2️⃣ Pass DistributedSampler to CustomRandomSampler
            # This allows regenerating indices for each epoch
            sampler.set_distributed_sampler(distributed_sampler)

        return BatchedRandomSampler(sampler, batch_size, drop_last)


class MulDataset(EasyDataset):
    """Artifically augmenting the size of a dataset."""

    multiplicator: int

    def __init__(self, multiplicator, dataset):
        assert isinstance(multiplicator, int) and multiplicator > 0
        self.multiplicator = multiplicator
        self.dataset = dataset

    def __len__(self):
        return self.multiplicator * len(self.dataset)

    def __repr__(self):
        return f"{self.multiplicator}*{repr(self.dataset)}"

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, other, another = idx
            return self.dataset[idx // self.multiplicator, other, another]
        else:
            return self.dataset[idx // self.multiplicator]

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views


class ResizedDataset(EasyDataset):
    """Artifically changing the size of a dataset."""

    new_size: int

    def __init__(self, new_size, dataset):
        assert isinstance(new_size, int) and new_size > 0
        self.new_size = new_size
        self.dataset = dataset

        # self.set_epoch(0) # hard coded

    def __len__(self):
        return self.new_size

    def __repr__(self):
        size_str = str(self.new_size)
        for i in range((len(size_str) - 1) // 3):
            sep = -4 * i - 3
            size_str = size_str[:sep] + "_" + size_str[sep:]
        return f"{size_str} @ {repr(self.dataset)}"

    def set_epoch(self, epoch):
        base_len = len(self.dataset)
        if base_len <= 0:
            details = []
            for attr in ("ROOT", "root", "split", "num_views"):
                if hasattr(self.dataset, attr):
                    details.append(f"{attr}={getattr(self.dataset, attr)!r}")
            details_str = (", " + ", ".join(details)) if details else ""
            raise ValueError(
                f"Cannot sample from an empty dataset inside `{repr(self)}`: "
                f"underlying dataset has len=0{details_str}. "
                "This usually means the dataset ROOT/split has no valid sequences for the requested num_views "
                "(or the ROOT path is missing). Remove it from the YAML, fix ROOT, or lower num_views."
            )

        # this random shuffle only depends on the epoch
        rng = np.random.default_rng(seed=epoch + 777)

        # shuffle all indices
        perm = rng.permutation(base_len)

        # rotary extension until target size is met
        # st()
        shuffled_idxs = np.concatenate(
            [perm] * (1 + (len(self) - 1) // base_len))
        self._idxs_mapping = shuffled_idxs[:self.new_size]

        assert len(self._idxs_mapping) == self.new_size

    def __getitem__(self, idx):
        assert hasattr(
            self, "_idxs_mapping"
        ), "You need to call dataset.set_epoch() to use ResizedDataset.__getitem__()"
        if isinstance(idx, tuple):
            idx, other, another = idx
            return self.dataset[self._idxs_mapping[idx], other, another]
        else:
            return self.dataset[self._idxs_mapping[idx]]

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views


class TakeDataset(EasyDataset):
    """
    Deterministic subset wrapper: take the first `n` items of an underlying dataset.

    Designed for overfitting/debug runs where you want the same small set of samples
    across epochs (even if the wrapped dataset supports/uses `set_epoch()`).
    """

    def __init__(self, n, dataset, fixed_epoch: int = 0):
        assert isinstance(n, int) and n > 0
        self.n = int(n)
        self.dataset = dataset
        self.fixed_epoch = int(fixed_epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.fixed_epoch)

    def __len__(self):
        return min(self.n, len(self.dataset))

    def __repr__(self):
        return f"TakeDataset(n={self.n}, fixed_epoch={self.fixed_epoch}, dataset={repr(self.dataset)})"

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.fixed_epoch)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, other, another = idx
            return self.dataset[int(idx), other, another]
        return self.dataset[int(idx)]

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views


class CatDataset(EasyDataset):
    """Concatenation of several datasets"""

    def __init__(self, datasets):
        for dataset in datasets:
            assert isinstance(dataset, EasyDataset)
        self.datasets = datasets
        self._cum_sizes = np.cumsum([len(dataset) for dataset in datasets])

    def __len__(self):
        return self._cum_sizes[-1]

    def __repr__(self):
        # remove uselessly long transform
        return " + ".join(
            repr(dataset).replace(
                ",transform=Compose( ToTensor() Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))",
                "",
            ) for dataset in self.datasets)

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        other = None
        if isinstance(idx, tuple):
            idx, other, another = idx

        if not (0 <= idx < len(self)):
            raise IndexError()

        db_idx = np.searchsorted(self._cum_sizes, idx, "right")
        dataset = self.datasets[db_idx]
        new_idx = idx - (self._cum_sizes[db_idx - 1] if db_idx > 0 else 0)

        if other is not None and another is not None:
            new_idx = (new_idx, other, another)
        return dataset[new_idx]

    @property
    def _resolutions(self):
        def _normalize_res_list(res_list):
            norm = []
            for r in res_list:
                # BaseMultiViewDataset style: (w, h)
                if isinstance(r, (list, tuple)) and len(r) == 2 and all(isinstance(x, int) for x in r):
                    norm.append((int(r[0]), int(r[1])))
                    continue
                # BaseStereoViewDataset style: ((w, h), num_views)
                if (
                    isinstance(r, (list, tuple))
                    and len(r) == 2
                    and isinstance(r[0], (list, tuple))
                    and len(r[0]) == 2
                    and all(isinstance(x, int) for x in r[0])
                ):
                    norm.append((int(r[0][0]), int(r[0][1])))
                    continue
                raise AssertionError(f"Unsupported resolution format in CatDataset: {r!r}")
            return tuple(norm)

        resolutions = self.datasets[0]._resolutions
        norm_ref = _normalize_res_list(resolutions)
        for dataset in self.datasets[1:]:
            norm_cur = _normalize_res_list(dataset._resolutions)
            assert norm_cur == norm_ref, (
                f"Resolution mismatch across concatenated datasets: "
                f"{dataset._resolutions!r} vs {resolutions!r}"
            )
        return resolutions

    @property
    def num_views(self):
        num_views = self.datasets[0].num_views
        for dataset in self.datasets[1:]:
            assert dataset.num_views == num_views
        return num_views
