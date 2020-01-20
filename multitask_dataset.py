import bisect

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate


def heterogeneous_dict_collate(batch):
    if isinstance(batch[0], dict):
        keys = sum((list(dic.keys()) for dic in batch), [])
        return {key: default_collate([d[key] for d in batch if key in d]) for key in keys}
    raise TypeError


class PermutationProvider:
    def __init__(self, n):
        self.n = n
        self.idx = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.n:
            self.perm = torch.randperm(self.n)
            self.idx = 0
        val = self.perm[self.idx]
        self.idx += 1
        return val


class MultitaskProductDataset(IterableDataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.permutations = [PermutationProvider(len(ds)) for ds in self.datasets]

    def __iter__(self):
        return self

    def __next__(self):
        return {ds_idx: ds[next(perm)] for ds_idx, (ds, perm) in enumerate(zip(self.datasets, self.permutations))}


class MultitaskSequentialDataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return {dataset_idx: self.datasets[dataset_idx][sample_idx]}

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
