import pickle
from os.path import join

import numpy as np
from torch.utils.data import Subset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from multitask_dataset import MultitaskProductDataset, MultitaskSequentialDataset


def get_tasks(dataroot, num_tasks):
    tasks = list(range(20))
    output_sizes = [5] * 20
    trainsets, testsets = [], []
    for task in tasks[:num_tasks]:
        trainset, testset = get_single_task(dataroot, task)
        trainsets.append(trainset)
        testsets.append(testset)
    mtl_train = MultitaskProductDataset(trainsets)
    mtl_test = MultitaskSequentialDataset(testsets)
    return mtl_train, mtl_test, output_sizes


def get_single_task(dataroot, task):
    trainset = CIFAR100(dataroot, train=True, transform=transforms.ToTensor())
    trainset = filter_by_coarse_label(trainset, task)

    testset = CIFAR100(dataroot, train=False, transform=transforms.ToTensor())
    testset = filter_by_coarse_label(testset, task)

    return trainset, testset


def filter_by_coarse_label(dataset, target_label):
    # Load coarse and fine labels
    file_path = join(dataset.root, dataset.base_folder, 'train' if dataset.train else 'test')
    with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    coarse_labels = np.array(entry['coarse_labels'])
    fine_labels = np.array(entry['fine_labels'])

    # Filter dataset by coarse labels
    indices = np.flatnonzero(coarse_labels == target_label)
    dataset = Subset(dataset, indices)

    # Map remaining fine labels to a continuous range starting at 0
    contained_fine_labels = sorted(np.unique(fine_labels[indices]).tolist())
    mapping = {x: i for i, x in enumerate(contained_fine_labels)}
    dataset = TargetMapping(dataset, mapping)

    return dataset


class TargetMapping(Dataset):
    def __init__(self, wrapped, mapping):
        super().__init__()
        self.wrapped = wrapped
        self.mapping = mapping
        self.num_classes = len(mapping)

    def __len__(self):
        return len(self.wrapped)

    def __getitem__(self, item):
        x, y = self.wrapped[item]
        y = self.mapping[y]
        return x, y
