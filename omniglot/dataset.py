import hashlib

import numpy as np
from torch.utils.data import Subset, Dataset
from torchvision import transforms
from torchvision.datasets import Omniglot

from multitask_dataset import MultitaskProductDataset, MultitaskSequentialDataset


def get_tasks(dataroot, num_tasks):
    tasks = create_tasks(dataroot, num_tasks)
    trainsets, testsets, output_sizes = [], [], []
    for alphabet in tasks:
        trainset, testset = omniglot_single_alphabet(dataroot, alphabet)
        trainsets.append(trainset)
        testsets.append(testset)
        output_sizes.append(trainset.num_classes)
    mtl_train = MultitaskProductDataset(trainsets)
    mtl_test = MultitaskSequentialDataset(testsets)
    return mtl_train, mtl_test, output_sizes


def create_tasks(dataroot, num_tasks):
    alphabets = Omniglot(dataroot, background=True)._alphabets + Omniglot(dataroot, background=False)._alphabets
    order = np.random.RandomState(666).permutation(len(alphabets))
    tasks = np.array(alphabets)[order]
    return tasks[:num_tasks]


def omniglot_single_alphabet(dataroot, alphabet):
    tf = transforms.ToTensor()

    # Load either background or evaluation dataset depending on alphabet
    dataset = Omniglot(root=dataroot, background=True, transform=tf)
    if alphabet not in dataset._alphabets:
        dataset = Omniglot(root=dataroot, background=False, transform=tf)

    # Filter to the single specified alphabet and split into train-test
    return omniglot_filter(dataset, alphabet)


def omniglot_filter(dataset, alphabet, split_ratio=0.8):
    seed = int(hashlib.md5(alphabet.encode('utf-8')).hexdigest(), 16) % (2 ** 32)
    rnd = np.random.RandomState(seed)

    # Create stratified train-test split
    train, test = [], []
    char_indices = [i for i, c in enumerate(dataset._characters) if c.startswith(alphabet)]
    for char_idx in char_indices:
        indices = np.array([i for i, (path, c) in enumerate(dataset._flat_character_images) if c == char_idx])
        order = rnd.permutation(len(indices))
        split = int(round(split_ratio * len(indices)))
        char_train = indices[order][:split]
        char_test = indices[order][split:]
        train += char_train.tolist()
        test += char_test.tolist()
        # print(alphabet, char_idx, len(char_train), len(char_test))

    # Filter dataset using indices
    trainset = Subset(dataset, train)
    testset = Subset(dataset, test)

    # Map class indices (which correspond to the indices list) to a continuous range starting at 0
    mapping = {x: i for i, x in enumerate(char_indices)}
    trainset = TargetMapping(trainset, mapping)
    testset = TargetMapping(testset, mapping)

    return trainset, testset


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
