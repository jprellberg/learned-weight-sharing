from os.path import join

from torch.utils.data import Subset
from torchvision import transforms

from multitask_dataset import MultitaskProductDataset, MultitaskSequentialDataset
from torchvision.datasets import EMNIST, KMNIST


def get_tasks(dataroot, num_tasks):
    tasks = ['EMNIST/digits', 'EMNIST/letters', 'KMNIST']
    trainsets, testsets, output_sizes = [], [], []
    for task in tasks[:num_tasks]:
        trainset, testset, output_size = get_single_task(dataroot, task)
        trainsets.append(trainset)
        testsets.append(testset)
        output_sizes.append(output_size)
    mtl_train = MultitaskProductDataset(trainsets)
    mtl_test = MultitaskSequentialDataset(testsets)
    return mtl_train, mtl_test, output_sizes


def get_single_task(dataroot, task):
    tf = transforms.ToTensor()

    if task.startswith('EMNIST'):
        split = task.split('/', maxsplit=2)[1]
        dataroot = join(dataroot, 'emnist')
        tf_target = (lambda x: x - 1) if split == 'letters' else None
        output_size = 26 if split == 'letters' else 10
        trainset = EMNIST(dataroot, split=split, train=True, transform=tf, target_transform=tf_target)
        trainset = stratified_subset(trainset, trainset.targets.tolist(), 500)
        testset = EMNIST(dataroot, split=split, train=False, transform=tf, target_transform=tf_target)
    elif task == 'KMNIST':
        dataroot = join(dataroot, 'kmnist')
        output_size = 10
        trainset = KMNIST(dataroot, train=True, transform=tf)
        trainset = stratified_subset(trainset, trainset.targets.tolist(), 500)
        testset = KMNIST(dataroot, train=False, transform=tf)
    else:
        raise ValueError(task)

    return trainset, testset, output_size


def stratified_subset(dataset, labels, target_size):
    from sklearn.model_selection import train_test_split
    x = list(range(len(labels)))
    indices = train_test_split(x, train_size=target_size, stratify=labels, random_state=666)[0]
    return Subset(dataset, indices)
