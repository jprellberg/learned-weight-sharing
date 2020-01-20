import itertools
from pathlib import Path
from launchers.launcher import config_product, launcher


def configs():
    cfgs = [config_product(dataset='cifar100',
                           dataroot='/raid/common/cifar100',
                           num_tasks=20,
                           model='learned_sharing',
                           num_modules=20,
                           batch_size=16,
                           iterations=4000,
                           test_interval=200,
                           sgd_samples=8,
                           sgd_lr=1e-3,
                           nes_samples=8,
                           nes_lr=[1e-1, 1e-2],
                           tag=Path(__file__).stem),

            config_product(dataset='cifar100',
                           dataroot='/raid/common/cifar100',
                           num_tasks=20,
                           model=['no_sharing', 'prior_sharing', 'full_sharing'],
                           num_modules=0,
                           batch_size=16,
                           iterations=4000,
                           test_interval=200,
                           sgd_samples=1,
                           sgd_lr=1e-3,
                           tag=Path(__file__).stem),

            ###########################

            config_product(dataset='cifar100',
                           dataroot='/raid/common/cifar100',
                           num_tasks=20,
                           model='LearnedSharingResNet18',
                           num_modules=20,
                           batch_size=16,
                           iterations=20000,
                           test_interval=200,
                           sgd_samples=8,
                           sgd_lr=1e-3,
                           nes_samples=8,
                           nes_lr=[1e-1, 1e-2],
                           tag=Path(__file__).stem),

            config_product(dataset='cifar100',
                           dataroot='/raid/common/cifar100',
                           num_tasks=20,
                           model=['NoSharingResNet18', 'PriorSharingResNet18', 'FullSharingResNet18'],
                           num_modules=0,
                           batch_size=16,
                           iterations=20000,
                           test_interval=200,
                           sgd_samples=1,
                           sgd_lr=1e-3,
                           tag=Path(__file__).stem),
            ]
    return itertools.chain(*cfgs)


if __name__ == '__main__':
    launcher(configs())
