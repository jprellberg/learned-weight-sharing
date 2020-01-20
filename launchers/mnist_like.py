import itertools
from pathlib import Path
from launchers.launcher import config_product, launcher


def configs():
    cfgs = [config_product(dataset='mnist_like',
                           dataroot='/raid/common/',
                           num_tasks=3,
                           model='learned_sharing',
                           num_modules=3,
                           batch_size=16,
                           iterations=5000,
                           test_interval=200,
                           sgd_samples=8,
                           sgd_lr=1e-3,
                           nes_samples=8,
                           nes_lr=1e-2,
                           tag=Path(__file__).stem),

            config_product(dataset='mnist_like',
                           dataroot='/raid/common/',
                           num_tasks=3,
                           model=['no_sharing', 'prior_sharing', 'full_sharing'],
                           num_modules=0,
                           batch_size=16,
                           iterations=5000,
                           test_interval=200,
                           sgd_samples=1,
                           sgd_lr=1e-3,
                           tag=Path(__file__).stem),
            ]
    return itertools.chain(*cfgs)


if __name__ == '__main__':
    launcher(configs())
