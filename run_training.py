import argparse
import importlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from trainer import Trainer, SameDataOptTrainer
from utils import set_seeds, unique_string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--num-tasks', type=int, required=True)

    parser.add_argument('--model', required=True)
    parser.add_argument('--num-modules', type=int)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--test-interval', type=int, required=True)

    parser.add_argument('--sgd-samples', type=int, required=True)
    parser.add_argument('--sgd-lr', type=float, required=True)

    parser.add_argument('--nes-samples', type=int)
    parser.add_argument('--nes-lr', type=float)

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=int(np.random.randint(0, 100000)))
    parser.add_argument('--out', default='results')
    parser.add_argument('--uid', default=unique_string())
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seeds(args.seed)
    print(args)

    # Load correct dataset and associated model modules
    dataset_module = importlib.import_module(f'{args.dataset}.dataset')
    model_module = importlib.import_module(f'{args.dataset}.model')

    # Load dataset
    mtl_train, mtl_test, output_sizes = dataset_module.get_tasks(args.dataroot, args.num_tasks)

    # Create model
    model = getattr(model_module, args.model)(output_sizes, args.num_modules)
    model = model.to(args.device)

    # Choose correct loss and metric functions for the dataset and create the appropriate trainer
    if args.dataset == 'omniglot' or args.dataset == 'mnist_like' or args.dataset == 'cifar100':
        loss_fn = F.cross_entropy
        metric_fn = accuracy
    else:
        raise ValueError()

    # Create the appropriate trainer
    if args.dataset == 'omniglot' or args.dataset == 'mnist_like' or args.dataset == 'cifar100':
        trainer = Trainer(model, mtl_train, mtl_test, loss_fn, metric_fn, args)
    else:
        raise ValueError()

    # Training loop
    with trange(args.iterations + 1) as progbar:
        for i in progbar:
            trainer.trainstep(i)
            if i % args.test_interval == 0:
                kwargs = trainer.evaluate(i)
                progbar.set_postfix(**kwargs)
                trainer.save_logfile(f'{args.out}/{args.tag}-{args.model}-{args.uid}')


def accuracy(logits, y, *args, **kwargs):
    return (logits.argmax(1) == y).to(torch.float32)


if __name__ == '__main__':
    main()
