import torch.nn as nn

from task_routing_model import StaticTaskRouting, LearnedTaskRouting, IgnoreTaskRouting
from utils import Flatten


def learned_sharing(task_outputs, num_modules):
    nt = len(task_outputs)
    return nn.Sequential(
        LearnedTaskRouting(nt, [conv_layer(1, 32) for _ in range(num_modules)]),
        LearnedTaskRouting(nt, [conv_layer(32, 32) for _ in range(num_modules)]),
        LearnedTaskRouting(nt, [conv_layer(32, 32) for _ in range(num_modules)]),

        IgnoreTaskRouting(Flatten()),
        LearnedTaskRouting(nt, [dense_layer(288, 128) for _ in range(num_modules)]),
        StaticTaskRouting(nt, [nn.Linear(128, s) for s in task_outputs]),
    )


def no_sharing(task_outputs, num_modules):
    nt = len(task_outputs)
    return nn.Sequential(
        StaticTaskRouting(nt, [conv_layer(1, 32) for _ in task_outputs]),
        StaticTaskRouting(nt, [conv_layer(32, 32) for _ in task_outputs]),
        StaticTaskRouting(nt, [conv_layer(32, 32) for _ in task_outputs]),

        IgnoreTaskRouting(Flatten()),
        StaticTaskRouting(nt, [dense_layer(288, 128) for _ in task_outputs]),
        StaticTaskRouting(nt, [nn.Linear(128, s) for s in task_outputs]),
    )


def prior_sharing(task_outputs, num_modules):
    nt = len(task_outputs)
    return nn.Sequential(
        IgnoreTaskRouting(conv_layer(1, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),
        StaticTaskRouting(nt, [conv_layer(32, 32) for _ in task_outputs]),

        IgnoreTaskRouting(Flatten()),
        StaticTaskRouting(nt, [dense_layer(288, 128) for _ in task_outputs]),
        StaticTaskRouting(nt, [nn.Linear(128, s) for s in task_outputs]),
    )


def full_sharing(task_outputs, num_modules):
    nt = len(task_outputs)
    return nn.Sequential(
        IgnoreTaskRouting(conv_layer(1, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),

        IgnoreTaskRouting(Flatten()),
        IgnoreTaskRouting(dense_layer(288, 128)),
        StaticTaskRouting(nt, [nn.Linear(128, s) for s in task_outputs]),
    )


def conv_layer(ch_in, ch_out):
    return nn.Sequential(nn.Conv2d(ch_in, ch_out, 3, bias=False, padding=1),
                         nn.BatchNorm2d(ch_out, momentum=0.1),
                         nn.ReLU(),
                         nn.MaxPool2d(2))


def dense_layer(ch_in, ch_out):
    return nn.Sequential(nn.Linear(ch_in, ch_out),
                         nn.ReLU())
