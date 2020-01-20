import torch.nn as nn
import torch.nn.functional as F

from task_routing_model import StaticTaskRouting, LearnedTaskRouting, IgnoreTaskRouting
from utils import Flatten


######################## MODEL FROM RoutingNetworks PAPER ########################


def learned_sharing(task_outputs, num_modules):
    nt = len(task_outputs)
    return nn.Sequential(
        IgnoreTaskRouting(conv_layer(3, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),

        IgnoreTaskRouting(Flatten()),
        LearnedTaskRouting(nt, [dense_layer(128, 128) for _ in range(num_modules)]),
        LearnedTaskRouting(nt, [dense_layer(128, 128) for _ in range(num_modules)]),
        LearnedTaskRouting(nt, [dense_layer(128, 128) for _ in range(num_modules)]),

        StaticTaskRouting(nt, [nn.Linear(128, s) for s in task_outputs]),
    )


def no_sharing(task_outputs, num_modules):
    nt = len(task_outputs)
    return nn.Sequential(
        IgnoreTaskRouting(conv_layer(3, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),

        IgnoreTaskRouting(Flatten()),
        StaticTaskRouting(nt, [dense_layer(128, 128) for _ in task_outputs]),
        StaticTaskRouting(nt, [dense_layer(128, 128) for _ in task_outputs]),
        StaticTaskRouting(nt, [dense_layer(128, 128) for _ in task_outputs]),

        StaticTaskRouting(nt, [nn.Linear(128, s) for s in task_outputs]),
    )


def prior_sharing(task_outputs, num_modules):
    nt = len(task_outputs)
    return nn.Sequential(
        IgnoreTaskRouting(conv_layer(3, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),

        IgnoreTaskRouting(Flatten()),
        IgnoreTaskRouting(dense_layer(128, 128)),
        StaticTaskRouting(nt, [dense_layer(128, 128) for _ in task_outputs]),
        StaticTaskRouting(nt, [dense_layer(128, 128) for _ in task_outputs]),

        StaticTaskRouting(nt, [nn.Linear(128, s) for s in task_outputs]),
    )


def full_sharing(task_outputs, num_modules):
    nt = len(task_outputs)
    return nn.Sequential(
        IgnoreTaskRouting(conv_layer(3, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),
        IgnoreTaskRouting(conv_layer(32, 32)),

        IgnoreTaskRouting(Flatten()),
        IgnoreTaskRouting(dense_layer(128, 128)),
        IgnoreTaskRouting(dense_layer(128, 128)),
        IgnoreTaskRouting(dense_layer(128, 128)),

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


######################## RESNET18 MODELS ########################


class ResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or ch_in != ch_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = self.main(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, task_outputs, num_modules):
        super().__init__()
        self.num_tasks = len(task_outputs)
        self.num_modules = num_modules
        self.ch_in = 64
        self.conv1 = IgnoreTaskRouting(nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ))
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)
        self.pool_flatten = IgnoreTaskRouting(nn.Sequential(
            nn.AvgPool2d(4),
            Flatten()
        ))
        self.fc = StaticTaskRouting(self.num_tasks, [nn.Linear(512, s) for s in task_outputs])

    def make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block = self.make_wrapped_block(channels, stride)
            layers.append(block)
            self.ch_in = channels
        return nn.Sequential(*layers)

    def make_wrapped_block(self, channels, stride):
        raise NotImplementedError

    def make_block(self, channels, stride):
        return ResidualBlock(self.ch_in, channels, stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool_flatten(out)
        out = self.fc(out)
        return out


class LearnedSharingResNet18(ResNet18):
    def make_wrapped_block(self, channels, stride):
        return LearnedTaskRouting(self.num_tasks, [
            self.make_block(channels, stride) for _ in range(self.num_modules)
        ])


class NoSharingResNet18(ResNet18):
    def make_wrapped_block(self, channels, stride):
        return StaticTaskRouting(self.num_tasks, [
            self.make_block(channels, stride) for _ in range(self.num_tasks)
        ])


class PriorSharingResNet18(ResNet18):
    def make_wrapped_block(self, channels, stride):
        if channels == 64 or channels == 128:
            return IgnoreTaskRouting(self.make_block(channels, stride))
        else:
            return StaticTaskRouting(self.num_tasks, [
                self.make_block(channels, stride) for _ in range(self.num_tasks)
            ])


class FullSharingResNet18(ResNet18):
    def make_wrapped_block(self, channels, stride):
        return IgnoreTaskRouting(self.make_block(channels, stride))
