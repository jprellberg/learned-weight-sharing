import torch
import torch.nn as nn
from torch.distributions import Categorical


class TaskRouting(nn.Module):
    def normalize_weight_gradients(self, num_tasks, num_samples):
        raise NotImplementedError


class StaticTaskRouting(TaskRouting):
    def __init__(self, num_tasks, module_list):
        super().__init__()
        assert num_tasks == len(module_list)
        self.num_tasks = num_tasks
        self.module_list = nn.ModuleList(module_list)

    def forward(self, input):
        x, task = input
        module = self.module_list[task]
        return module(x), task

    def normalize_weight_gradients(self, num_tasks, num_samples):
        for p in self.parameters():
            if p.grad is not None:
                p.grad /= num_samples


class LearnedTaskRouting(TaskRouting):
    def __init__(self, num_tasks, module_list):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_modules = len(module_list)
        self.module_list = nn.ModuleList(module_list)
        routing_probs = torch.ones(num_tasks, self.num_modules) / self.num_modules
        routing_sample = Categorical(probs=routing_probs).sample()
        # routing_probs shape is [num_tasks, num_modules]
        # routing_sample shape is [num_tasks]
        self.register_buffer('routing_probs', routing_probs)
        self.register_buffer('routing_sample', routing_sample)
        self.routing_history = []

    def sample_routing(self, record=False):
        self.routing_sample = Categorical(probs=self.routing_probs).sample()
        if record:
            self.routing_history.append(self.routing_sample.clone())

    def set_routing_from_history(self, index):
        self.routing_sample = self.routing_history[index]

    def maximum_likelihood_routing(self):
        self.routing_sample = self.routing_probs.argmax(-1)

    def clear_routing_history(self):
        self.routing_history.clear()

    def update_routing_probs(self, utilities, stepsize, p_min):
        # Utility-weighted mean of gradients
        stacked_grads = torch.stack([natural_gradient(self.routing_probs, sample) for sample in self.routing_history])
        grad = (stacked_grads * utilities.view(-1, 1, 1)).mean(0)

        # Update step on probabilities
        self.routing_probs = self.routing_probs + stepsize * grad
        assert self.routing_probs.sum(-1).allclose(torch.tensor(1.0, device=self.routing_probs.device))

        # Clamp probabilities to prevent degenerate distribution
        self.routing_probs.clamp_(min=p_min)
        self.routing_probs /= self.routing_probs.sum(-1, keepdim=True)

    # def normalize_weight_gradients(self, num_tasks, num_samples):
    #     # Different modules will have accumulated different numbers of gradients when backward has been
    #     # called depending on the routing_sample that was set at that time
    #     usage_counts = torch.cat(self.routing_history).bincount(minlength=self.num_modules)
    #     for mod, count in zip(self.module_list, usage_counts):
    #         for p in mod.parameters():
    #             if p.grad is not None:
    #                 p.grad /= count * num_samples

    def normalize_weight_gradients(self, num_tasks, num_samples):
        module_factors = self.routing_probs.sum(0)
        for mod, factor in zip(self.module_list, module_factors):
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad /= factor * num_samples

    def forward(self, input):
        x, task = input
        route = self.routing_sample[task]
        module = self.module_list[route]
        return module(x), task


class IgnoreTaskRouting(TaskRouting):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, input):
        x, task = input
        return self.wrapped(x), task

    def normalize_weight_gradients(self, num_tasks, num_samples):
        for p in self.parameters():
            if p.grad is not None:
                p.grad /= num_tasks * num_samples


def natural_gradient(probs, sample):
    case_eq = torch.zeros_like(probs)
    case_eq[range(case_eq.size(0)), sample] = 1
    grad = case_eq - probs
    return grad


def learned_task_routing_apply(model, fn):
    """Applies a function to every LearnedTaskRouting module in the given model and returns the results"""
    results = []
    for m in model.modules():
        if isinstance(m, LearnedTaskRouting):
            out = fn(m)
            results.append(out)
    return results


def task_routing_apply(model, fn):
    """Applies a function to every TaskRouting module in the given model and returns the results"""
    results = []
    for m in model.modules():
        if isinstance(m, TaskRouting):
            out = fn(m)
            results.append(out)
    return results


def get_routing_probs(model):
    return learned_task_routing_apply(model, lambda m: m.routing_probs)
