from collections import defaultdict

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from task_routing_model import learned_task_routing_apply, get_routing_probs, task_routing_apply
from multitask_dataset import heterogeneous_dict_collate


class Trainer:
    def __init__(self, model, mtl_train, mtl_test, loss_fn, metric_fn, args):
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.args = args

        self.trainloader = iter(DataLoader(mtl_train, batch_size=args.batch_size, num_workers=0))
        self.testloader = DataLoader(mtl_test, batch_size=args.batch_size, collate_fn=heterogeneous_dict_collate, num_workers=0)
        self.opt = Adam(model.parameters(), lr=args.sgd_lr)

        self.logdata = defaultdict(list)
        self.logdata['args'] = args

    def _log(self, key, value):
        self.logdata[key].append(value)

    def save_logfile(self, file):
        np.savez(file, **self.logdata)

    def _batch_transform_hook(self, batch):
        return taskbatch_to_device(batch, self.args.device)

    def _pass_through_model(self, batch, backward):
        losses = []
        for task, (x, y) in batch.items():
            bs = x.size(0)
            with torch.set_grad_enabled(backward):
                out, task = self.model((x, task))
                loss = self.loss_fn(out, y, reduction='none').view(bs, -1).mean(1)
            if backward:
                loss.mean().backward()
            losses.append(loss.detach())
        return torch.cat(losses).mean()

    def _nes_update_step(self):
        self.model.eval()
        learned_task_routing_apply(self.model, lambda m: m.clear_routing_history())

        batch = next(self.trainloader)
        batch = self._batch_transform_hook(batch)
        samples_loss = []
        for j in range(self.args.nes_samples):
            learned_task_routing_apply(self.model, lambda m: m.sample_routing(record=True))
            loss = self._pass_through_model(batch, backward=False)
            samples_loss.append(loss)
        samples_loss = torch.stack(samples_loss)

        # descending_ranks is 0 for the highest loss
        descending_ranks = samples_loss.argsort(descending=True).argsort()
        utilities = descending_ranks.to(torch.float32) / (descending_ranks.size(0) - 1) * 2 - 1
        learned_task_routing_apply(self.model, lambda m: m.update_routing_probs(utilities, self.args.nes_lr, p_min=0.001))

        return samples_loss

    def _sgd_update_step(self):
        self.opt.zero_grad()
        self.model.train()
        learned_task_routing_apply(self.model, lambda m: m.clear_routing_history())

        batch = next(self.trainloader)
        batch = self._batch_transform_hook(batch)
        samples_loss = []
        for j in range(self.args.sgd_samples):
            learned_task_routing_apply(self.model, lambda m: m.sample_routing())
            loss = self._pass_through_model(batch, backward=True)
            samples_loss.append(loss)
        task_routing_apply(self.model, lambda m: m.normalize_weight_gradients(self.args.num_tasks, self.args.sgd_samples))
        self.opt.step()

        samples_loss = torch.stack(samples_loss)
        return samples_loss

    def evaluate(self, iteration):
        self.model.eval()
        learned_task_routing_apply(self.model, lambda m: m.maximum_likelihood_routing())

        batches_loss, batches_metric = defaultdict(list), defaultdict(list)
        for batch in self.testloader:
            batch = self._batch_transform_hook(batch)
            for task, (x, y) in batch.items():
                with torch.no_grad():
                    bs = x.size(0)
                    out, task = self.model((x, task))
                    loss = self.loss_fn(out, y, reduction='none').view(bs, -1).mean(1)
                    metric = self.metric_fn(out, y, reduction='none').view(bs, -1).mean(1)
                    batches_loss[task].append(loss)
                    batches_metric[task].append(metric)

        # Task-average loss and metric
        loss = torch.cat(sum(batches_loss.values(), [])).mean().item()
        metric = torch.cat(sum(batches_metric.values(), [])).mean().item()

        # Individual loss and metric
        tasks = list(range(len(batches_loss)))
        loss_per_task = [torch.cat(batches_loss[t]).mean().item() for t in tasks]
        metric_per_task = [torch.cat(batches_metric[t]).mean().item() for t in tasks]

        # Entropy of routing distributions
        ent = model_entropy(self.model)
        probslist = [p.cpu().tolist() for p in get_routing_probs(self.model)]

        self._log('test/iteration', iteration)
        self._log('test/loss', loss)
        self._log('test/metric', metric)
        self._log('test/loss_per_task', loss_per_task)
        self._log('test/metric_per_task', metric_per_task)
        self._log('test/entropy', ent)
        self._log('test/routing_probs', probslist)
        return {'loss': loss, 'metric': metric, 'entropy': ent}

    def trainstep(self, iteration):
        if self.args.nes_samples is not None:
            self._nes_update_step()

        samples_loss = self._sgd_update_step()
        self._log('train/iteration', iteration)
        self._log('train/loss', samples_loss.mean().item())


def taskbatch_to_device(batch, device):
    # Batches are dictionaries of the form {0: [x, y], 7: [x, y]}
    return {k: (x.to(device), y.to(device)) for k, (x, y) in batch.items()}


def entropy(probs):
    """Entropy of PMFs in the last dimension of the given array"""
    return -(probs * probs.log()).sum(-1)


def model_entropy(model):
    probs = get_routing_probs(model)
    ent = [entropy(p) for p in probs]
    if ent:
        return torch.cat(ent).mean().item()
    else:
        return float('nan')


def reset_parameters(mod):
    if hasattr(mod, 'reset_parameters'):
        mod.reset_parameters()
