import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer, Adam, SGD, AdamW
from torch.optim.lr_scheduler import LRScheduler
from typing import TYPE_CHECKING, Callable, List, Tuple
from functools import partial
from decent_dp.optim import AccumAdam
if TYPE_CHECKING:
    from conf import Config

def get_params(model: nn.Module, weight_decay: float) -> list:
    bn_params = [v for n, v in model.named_parameters() if ('bn' in n) or ('bias' in n)]
    rest_params = [v for n, v in model.named_parameters() if not (('bn' in n) or ('bias' in n))]
    return [
        {"params": bn_params, "weight_decay": 0},
        {"params": rest_params, "weight_decay": weight_decay}
    ]


def get_params_from_list(params: List[Tuple[Tensor, str]], weight_decay: float) -> list:
    bn_params = [p for p, n in params if ('bn' in n) or ('bias' in n)]
    rest_params = [p for p, n in params if not (('bn' in n) or ('bias' in n))]
    return [
        {"params": bn_params, "weight_decay": 0},
        {"params": rest_params, "weight_decay": weight_decay}
    ]


def get_optim(cfg: "Config", model: nn.Module) -> Optimizer:
    match cfg.train.optim.name.lower():
        case "adam":
            return Adam(get_params(model, cfg.train.optim.weight_decay),
                        lr=cfg.train.lr,
                        betas=(cfg.train.optim.adam.beta1, cfg.train.optim.adam.beta2),
                        eps=cfg.train.optim.adam.epsilon)
        case "sgd":
            return SGD(get_params(model, cfg.train.optim.weight_decay),
                       lr=cfg.train.lr,
                       momentum=cfg.train.optim.sgd.momentum)
        case "adamw":
            return AdamW(get_params(model, cfg.train.optim.weight_decay),
                         lr=cfg.train.lr,
                         betas=(cfg.train.optim.adam.beta1, cfg.train.optim.adam.beta2),
                         eps=cfg.train.optim.adam.epsilon)
        case _:
            raise ValueError(f"Unknown optimizer: {cfg.train.optim.name}")


def get_optim_fn(cfg: "Config") -> Callable[[List[Tuple[Tensor, str]]], Optimizer]:
    match cfg.train.optim.name.lower():
        case "adam":
            def adam_optim_fn(params: List[Tuple[Tensor, str]],
                              lr: float,
                              betas: Tuple[float, float],
                              eps: float,
                              weight_decay: float) -> Optimizer:
                return Adam(get_params_from_list(params, weight_decay), lr=lr, betas=betas, eps=eps)
            return partial(adam_optim_fn,
                           lr=cfg.train.lr,
                           betas=(cfg.train.optim.adam.beta1, cfg.train.optim.adam.beta2),
                           eps=cfg.train.optim.adam.epsilon,
                           weight_decay=cfg.train.optim.weight_decay)
        case "accumadam":
            def accumadam_optim_fn(params: List[Tuple[Tensor, str]],
                                   lr: float,
                                   betas: Tuple[float, float],
                                   eps: float,
                                   weight_decay: float,
                                   accum_iter: int) -> Optimizer:
                return AccumAdam(get_params_from_list(params, weight_decay),
                                 lr=lr,
                                 betas=betas,
                                 eps=eps,
                                 accum_iter=accum_iter)
            return partial(accumadam_optim_fn,
                           lr=cfg.train.lr,
                           betas=(cfg.train.optim.adam.beta1, cfg.train.optim.adam.beta2),
                           eps=cfg.train.optim.adam.epsilon,
                           weight_decay=cfg.train.optim.weight_decay,
                           accum_iter=cfg.train.optim.accum_iter)
        case "sgd":
            def sgd_optim_fn(params: List[Tuple[Tensor, str]],
                             lr: float,
                             momentum: float,
                             weight_decay: float) -> Optimizer:
                return SGD(get_params_from_list(params, weight_decay), lr=lr, momentum=momentum)
            return partial(sgd_optim_fn,
                           lr=cfg.train.lr,
                           momentum=cfg.train.optim.sgd.momentum,
                           weight_decay=cfg.train.optim.weight_decay)
        case _:
            raise ValueError(f"Unknown optimizer: {cfg.train.optim.name}")



def get_lr_scheduler(cfg: "Config", optim: Optimizer, num_steps_per_epoch: int) -> LRScheduler:
    match cfg.train.lr_scheduler.name.lower():
        case "cosine":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=cfg.train.lr_scheduler.warmup_decay,
                total_iters=num_steps_per_epoch * cfg.train.lr_scheduler.warmup_epochs
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                T_max=num_steps_per_epoch * (cfg.train.max_epochs - cfg.train.lr_scheduler.warmup_epochs),
                eta_min=1e-5
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optim,
                schedulers=[warmup_lr_scheduler, main_scheduler],
                milestones=[num_steps_per_epoch * cfg.train.lr_scheduler.warmup_epochs]
            )
        case _:
            raise ValueError(f"Unknown LR scheduler: {cfg.train.lr_scheduler.name}")


def get_lr_scheduler_fn(cfg: "Config", num_steps_per_epoch: int) -> Callable[[Optimizer], LRScheduler]:
    match cfg.train.lr_scheduler.name.lower():
        case "cosine":
            def scheduler_fn(optim: Optimizer,
                             warmup_decay: float,
                             warmup_epochs: int,
                             max_epochs: int,
                             num_steps_per_epoch: int) -> LRScheduler:
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optim,
                    start_factor=warmup_decay,
                    total_iters=num_steps_per_epoch * warmup_epochs
                )
                main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim,
                    T_max=num_steps_per_epoch * (max_epochs - warmup_epochs),
                    eta_min=1e-5
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    optim,
                    schedulers=[warmup_lr_scheduler, main_scheduler],
                    milestones=[num_steps_per_epoch * warmup_epochs]
                )
            return partial(scheduler_fn,
                           warmup_decay=cfg.train.lr_scheduler.warmup_decay,
                           warmup_epochs=cfg.train.lr_scheduler.warmup_epochs,
                           max_epochs=cfg.train.max_epochs,
                           num_steps_per_epoch=num_steps_per_epoch)
        case _:
            raise ValueError(f"Unknown LR scheduler: {cfg.train.lr_scheduler.name}")


