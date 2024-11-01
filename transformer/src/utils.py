from functools import partial
import os
from statistics import mean
import torch.distributed as dist
from collections import deque
from loguru import logger
from decent_dp.ddp import OPTIM_FN_TYPE, LR_SCHEDULER_FN_TYPE
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from typing import TYPE_CHECKING, List, Tuple
if TYPE_CHECKING:
    from .conf import Config


class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{avg:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.window_size = window_size
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append((value, n))
        self.count += n
        self.total += value * n

    @property
    def avg(self) -> float:
        count = 0.
        total = 0.
        for i in reversed(range(len(self.deque))):
            if count + self.deque[i][1] >= self.window_size:
                total += self.deque[i][0] * (self.window_size - count)
                count = self.window_size
                break
            count += self.deque[i][1]
            total += self.deque[i][0] * self.deque[i][1]
        if count == 0:
            return 0
        return total / count

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(avg=self.avg, global_avg=self.global_avg)


def initialize_dist() -> None:
    if ("WORLD_SIZE" in os.environ) and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        if dist.get_rank() == 0:
            logger.info(f"Initialized the process group with {dist.get_world_size()} processes.")
        dist.barrier()


def get_optim(cfg: "Config", model: Module) -> Optimizer:
    match cfg.train.optim.name.lower():
        case "adam":
            return torch.optim.Adam(model.parameters(),
                                    lr=cfg.train.optim.lr,
                                    betas=cfg.train.optim.betas,
                                    eps=cfg.train.optim.eps)
        case _:
            raise ValueError(f"Unsupported optimizer: {cfg.train.optim.name}")


def get_optim_fn(cfg: "Config") -> OPTIM_FN_TYPE:
    match cfg.train.optim.name.lower():
        case "adam":
            def fn(params: List[Tuple[Tensor, str]],
                   lr: float,
                   betas: Tuple[float, float],
                   eps: float) -> Optimizer:
                return torch.optim.Adam([x for x, _ in params], lr=lr, betas=betas, eps=eps)
            return partial(fn,
                           lr=cfg.train.optim.lr,
                           betas=cfg.train.optim.betas,
                           eps=cfg.train.optim.eps)
        case "accum-adam":
            from decent_dp.optim import AccumAdam
            def fn(params: List[Tuple[Tensor, str]],
                   lr: float,
                   betas: Tuple[float, float],
                   eps: float,
                   accum_iter: int) -> Optimizer:
                return AccumAdam([x for x, _ in params], lr=lr, betas=betas, eps=eps, accum_iter=accum_iter)
            return partial(fn,
                           lr=cfg.train.optim.lr,
                           betas=cfg.train.optim.betas,
                           eps=cfg.train.optim.eps,
                           accum_iter=cfg.train.optim.accum_iter)
        case _:
            raise ValueError(f"Unsupported optimizer: {cfg.train.optim.name}")


def get_lr_scheduler(cfg: "Config", optimizer: Optimizer) -> LRScheduler:
    match cfg.train.lr_scheduler.type.lower():
        case "inverse_sqrt":
            def lr_lambda(step: int) -> float:
                if step < cfg.train.lr_scheduler.warmup_steps:
                    return (1 - cfg.train.lr_scheduler.warmup_decay) * (step + 1) / cfg.train.lr_scheduler.warmup_steps \
                        + cfg.train.lr_scheduler.warmup_decay
                else:
                    return (cfg.train.lr_scheduler.warmup_steps ** 0.5) * ((step + 1) ** (-0.5))
            return LambdaLR(optimizer, lr_lambda=lr_lambda)
        case _:
            raise ValueError(f"Unsupported lr_scheduler: {cfg.train.lr_scheduler.type}")


def get_lr_scheduler_fn(cfg: "Config") -> LR_SCHEDULER_FN_TYPE:
    match cfg.train.lr_scheduler.type.lower():
        case "inverse_sqrt":
            def fn(optimizer: Optimizer,
                   warmup_steps: int,
                   warmup_decay: float) -> LRScheduler:
                def lr_lambda(step: int) -> float:
                    if step < warmup_steps:
                        return (1 - warmup_decay) * (step + 1) / warmup_steps + warmup_decay
                    else:
                        return (warmup_steps ** 0.5) * ((step + 1) ** (-0.5))
                return LambdaLR(optimizer, lr_lambda=lr_lambda)
            return partial(fn,
                           warmup_steps=cfg.train.lr_scheduler.warmup_steps,
                           warmup_decay=cfg.train.lr_scheduler.warmup_decay)
        case _:
            raise ValueError(f"Unsupported lr_scheduler: {cfg.train.lr_scheduler.type}")


def gather_statistics(train_loss: float,
                      train_perplexity: float,
                      val_loss: float,
                      val_perplexity: float) -> List[float]:
    log = [train_loss, train_perplexity, val_loss, val_perplexity]
    if dist.is_available() and dist.is_initialized():
        object_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(object_list, log)
        for i in range(len(log)):
            log[i] = mean([x[i] for x in object_list])
    return log


@torch.no_grad()
def batch_beam_search(model: torch.nn.Module,
                      src: torch.Tensor,
                      token_sos_id: int,
                      token_eos_id: int,
                      token_pad_id: int,
                      beam_size: int=4,
                      len_penalty: float=0.6,
                      tolerance: int=50) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Batch beam search algorithm for sequence generation. Note that the efficiency of this implementation is not guaranteed.
    
    Args:
        model (torch.nn.Module): The model to generate sequence.
        src (torch.Tensor): The source sequence.
        token_sos_id (int): The id of start-of-sequence token.
        token_eos_id (int): The id of end-of-sequence token.
        token_pad_id (int): The id of padding token.
        beam_size (int): The size of beam.
        len_penalty (float): The length penalty factor.
        tolerance (int): The maximum length of generated sequence.
    '''

    batch_size = src.size(0)

    src_lens = torch.sum(src != token_pad_id, dim=-1, keepdim=True)
    src_lens = torch.tile(src_lens, (1, beam_size)).view(batch_size * beam_size, 1) # (batch_size * beam_size, 1)

    memory, src_padding_mask = model.get_memory(src.unsqueeze(1).repeat(1, beam_size, 1).view(batch_size * beam_size, -1)) # (batch_size * beam_size, src_seq_len, d_model)

    output = {
        "tgt": torch.full((batch_size * beam_size, 1), token_sos_id, dtype=torch.int64, device=src.device),
        "logprob": torch.zeros((batch_size * beam_size, 1), dtype=torch.float32, device=src.device),
        "is_ended": torch.zeros((batch_size * beam_size, 1), dtype=torch.bool, device=src.device),
        "seq_lens": torch.ones((batch_size * beam_size, 1), dtype=torch.float32, device=src.device)
    }

    for i in range(tolerance + src.size(1) + 1):
        # extract logprob of the lastest token
        logit = model.get_tgt_from_memory(src_padding_mask, memory, output["tgt"]) # (batch_size * beam_size, tgt_seq_len, vocab_size)
        logprob = nn.functional.log_softmax(logit[:, -1, :], -1) # (batch_size * beam_size, vocab_size)

        # pick topk tokens at current step
        topk = torch.topk(logprob, beam_size, dim=-1) # (batch_size * beam_size, beam_size)

        # update logprob for candidate sequences
        topk_logprob = torch.where(output["is_ended"].repeat(1, beam_size), torch.zeros_like(topk.values), topk.values) + output["logprob"] # (batch_size * beam_size, beam_size)

        # update sequence length
        topk_token = topk.indices # (batch_size * beam_size, beam_size)
        topk_is_ended = output["is_ended"].repeat(1, beam_size) # (batch_size * beam_size, beam_size)
        topk_seq_len = output["seq_lens"] + (~topk_is_ended).float() # (batch_size * beam_size, beam_size)
        topk_is_ended = topk_is_ended | (topk_token == token_eos_id) | (output["seq_lens"] >= (src_lens + tolerance)) # (batch_size * beam_size, beam_size)

        # compute corresponding length penalty
        topk_penalty = ((5 + topk_seq_len) / 6) ** len_penalty # (batch_size * beam_size, beam_size)

        # compute score
        topk_score = topk_logprob / topk_penalty # (batch_size * beam_size, beam_size)

        # append new token to candidate sequences
        topk_seq = torch.concat([output["tgt"].unsqueeze(-1).repeat(1, 1, beam_size), topk_token.unsqueeze(1)], dim=1) # (batch_size * beam_size, tgt_seq_len + 1, beam_size)
        topk_seq = topk_seq.transpose(1, 2) # (batch_size * beam_size, beam_size, tgt_seq_len + 1)
        topk_seq = topk_seq.reshape(batch_size, beam_size * beam_size, i + 2) # (batch_size, beam_size * beam_size, tgt_seq_len + 1)

        # select topk for each sample
        topk_score = topk_score.view(batch_size, beam_size * beam_size)
        new_topk = torch.topk(topk_score, beam_size, dim=-1)

        # update output variables
        output["logprob"] = topk_logprob.view(batch_size, beam_size * beam_size).gather(1, new_topk.indices).reshape(batch_size * beam_size, 1)
        output["tgt"] = topk_seq.gather(1, new_topk.indices.unsqueeze(2).repeat(1, 1, i + 2)).reshape(batch_size * beam_size, i + 2)
        output["is_ended"] = topk_is_ended.view(batch_size, beam_size * beam_size).gather(1, new_topk.indices).reshape(batch_size * beam_size, 1)
        output["seq_lens"] = topk_seq_len.view(batch_size, beam_size * beam_size).gather(1, new_topk.indices).reshape(batch_size * beam_size, 1)

        if output["is_ended"].all():
            break
    
    return output["seq_lens"], output["tgt"]


def get_run_name(cfg: "Config"):
    world_size = cfg.train.network.world_size
    local_world_size = cfg.train.network.local_world_size
    gpu = cfg.train.network.gpu_model.split(' ')[1]

    if cfg.train.backend.lower() == 'pytorchddp':
        return f'{cfg.train.backend} - {cfg.train.optim.name} - ' + \
            f'{world_size // local_world_size}x{local_world_size}x{gpu} - ' + \
            f'{cfg.train.reproduce.seed}'
    elif cfg.train.backend.lower() == 'decentdp':
        return f'{cfg.train.backend} - {cfg.train.optim.name} - ' + \
            f'{world_size // local_world_size}x{local_world_size}x{gpu} - ' + \
            f'{cfg.train.decent.topology} - {cfg.train.reproduce.seed}'
    else:
        raise ValueError(f'Unsupported backend: {cfg.train.backend}')
        
