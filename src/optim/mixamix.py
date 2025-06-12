"""
Generalization of AdEMAMix.
"""

import math
from typing import Callable, Optional

import torch
from torch.optim.optimizer import ParamsT

from optim.ademamix import linear_hl_warmup_scheduler, linear_warmup_scheduler

class MixAMix(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, ...] = (0.9, 0.9999),
        alphas: tuple[float, ...] = (1.0, 2.0),
        delta: float = 0.999,  # Beta2 in adamw and ademamix.
        beta_warmups: tuple[int, ...] = (1, 1),
        alpha_warmups: tuple[int, ...] = (1, 1),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
    ):
        assert len(betas) == len(alphas) == len(beta_warmups) == len(alpha_warmups)
        defaults = dict(
            lr=lr,
            alphas=alphas,
            betas=betas,
            delta=delta,
            eps=eps,
            beta_warmups=beta_warmups,
            alpha_warmups=alpha_warmups,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in filter(lambda param: param.grad is not None, group["params"]):
                # Initialize state.
                if len(self.state[param]) == 0:
                    self.state[param]["step"] = 1
                    self.state[param]["ema_sq"] = torch.zeros_like(param)
                    self.state[param]["emas"] = [torch.zeros_like(param)
                                                 for _ in range(len(group["betas"]))]

                # Main step.
                mixamix_step(param, param.grad, self.state[param]["step"], self.state[param]["emas"],
                             self.state[param]["ema_sq"], group["lr"], group["weight_decay"],
                             group["eps"], group["alphas"], group["betas"], group["delta"],
                             group["bias_correction"], group["alpha_warmups"], group["beta_warmups"])
        return loss


@torch.no_grad()
def mixamix_step(param: torch.Tensor, grad: torch.Tensor, step: int,
                 emas: list[torch.Tensor], ema_sq: torch.Tensor,
                 lr: float, decay: float, eps: float,
                 alphas: tuple[float, ...], betas: tuple[float, ...],
                 delta: float, bias_correction: bool,
                 alpha_warmups: tuple[int, ...], beta_warmups: tuple[int, ...]):

    # Determine true alpha and deltas.
    true_alphas = []
    true_betas = []
    for i, (alpha, beta, alpha_warmup, beta_warmup) in enumerate(zip(alphas, betas, alpha_warmups, beta_warmups)):
        true_alphas.append(linear_warmup_scheduler(step, alpha, 0.0, alpha_warmup))
        beta_start = 0.0 if i == 0 else betas[i - 1]
        true_betas.append(linear_hl_warmup_scheduler(step, beta, beta_start, beta_warmup))

    # Apply first momentums.
    update = torch.zeros_like(param)
    for alpha, beta, ema in zip(true_alphas, true_betas, emas):
        ema.mul_(beta).add_(grad, alpha=1 - beta)
        update.add_(ema, alpha=alpha/(1 - beta**step if bias_correction else 1))

    # Apply second momentum, always bias-corrected.
    ema_sq.mul_(delta).addcmul_(grad, grad, value=1 - delta)
    update.div_(ema_sq.sqrt().div_(math.sqrt(1 - delta**step)).add_(eps))

    # Apply weight decay and update param.
    update.add_(param, alpha=decay)
    param.add_(update, alpha=-lr)
