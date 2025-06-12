"""
Muon optimizer with ademamix-like second EMA.
"""

import os
import math
from typing import Any, Optional

import torch
from torch.optim.optimizer import ParamsT
from torch import distributed as dist

from optim.ademamix import linear_hl_warmup_scheduler, linear_warmup_scheduler
from optim.muon import zeropower_via_newtonschulz5


class MuonMix(torch.optim.Optimizer):
    def __init__(
        self,
        muon_params: ParamsT,
        lr: float = 0.02,
        momentum1: float = 0.95,
        momentum2: float = 0.9999,
        nesterov: bool = True,
        ns_steps: int = 6,
        adema_params: Optional[ParamsT] = None,
        adema_lr: float = 3e-4,
        adema_betas: tuple[float, float, float] = (0.9, 0.95, 0.9999),
        adema_eps: float = 1e-8,
        adema_wd: float = 0.0,
        adema_alpha: float = 2.0,
        beta3_warmup: Optional[int] = None,
        alpha_warmup: Optional[int] = None,
    ):

        defaults = {
            "lr": lr, "momentum1": momentum1, "momentum2": momentum2, "nesterov": nesterov,
            "ns_steps": ns_steps, "adema_lr": adema_lr, "adema_lr_ratio": adema_lr/lr,
            "adema_betas": adema_betas, "adema_eps": adema_eps, "adema_wd": adema_wd,
            "adema_alpha": adema_alpha, "beta3_warmup": beta3_warmup, "alpha_warmup": alpha_warmup
        }
        params = list(muon_params) + (list(adema_params) if adema_params is not None else [])
        super().__init__(params, defaults)

        for p in muon_params:
            self.state[p]["use_muon"] = p.ndim >= 2 and p.size(0) < 10_000
        for p in adema_params if adema_params is not None else []:
            self.state[p]["use_muon"] = False

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))


    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            self._muon_step(group)
            self._ademamix_step(group)


    def _muon_step(self, group: dict[str, Any]):
        # Generate weight updates in distributed fashion.
        params = [p for p in group["params"] if self.state[p]["use_muon"]]
        total_params = sum(p.numel() for p in params)
        updates_flat = torch.zeros(
            total_params, device=params[0].device, dtype=torch.bfloat16
        )
        curr_idx = 0
        for i, p in enumerate(params):
            # Initialize "step" if not found.
            if "step" not in self.state[p]:
                self.state[p]["step"] = 1

            # Get alpha and momentum2.
            if group["alpha_warmup"] is None:
                alpha = group["adema_alpha"]
            else:
                alpha = linear_warmup_scheduler(
                    self.state[p]["step"],
                    alpha_end=group["adema_alpha"],
                    alpha_start=0,
                    warmup=group["alpha_warmup"],
                )
            if group["beta3_warmup"] is None:
                momentum2 = group["momentum2"]
            else:
                momentum2 = linear_hl_warmup_scheduler(
                    self.state[p]["step"],
                    beta_end=group["momentum2"],
                    beta_start=group["momentum1"],
                    warmup=group["beta3_warmup"],
                )

            # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
            if i % self.world_size == self.rank:
                g = p.grad
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None
                state = self.state[p]
                if "momentum_buffer1" not in state:
                    state["momentum_buffer1"] = torch.zeros_like(g)
                    state["momentum_buffer2"] = torch.zeros_like(g)
                buf1 = state["momentum_buffer1"]
                buf2 = state["momentum_buffer2"]

                buf1.mul_(group["momentum1"]).add_(g)
                buf2.mul_(momentum2).add_(g)
                if group["nesterov"]:
                    g1 = g.add(buf1, alpha=group["momentum1"])
                    g2 = g.add(buf2, alpha=momentum2)
                else:
                    g1 = buf1
                    g2 = buf2

                g1 = zeropower_via_newtonschulz5(g1, steps=group["ns_steps"])
                g2 = zeropower_via_newtonschulz5(g2, steps=group["ns_steps"])

                g1 *= max(1, g1.size(0) / g1.size(1)) ** 0.5
                g2 *= max(1, g2.size(0) / g2.size(1)) ** 0.5
                update = g1 + alpha*g2

                updates_flat[curr_idx : curr_idx + p.numel()] = update.flatten()
            curr_idx += p.numel()
            self.state[p]["step"] += 1

        # sync updates across devices. we are not memory-constrained so can do this simple deserialization
        if self.world_size > 1:
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

        # deserialize and apply updates
        curr_idx = 0
        for p in params:
            g = (
                updates_flat[curr_idx : curr_idx + p.numel()]
                .view_as(p.data)
                .type_as(p.data)
            )
            p.data.add_(g, alpha=-group["lr"])
            curr_idx += p.numel()

    def _ademamix_step(self, group):
        params = [p for p in group["params"] if not self.state[p]["use_muon"]]
        lr = group["adema_lr"]
        lmbda = group["adema_wd"]
        eps = group["adema_eps"]
        beta1, beta2, beta3_final = group["adema_betas"]
        beta3_warmup = group["beta3_warmup"]
        alpha_final = group["adema_alpha"]
        alpha_warmup = group["alpha_warmup"]

        for p in params:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("AdEMAMix does not support sparse gradients.")

            state = self.state[p]

            # State initialization
            if "step" not in state:
                state["step"] = 0
                # Exponential moving average of gradient values
                if beta1 != 0.0:  # save memory in case beta1 is 0.0
                    state["exp_avg_fast"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                else:
                    state["exp_avg_fast"] = None
                state["exp_avg_slow"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            exp_avg_fast, exp_avg_slow, exp_avg_sq = (
                state["exp_avg_fast"],
                state["exp_avg_slow"],
                state["exp_avg_sq"],
            )

            state["step"] += 1
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            # Compute the effective alpha and beta3 in case warmup is used
            if alpha_warmup is not None:
                alpha = linear_warmup_scheduler(
                    state["step"],
                    alpha_end=alpha_final,
                    alpha_start=0,
                    warmup=alpha_warmup,
                )
            else:
                alpha = alpha_final

            if beta3_warmup is not None:
                beta3 = linear_hl_warmup_scheduler(
                    state["step"],
                    beta_end=beta3_final,
                    beta_start=beta1,
                    warmup=beta3_warmup,
                )
            else:
                beta3 = beta3_final

            # Decay the first and second moment running average coefficient
            if beta1 != 0.0:
                exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
            else:
                exp_avg_fast = grad
            exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            update = (
                exp_avg_fast.div(bias_correction1) + alpha * exp_avg_slow
            ) / denom

            # decay
            update.add_(p, alpha=lmbda)

            p.add_(-lr * update)
