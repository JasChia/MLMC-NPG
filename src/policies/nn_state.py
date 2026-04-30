"""NN policy for tabular-state envs (DST family, Resource Gathering, Fruit
Tree, Bonus World). Takes int state-index as input, one-hot encodes it,
passes through an MLP, and emits action logits.

Used for d-scaling experiments: vary the hidden width to control the
parameter count d, holding env fixed.

Implements the same narrow contract as TabularGridPolicy / NNPolicy:
  - policy.theta            : flat torch.Tensor of shape (d,)
  - policy.d                : int
  - policy.M                : num_actions
  - policy.action_probs_batch(state_idx_np)  -> ndarray (B, num_actions)
  - policy.scores(state_idx_batch, action_batch) -> torch.Tensor (..., d)

Per-sample autograd loop fallback (compatible with torch < 2.0).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU,
                "sigmoid": nn.Sigmoid, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}


class NNStatePolicy:
    def __init__(self, num_states: int, num_actions: int,
                 hidden: Sequence[int] = (32,),
                 activation: str = "tanh",
                 device: str = "cpu", dtype=torch.float32):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.M = self.num_actions
        self.device = device
        self.dtype = dtype

        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation}; pick from {list(_ACTIVATIONS)}")
        Act = _ACTIVATIONS[activation]
        layers, prev = [], num_states
        for h in hidden:
            layers += [nn.Linear(prev, h), Act()]
            prev = h
        layers += [nn.Linear(prev, num_actions)]
        self.net = nn.Sequential(*layers).to(device=device, dtype=dtype)

        # Initialize last layer near zero so policy starts near uniform.
        with torch.no_grad():
            last_linear = self.net[-1]
            last_linear.weight.mul_(0.01)
            last_linear.bias.zero_()

        self._shapes = [p.shape for p in self.net.parameters()]
        self._sizes = [p.numel() for p in self.net.parameters()]
        self._names = [n for n, _ in self.net.named_parameters()]
        self.d = sum(self._sizes)
        with torch.no_grad():
            self.theta = torch.cat([p.flatten() for p in self.net.parameters()]).clone()

    def _sync_to_net(self) -> None:
        offset = 0
        with torch.no_grad():
            for p, n, shp in zip(self.net.parameters(), self._sizes, self._shapes):
                p.data.copy_(self.theta[offset:offset + n].view(shp))
                offset += n

    def _onehot(self, s) -> torch.Tensor:
        s_t = torch.as_tensor(s, dtype=torch.long, device=self.device)
        flat = s_t.reshape(-1)
        # One-hot: (N_flat, num_states)
        oh = torch.zeros(flat.shape[0], self.num_states,
                         dtype=self.dtype, device=self.device)
        oh.scatter_(1, flat.unsqueeze(-1), 1.0)
        return oh.reshape(*s_t.shape, self.num_states)

    def action_probs(self, s) -> torch.Tensor:
        self._sync_to_net()
        x = self._onehot(s)
        # Flatten batch dims for net forward
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.num_states)
        logits = self.net(x_flat).reshape(*orig_shape, self.num_actions)
        return torch.softmax(logits, dim=-1)

    def action_probs_batch(self, s_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.action_probs(s_np).cpu().numpy()

    def scores(self, s_batch, a_batch) -> torch.Tensor:
        """Per-sample score = grad of log pi(a|s) w.r.t. every parameter.

        Prefer a batched torch.func path; fall back to a Python autograd loop on
        older torch builds or unsupported ops.
        """
        self._sync_to_net()
        s_t = torch.as_tensor(s_batch, dtype=torch.long, device=self.device)
        a_t = torch.as_tensor(a_batch, dtype=torch.long, device=self.device)
        orig_shape = s_t.shape
        flat_s = s_t.reshape(-1)
        flat_a = a_t.reshape(-1)
        N_flat = flat_s.shape[0]

        if len(self._shapes) == 4:
            # Fast analytic score for the one-hidden-layer MLP used by NN-32
            # pilots: Linear -> Tanh -> Linear -> softmax.
            with torch.no_grad():
                first = self.net[0]
                last = self.net[2]
                flat_oh = torch.zeros(N_flat, self.num_states,
                                      dtype=self.dtype, device=self.device)
                flat_oh.scatter_(1, flat_s.unsqueeze(-1), 1.0)
                hidden = torch.tanh(flat_oh @ first.weight.T + first.bias)
                logits = hidden @ last.weight.T + last.bias
                probs = torch.softmax(logits, dim=-1)
                delta = torch.zeros_like(probs)
                delta.scatter_(1, flat_a.unsqueeze(-1), 1.0)
                delta -= probs

                grad_w2 = delta.unsqueeze(-1) * hidden.unsqueeze(1)
                grad_b2 = delta
                grad_hidden = (delta @ last.weight) * (1.0 - hidden * hidden)
                grad_b1 = grad_hidden
                grad_w1 = torch.zeros(N_flat, first.weight.shape[0],
                                      self.num_states, dtype=self.dtype,
                                      device=self.device)
                rows = torch.arange(N_flat, device=self.device).unsqueeze(-1)
                hidden_idx = torch.arange(first.weight.shape[0],
                                          device=self.device).unsqueeze(0)
                grad_w1[rows, hidden_idx, flat_s.unsqueeze(-1)] = grad_hidden
                flat_scores = torch.cat([
                    grad_w1.reshape(N_flat, -1),
                    grad_b1,
                    grad_w2.reshape(N_flat, -1),
                    grad_b2,
                ], dim=1)
            return flat_scores.reshape(*orig_shape, self.d)

        try:
            flat_oh = torch.zeros(N_flat, self.num_states,
                                  dtype=self.dtype, device=self.device)
            flat_oh.scatter_(1, flat_s.unsqueeze(-1), 1.0)
            params = {n: p.detach() for n, p in self.net.named_parameters()}

            def logprob_one(p_dict, oh_single, a_single):
                logits = torch.func.functional_call(self.net, p_dict, oh_single)
                log_probs = torch.log_softmax(logits, dim=-1)
                idx = torch.arange(log_probs.shape[-1], device=log_probs.device)
                mask = (idx == a_single).to(log_probs.dtype)
                return (log_probs * mask).sum()

            per_sample_grads = torch.func.vmap(
                torch.func.grad(logprob_one, argnums=0),
                in_dims=(None, 0, 0),
            )(params, flat_oh, flat_a)
            flat_scores = torch.cat(
                [per_sample_grads[n].reshape(N_flat, -1) for n in self._names],
                dim=1,
            )
        except (RuntimeError, AttributeError) as _e:
            if not getattr(self, "_warned_vmap", False):
                import sys
                print(f"[NNStatePolicy] vmap failed, using per-sample autograd loop: {_e}",
                      file=sys.stderr)
                self._warned_vmap = True
            for p in self.net.parameters():
                p.requires_grad_(True)

            per_sample = []
            for i in range(N_flat):
                self.net.zero_grad(set_to_none=True)
                oh = torch.zeros(self.num_states, dtype=self.dtype, device=self.device)
                oh[flat_s[i]] = 1.0
                logits = self.net(oh)
                log_probs = torch.log_softmax(logits, dim=-1)
                idx = torch.arange(log_probs.shape[-1], device=log_probs.device)
                mask = (idx == flat_a[i]).to(log_probs.dtype)
                lp = (log_probs * mask).sum()
                grads = torch.autograd.grad(lp, list(self.net.parameters()),
                                            retain_graph=False, create_graph=False)
                per_sample.append(torch.cat([g.flatten() for g in grads]).detach())

            for p in self.net.parameters():
                p.requires_grad_(False)
            flat_scores = torch.stack(per_sample, dim=0)
        return flat_scores.reshape(*orig_shape, self.d)
