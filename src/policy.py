"""Policy classes for the M-queue server.

All three classes (LogLinearPolicy, TabularPolicy, NNPolicy) satisfy the same
narrow contract used by algorithms.py and train.py:

    policy.theta             : flat torch.Tensor, mutated in place by Adam
    policy.d                 : int, dimension of theta
    policy.device, dtype     : torch device / dtype
    policy.action_probs_batch(q_np) -> np.ndarray of shape (B, M)
    policy.scores(q_batch, a_batch) -> torch.Tensor (..., d)

LogLinearPolicy is the default — it is the consistent policy class used for
all three algorithms in the main experiments.  TabularPolicy is a special
case of LogLinearPolicy with a one-hot state feature; feasible for M=2, too
large for M>=4.  NNPolicy uses torch.func.vmap + grad for per-sample scores.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch
from torch import nn


# -----------------------------------------------------------------------------
# Abstract base class
# -----------------------------------------------------------------------------

class Policy(ABC):
    theta: torch.Tensor
    d: int
    device: str
    dtype: torch.dtype

    @abstractmethod
    def action_probs_batch(self, q_np: np.ndarray) -> np.ndarray:
        """Per-state action probability table; shape (B, M)."""

    @abstractmethod
    def scores(self, q_batch, a_batch) -> torch.Tensor:
        """Flat score ∇_θ log π(a|s) for a batch of (s, a)."""


# -----------------------------------------------------------------------------
# Log-linear softmax — features psi(s,a) = e_a ⊗ [1, q_1/L, ..., q_M/L]
# -----------------------------------------------------------------------------

class LogLinearPolicy(Policy):
    def __init__(self, M: int, L_max: int, device: str = "cpu", dtype=torch.float32):
        self.M = M
        self.L_max = L_max
        self.d = M * (M + 1)
        self.device = device
        self.dtype = dtype
        self.theta = torch.zeros(self.d, device=device, dtype=dtype)

    def _phi(self, q) -> torch.Tensor:
        q_t = torch.as_tensor(q, dtype=self.dtype, device=self.device)
        ones = torch.ones(q_t.shape[:-1] + (1,), dtype=self.dtype, device=self.device)
        return torch.cat([ones, q_t / self.L_max], dim=-1)

    def _theta_matrix(self) -> torch.Tensor:
        return self.theta.view(self.M, self.M + 1)

    def logits(self, q) -> torch.Tensor:
        return self._phi(q) @ self._theta_matrix().T

    def action_probs(self, q) -> torch.Tensor:
        return torch.softmax(self.logits(q), dim=-1)

    def action_probs_batch(self, q_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.action_probs(q_np).cpu().numpy()

    def scores(self, q_batch, a_batch) -> torch.Tensor:
        q_t = torch.as_tensor(q_batch, dtype=self.dtype, device=self.device)
        a_t = torch.as_tensor(a_batch, dtype=torch.long, device=self.device)
        phi = self._phi(q_t)
        logits = phi @ self._theta_matrix().T
        probs = torch.softmax(logits, dim=-1)
        e_a = torch.zeros_like(probs)
        e_a.scatter_(-1, a_t.unsqueeze(-1), 1.0)
        delta = e_a - probs
        outer = delta.unsqueeze(-1) * phi.unsqueeze(-2)
        return outer.reshape(*outer.shape[:-2], self.d)


# -----------------------------------------------------------------------------
# Tabular softmax — features psi(s,a) = e_s ⊗ e_a (identity in state index)
# -----------------------------------------------------------------------------

class TabularPolicy(Policy):
    """Full tabular softmax, pi(a|s) = softmax(theta[s, a]).

    Shares the same score formula as LogLinearPolicy with phi(s) replaced by
    a one-hot state indicator.  Feasible memory-wise only for small (M, L_max)
    since d = (L_max+1)^M * M and the NPG inner loop forms a (d, d) Fisher.
    """

    def __init__(self, M: int, L_max: int, device: str = "cpu", dtype=torch.float32):
        self.M = M
        self.L_max = L_max
        self.num_states = (L_max + 1) ** M
        self.d = self.num_states * M
        self.device = device
        self.dtype = dtype
        self.theta = torch.zeros(self.d, device=device, dtype=dtype)
        self._base = torch.tensor(
            [(L_max + 1) ** i for i in range(M)],
            device=device, dtype=torch.long,
        )

    def _state_idx(self, q) -> torch.Tensor:
        q_t = torch.as_tensor(q, dtype=torch.long, device=self.device)
        return (q_t * self._base).sum(dim=-1)  # (...,)

    def _theta_matrix(self) -> torch.Tensor:
        return self.theta.view(self.num_states, self.M)

    def logits(self, q) -> torch.Tensor:
        idx = self._state_idx(q)  # (...,)
        return self._theta_matrix()[idx]  # (..., M)

    def action_probs(self, q) -> torch.Tensor:
        return torch.softmax(self.logits(q), dim=-1)

    def action_probs_batch(self, q_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.action_probs(q_np).cpu().numpy()

    def scores(self, q_batch, a_batch) -> torch.Tensor:
        """Score has only M non-zero entries (within the block of the visited state).

        We still return a dense (..., d) tensor for interface uniformity; for
        moderately sized state spaces this is acceptable, but for very large
        d you'd want a sparse representation.
        """
        q_t = torch.as_tensor(q_batch, dtype=torch.long, device=self.device)
        a_t = torch.as_tensor(a_batch, dtype=torch.long, device=self.device)
        idx = self._state_idx(q_t)  # (...,)
        probs = self.action_probs(q_t)  # (..., M)
        e_a = torch.zeros_like(probs)
        e_a.scatter_(-1, a_t.unsqueeze(-1), 1.0)
        delta = e_a - probs  # (..., M)

        out_shape = q_t.shape[:-1]
        flat_shape = (int(np.prod(out_shape)),) if len(out_shape) > 0 else (1,)
        idx_flat = idx.reshape(flat_shape)
        delta_flat = delta.reshape(*flat_shape, self.M)

        scores = torch.zeros(*flat_shape, self.d, dtype=self.dtype, device=self.device)
        # Write delta[i, :] into scores[i, idx_flat[i] * M : idx_flat[i] * M + M]
        # Using advanced indexing:
        row_start = idx_flat * self.M  # (N,)
        cols = row_start.unsqueeze(-1) + torch.arange(self.M, device=self.device)  # (N, M)
        rows = torch.arange(flat_shape[0], device=self.device).unsqueeze(-1).expand_as(cols)
        scores[rows, cols] = delta_flat
        return scores.reshape(*out_shape, self.d)


# -----------------------------------------------------------------------------
# Neural network — MLP on queue-length input, logits over M actions
# -----------------------------------------------------------------------------

class NNPolicy(Policy):
    """MLP policy: logits = MLP(q / L_max), pi(a|s) = softmax(logits).

    theta is a flat concatenation of all MLP parameters; we call _sync_to_net()
    at the start of every forward pass so Adam's in-place update of theta is
    mirrored in the nn.Module.  Per-sample scores use torch.func.vmap + grad.
    """

    def __init__(self, M: int, L_max: int, hidden: Sequence[int] = (64, 64),
                 device: str = "cpu", dtype=torch.float32):
        self.M = M
        self.L_max = L_max
        self.device = device
        self.dtype = dtype

        layers, prev = [], M
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, M)]
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

    def _normalize(self, q) -> torch.Tensor:
        return torch.as_tensor(q, dtype=self.dtype, device=self.device) / self.L_max

    def action_probs(self, q) -> torch.Tensor:
        self._sync_to_net()
        return torch.softmax(self.net(self._normalize(q)), dim=-1)

    def action_probs_batch(self, q_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.action_probs(q_np).cpu().numpy()

    def scores(self, q_batch, a_batch) -> torch.Tensor:
        """Per-sample score = grad of log pi(a|s) w.r.t. every parameter.

        Preferred path: torch.func.vmap(grad(...)) for batched per-sample grads.
        If vmap chokes on an internal op, fall back to a Python autograd loop
        (slow but always correct).
        """
        self._sync_to_net()
        q_t = self._normalize(q_batch)
        a_t = torch.as_tensor(a_batch, dtype=torch.long, device=self.device)
        orig_shape = q_t.shape[:-1]
        flat_q = q_t.reshape(-1, self.M)
        flat_a = a_t.reshape(-1)
        N_flat = flat_q.shape[0]

        try:
            params = {n: p.detach() for n, p in self.net.named_parameters()}

            def logprob_one(p_dict, s_single, a_single):
                logits = torch.func.functional_call(self.net, p_dict, s_single)
                log_probs = torch.log_softmax(logits, dim=-1)
                # Arithmetic one-hot: avoids F.one_hot (which internally .item()s).
                idx = torch.arange(log_probs.shape[-1], device=log_probs.device)
                mask = (idx == a_single).to(log_probs.dtype)
                return (log_probs * mask).sum()

            per_sample_grads = torch.func.vmap(
                torch.func.grad(logprob_one, argnums=0),
                in_dims=(None, 0, 0),
            )(params, flat_q, flat_a)
            flat_scores = torch.cat(
                [per_sample_grads[n].reshape(N_flat, -1) for n in self._names], dim=1
            )
        except RuntimeError as _e:
            if not getattr(self, "_warned_vmap", False):
                import sys
                print(f"[NNPolicy] vmap failed, using per-sample autograd loop: {_e}",
                      file=sys.stderr)
                self._warned_vmap = True
            # Slow fallback: per-sample autograd loop.
            for p in self.net.parameters():
                p.requires_grad_(True)
            per_sample = []
            for i in range(N_flat):
                self.net.zero_grad(set_to_none=True)
                logits = self.net(flat_q[i])
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


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def make_policy(kind: str, M: int, L_max: int, device: str = "cpu",
                dtype=torch.float32, **kwargs) -> Policy:
    if kind == "log_linear":
        return LogLinearPolicy(M, L_max, device=device, dtype=dtype)
    if kind == "tabular":
        return TabularPolicy(M, L_max, device=device, dtype=dtype)
    if kind == "nn":
        hidden = kwargs.get("hidden", (64, 64))
        return NNPolicy(M, L_max, hidden=hidden, device=device, dtype=dtype)
    raise ValueError(f"Unknown policy kind: {kind}")
