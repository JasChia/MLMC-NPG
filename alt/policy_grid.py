"""Tabular softmax policy on a finite (state-index, action-index) MDP.

Designed for the Deep Sea Treasure env in alt/envs_dst.py. Implements the
narrow contract used by src/algorithms.py:
  - policy.theta            : flat torch.Tensor of shape (d,), mutated by Adam
  - policy.d                : int
  - policy.M                : number of actions (used by algorithms via reward shape)
  - policy.action_probs_batch(state_idx_np)  -> ndarray (B, num_actions)
  - policy.scores(state_idx_batch, action_batch) -> torch.Tensor (..., d)

Score: ∇log π(a|s) for tabular softmax has a particularly simple form,
nonzero only in the slot for the visited state s.
"""
from __future__ import annotations

import numpy as np
import torch


class TabularGridPolicy:
    def __init__(self, num_states: int, num_actions: int,
                 device: str = "cpu", dtype=torch.float32):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        # Algorithms read `policy.M` to interpret the reward shape (B, H, M).
        # Here M means "action dimension" in the policy sense; the env's reward
        # dimension can differ. We expose num_actions for clarity in callers
        # but keep .M for algorithms uniformity.
        self.M = self.num_actions
        self.d = self.num_states * self.num_actions
        self.device = device
        self.dtype = dtype
        self.theta = torch.zeros(self.d, device=device, dtype=dtype)

    def _theta_matrix(self) -> torch.Tensor:
        return self.theta.view(self.num_states, self.num_actions)

    def logits(self, s) -> torch.Tensor:
        idx = torch.as_tensor(s, dtype=torch.long, device=self.device)
        return self._theta_matrix()[idx]

    def action_probs(self, s) -> torch.Tensor:
        return torch.softmax(self.logits(s), dim=-1)

    def action_probs_batch(self, s_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.action_probs(s_np).cpu().numpy()

    def scores(self, s_batch, a_batch) -> torch.Tensor:
        s_t = torch.as_tensor(s_batch, dtype=torch.long, device=self.device)
        a_t = torch.as_tensor(a_batch, dtype=torch.long, device=self.device)
        # Compute per-(state,action) one-hot minus pi(a|s) within the visited row.
        probs = self.action_probs(s_t)         # (..., A)
        e_a = torch.zeros_like(probs)
        e_a.scatter_(-1, a_t.unsqueeze(-1), 1.0)
        delta = e_a - probs                    # (..., A)

        out_shape = s_t.shape
        flat_n = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
        idx_flat = s_t.reshape(flat_n)
        delta_flat = delta.reshape(flat_n, self.num_actions)

        scores = torch.zeros(flat_n, self.d, dtype=self.dtype, device=self.device)
        row_start = idx_flat * self.num_actions
        cols = row_start.unsqueeze(-1) + torch.arange(self.num_actions, device=self.device)
        rows = torch.arange(flat_n, device=self.device).unsqueeze(-1).expand_as(cols)
        scores[rows, cols] = delta_flat
        return scores.reshape(*out_shape, self.d)
