"""NN policy compatible with torch < 2.0 (no torch.func.vmap).

Subclasses src.policy.NNPolicy and overrides scores() to skip the vmap
preferred path (which AttributeErrors on torch 1.x) and go straight to the
per-sample autograd loop. Algorithmically identical to NNPolicy — only
differs in how the per-sample gradient computation is dispatched.
"""
from __future__ import annotations

import numpy as np
import torch

from src.policy import NNPolicy


class NNPolicyCompat(NNPolicy):
    def scores(self, q_batch, a_batch) -> torch.Tensor:
        self._sync_to_net()
        q_t = self._normalize(q_batch)
        a_t = torch.as_tensor(a_batch, dtype=torch.long, device=self.device)
        orig_shape = q_t.shape[:-1]
        flat_q = q_t.reshape(-1, self.M)
        flat_a = a_t.reshape(-1)
        N_flat = flat_q.shape[0]

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
