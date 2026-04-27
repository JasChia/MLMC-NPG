"""Diagnose vmap fallback + benchmark CPU-loop vs GPU-loop vs GPU-vmap for NNPolicy.scores."""
from __future__ import annotations

import time
import traceback
import numpy as np
import torch
from torch import nn

from src.policy import NNPolicy


def bench(label, fn, reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / reps


def run_trace_vmap(device):
    """Force the vmap path and surface the exact error if any."""
    policy = NNPolicy(M=2, L_max=10, device=device)
    policy._sync_to_net()
    q = np.random.randint(0, 11, size=(4, 500, 2))
    a = np.random.randint(0, 2, size=(4, 500))
    q_t = policy._normalize(q)
    a_t = torch.as_tensor(a, dtype=torch.long, device=device)
    flat_q = q_t.reshape(-1, 2)
    flat_a = a_t.reshape(-1)
    params = {n: p.detach() for n, p in policy.net.named_parameters()}

    def logprob_one(p_dict, s_single, a_single):
        logits = torch.func.functional_call(policy.net, p_dict, s_single)
        log_probs = torch.log_softmax(logits, dim=-1)
        idx = torch.arange(log_probs.shape[-1], device=log_probs.device)
        mask = (idx == a_single).to(log_probs.dtype)
        return (log_probs * mask).sum()

    try:
        per_sample_grads = torch.func.vmap(
            torch.func.grad(logprob_one, argnums=0),
            in_dims=(None, 0, 0),
        )(params, flat_q, flat_a)
        flat_scores = torch.cat(
            [per_sample_grads[n].reshape(flat_q.shape[0], -1) for n in policy._names], dim=1
        )
        return ("ok", flat_scores.shape, None)
    except Exception as e:
        return ("fail", None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")


def main():
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device:    {torch.cuda.get_device_name(0)}")
    print()

    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        print(f"--- device = {device} ---")
        status, shape, err = run_trace_vmap(device)
        print(f"vmap trace: status = {status}, shape = {shape}")
        if err:
            print(err)
        print()

    # Benchmark actual scores() wall time on representative batch sizes.
    print("--- scores() wall-time benchmark (NNPolicy, M=8) ---")
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        policy = NNPolicy(M=8, L_max=10, device=device)
        q_np = np.random.randint(0, 11, size=(16, 500, 8))
        a_np = np.random.randint(0, 8, size=(16, 500))
        def go():
            policy.scores(q_np, a_np)
        t = bench(f"{device}", go, reps=3, warmup=1)
        print(f"  device={device:4s}  N=16 H=500 M=8  d={policy.d}  time={t:.3f} s / call")

    print()
    print("--- scores() wall-time benchmark (M=2, N=16) ---")
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        policy = NNPolicy(M=2, L_max=10, device=device)
        q_np = np.random.randint(0, 11, size=(16, 500, 2))
        a_np = np.random.randint(0, 2, size=(16, 500))
        def go():
            policy.scores(q_np, a_np)
        t = bench(f"{device}", go, reps=3, warmup=1)
        print(f"  device={device:4s}  N=16 H=500 M=2  d={policy.d}  time={t:.3f} s / call")


if __name__ == "__main__":
    main()
