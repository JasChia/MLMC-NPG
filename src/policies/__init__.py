"""Policy classes — all satisfy the contract used by src.algorithms / src.train:

    policy.theta             flat torch.Tensor mutated in place by Adam
    policy.d                 int, dimension of theta
    policy.device, policy.dtype
    policy.action_probs_batch(state) -> ndarray (B, num_actions)
    policy.scores(states, actions)   -> torch.Tensor (..., d)

Submodules:
    log_linear  LogLinearPolicy / TabularPolicy / NNPolicy for the M-queue server
                (Paper A — state is a length-M queue-length vector).
    tabular     TabularGridPolicy for finite (state-index, action) MDPs (DST family).
    nn_state    NNStatePolicy: MLP over one-hot state index (DST family / Fruit Tree).
"""
