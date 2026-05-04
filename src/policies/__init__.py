"""Policy classes — all satisfy the contract used by src.algorithms / src.train:

    policy.theta             flat torch.Tensor mutated in place by Adam
    policy.d                 int, dimension of theta
    policy.device, policy.dtype
    policy.action_probs_batch(state) -> ndarray (B, num_actions)
    policy.scores(states, actions)   -> torch.Tensor (..., d)

Submodules:
    nn_state    NNStatePolicy: MLP over one-hot state index.
"""
