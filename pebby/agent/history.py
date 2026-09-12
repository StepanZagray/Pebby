"""Caller-owned causal history for temporal policies; no environment access."""

import torch

class PolicyHistory:
    def __init__(self, policy, device=None):
        self.policy = policy
        self.device = device
        self.length = policy.config().get("history", 1)
        self.frames = []
        self.actions = []

    def observe(self, frame, action_index=-1, *, reset=False):
        if reset:
            self.frames.clear()
            self.actions.clear()
            action_index = -1
        self.frames.append(frame)
        self.actions.append(action_index)
        self.frames = self.frames[-self.length:]
        self.actions = self.actions[-self.length:]

    def scores(self):
        if not self.frames:
            raise ValueError("observe an initial frame before requesting scores")
        padding = self.length - len(self.frames)
        frames = [self.frames[0]] * padding + self.frames
        valid = [False] * padding + [True] * len(self.frames)
        actions = [-1] * padding + self.actions
        return self.policy(torch.as_tensor(frames, device=self.device)[None],
                           history_valid=torch.tensor(valid, device=self.device)[None],
                           previous_actions=torch.tensor(actions, device=self.device)[None])[0].float()


def for_policy(policy, frame, device=None):
    if not hasattr(policy, "config") or policy.config().get("architecture") not in ("world", "structured"):
        return None
    history = PolicyHistory(policy, device)
    history.observe(frame)
    return history
