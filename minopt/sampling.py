from typing import Literal

import torch
from torch import Tensor, nn

from minopt.model import OPT

SamplingMode = Literal["multinomial", "greedy", "nucleus"]


def sample_step(next_logits: Tensor, mode: str, nucleus_prob: float) -> Tensor:
    """Does a single sampling step on a given set of logits.

    Args:
        next_logits: The logits to sample from, with shape (B, C)
        mode: The sampling mode to use
        nucleus_prob: Nucleus sampling probability

    Returns:
        The sampled tokens, with shape (B, 1)

    Raises:
        NotImplementedError: If the mode is invalid
    """

    if mode == "multinomial":
        return torch.multinomial(next_logits.softmax(1), num_samples=1)

    if mode == "greedy":
        return next_logits.argmax(dim=1, keepdim=True)

    if mode == "nucleus":
        sorted_logits, inds = next_logits.sort(dim=1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=1)
        probs_cum_sum = torch.cumsum(sorted_probs, dim=1)
        min_prob = nucleus_prob + probs_cum_sum[..., :1]
        mask_inds = torch.masked_select(inds, probs_cum_sum > min_prob)
        next_logits = next_logits.index_fill(1, mask_inds, -1e4)
        return torch.multinomial(next_logits.softmax(1), num_samples=1)

    raise NotImplementedError(f"Invalid mode: {mode}")


class Sampler(nn.Module):
    __constants__ = ["mode", "nucleus_prob"]

    def __init__(
        self,
        model: OPT,
        mode: SamplingMode,
        *,
        nucleus_prob: float = 0.8,
    ) -> None:
        """Defines a wrapper module to sample from an OPT model.

        Args:
            model: The model to sample from
            mode: The sampling mode to use
            nucleus_prob: Nucleus sampling probability
        """

        super().__init__()

        self.model = model
        self.mode = mode
        self.nucleus_prob = nucleus_prob

    def sample(self, prev_token: Tensor, max_steps: int) -> Tensor:
        """Samples a sequence for a given prefix.

        Args:
            prev_token: The prefix to use, with shape (B, T)
            max_steps: The maximum number of steps to sample

        Returns:
            The sampled tokens, with shape (B, T)
        """

        offset = 0
        all_tokens = prev_token

        # Runs the first step to get the caches.
        next_logits, kv_caches = self.model(prev_token)
        offset += next_logits.size(2)
        prev_token = sample_step(next_logits[..., -1], self.mode, self.nucleus_prob)
        all_tokens = torch.cat((all_tokens, prev_token), dim=1)

        for _ in range(max_steps):
            next_logits, kv_caches = self.model(
                prev_token,
                kv_caches=kv_caches,
                offset=offset,
            )
            offset += next_logits.size(2)
            next_logits = next_logits[..., -1]
            prev_token = sample_step(next_logits, self.mode, self.nucleus_prob)
            all_tokens = torch.cat((all_tokens, prev_token), dim=1)

        return all_tokens

    def forward(self, prev_token: Tensor, max_steps: int) -> Tensor:
        return self.sample(prev_token, max_steps)
