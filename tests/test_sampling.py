import torch

from minopt.model import opt
from minopt.sampling import Sampler
from minopt.utils import auto_device_and_dtype


def test_sampling_matches_training() -> None:
    """Tests that sampling matches training.

    This is a simple test which does greedy decoding of some tokens, then feeds those
    same tokens back into the training forward pass to check that it outputs the same
    argmax token values.
    """

    _, device, _ = auto_device_and_dtype()
    model = opt("opt_mini", device=device, dtype=torch.float32)
    sampler = Sampler(model, "greedy", 8).eval()
    tokens = torch.randint(0, model.vocab_size, (1, 4), device=device)
    prefix_len = tokens.shape[1]

    eval_tokens = sampler(tokens)
    pred_logits, _ = model(eval_tokens)  # pylint: disable=not-callable
    train_tokens = pred_logits.argmax(dim=1)
    tokens_eq = eval_tokens[:, prefix_len + 1 :] == train_tokens[:, prefix_len:-1]
    assert tokens_eq.all().item()
