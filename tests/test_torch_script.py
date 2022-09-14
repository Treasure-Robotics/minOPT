from typing import get_args

import pytest
import torch

from minopt.model import opt
from minopt.sampling import Sampler, SamplingMode


@pytest.mark.parametrize("sampling_mode", get_args(SamplingMode))
def test_torch_script_compatible(sampling_mode: SamplingMode) -> None:
    """Simply tests that the sampler is exportable.

    The OPT model is kept on the Meta device for efficiency.

    Args:
        sampling_mode: The sampling mode to check
    """

    model = opt("opt_125m", keep_meta=True)
    sampler = Sampler(model, sampling_mode, max_steps=16)
    torch.jit.script(sampler)
