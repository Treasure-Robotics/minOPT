#!/usr/bin/env python
"""Allows OPT models to be loaded using `torch.hub.load`

Example usage:

    ```python
    model = torch.hub.load(
        "treasure-robotics/minopt",
        "opt_125m",
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )

    # All models have the same API.
    next_tokens, kv_caches = model(tokens)
    ```
"""

dependencies = ["torch"]

# pylint: disable=wrong-import-position
# pylint: disable=unused-import
# pylint: disable=unnecessary-lambda-assignment

from minopt.model import opt as _opt  # noqa

opt_125m = lambda *args, **kwargs: _opt("opt_125m", *args, **kwargs)
opt_1300m = lambda *args, **kwargs: _opt("opt_1.3b", *args, **kwargs)
opt_2700m = lambda *args, **kwargs: _opt("opt_2.7b", *args, **kwargs)
opt_6700m = lambda *args, **kwargs: _opt("opt_6.7b", *args, **kwargs)
opt_13b = lambda *args, **kwargs: _opt("opt_13b", *args, **kwargs)
opt_30b = lambda *args, **kwargs: _opt("opt_30b", *args, **kwargs)
opt_66b = lambda *args, **kwargs: _opt("opt_66b", *args, **kwargs)
