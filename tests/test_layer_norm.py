import torch
from torch import nn

from minopt.model import ConvLayerNorm
from minopt.utils import auto_device_and_dtype


def test_conv_layer_norm() -> None:
    """Tests that the ConvLayerNorm model matches the base later norm model."""

    _, device, _ = auto_device_and_dtype()
    is_mps = device == torch.device("mps")
    dtype = torch.float32 if is_mps else torch.float64
    eps = 1e-4 if is_mps else 1e-8
    weight = torch.randn(16, device=device, dtype=dtype)
    bias = torch.randn(16, device=device, dtype=dtype)

    norm = nn.LayerNorm(16, elementwise_affine=True, dtype=dtype)
    norm.weight.data.copy_(weight)
    norm.bias.data.copy_(bias)

    conv_norm = ConvLayerNorm(16, rank=4, elementwise_affine=True, dtype=dtype)
    conv_norm.weight.data.copy_(weight)
    conv_norm.bias.data.copy_(bias)

    x = torch.randn(1, 16, 1, 1, dtype=dtype)

    y_ref = norm(x.flatten(1)).unflatten(1, (-1, 1, 1))
    y_out = conv_norm(x)

    assert y_ref.shape == y_out.shape
    assert (y_ref - y_out).abs().max() < eps
