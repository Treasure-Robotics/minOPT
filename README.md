# minOPT

Minimal OPT implementation. Some features:

- [x] TorchScript compatible
- [ ] TensorRT compatible
- [ ] Int8 weights and inference
- [ ] Multi-GPU inference

## Getting Started

### Just Use It

- Use `torch.hub.load` from your own project:

```python
import torch

# Pre-trained weights are loaded separately; you need to set the `MODEL_DIR`
# environment variable to point at the root directory where you would like to
# download the model.
model = torch.hub.load("treasure-robotics/minopt", "opt_125m")

# All models have the same API. The KV caches can be dropped during training.
next_tokens, kv_caches = model(tokens)
```

- Alternatively, can clone and install:

```bash
pip install git+https://github.com/Treasure-Robotics/minOPT
```

Then:

```python
from minopt.model import opt

model = opt("opt_125m")
```

### Requirements

- Python 3.8
- PyTorch
- HuggingFace Transformers (for tokenizer)

### Sample from Pre-trained Model

```bash
python scripts/sample opt_125m  # Can be replaced with any model key
```

The stripped-down version of this is:

```python
from transformers import GPT2Tokenizer

from minopt.model import opt
from minopt.sampling import Sampler

model = opt("opt_125m")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")
sampler = Sampler(model, mode="greedy", max_steps=16)

# Converts some prompt from text to tokens.
prev_tokens = tokenizer(
  prompt,
  return_attention_mask=False,
  return_tensors="pt",
)["input_ids"]

# Gets the predicted next tokens.
sampled_tokens = sampler(prev_tokens)

# Converts back to text.
pred_tokens = tokenizer.decode(sampled_tokens[0, prev_tokens.shape[1] :])
```

## References

- [MinGPT](https://github.com/karpathy/minGPT)
- [Minimal OPT](https://github.com/zphang/minimal-opt)
