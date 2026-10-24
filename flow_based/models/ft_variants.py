"""FT ablations with the original architecture and tokenizer/CLS/head intact."""

import torch
from torch import nn

from flow_based.models.ft_transformer import build_transformer


def build_variant(n_features, n_classes, model_cfg, seed, independent_init=False):
    model = build_transformer(n_features, n_classes, model_cfg)
    if independent_init:
        for i in range(model_cfg["n_layers"]):
            # Construct CPU layers under a private CPU RNG state. CUDA and the
            # global dropout RNG sequence are unchanged by this ablation.
            with torch.random.fork_rng(devices=[]):
                local = torch.Generator(device="cpu").manual_seed(
                    seed + 10000 * (i + 1)
                )
                torch.set_rng_state(local.get_state())
                model.encoder.layers[i] = nn.TransformerEncoderLayer(
                    d_model=model_cfg["d_model"],
                    nhead=model_cfg["n_heads"],
                    dim_feedforward=model_cfg["d_ff"],
                    dropout=model_cfg["dropout"],
                    activation=model_cfg["activation"],
                    norm_first=model_cfg["norm_first"],
                    batch_first=True,
                    device="cpu",
                    dtype=torch.float32,
                )
    return model
