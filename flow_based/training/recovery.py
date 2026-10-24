"""Epoch-boundary recovery; never loads arbitrary pickled Python objects."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def array_digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def contract(model, arrays, cfg, seed, n_classes, device, context):
    return json.loads(
        json.dumps(
            {
                "version": 1,
                "seed": seed,
                "n_classes": n_classes,
                "train": cfg,
                "arrays": [array_digest(a) for a in arrays],
                "model": [(k, list(v.shape)) for k, v in model.state_dict().items()],
                "torch": str(torch.__version__),
                "device_type": device.type,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "context": context,
            },
            sort_keys=True,
        )
    )


def save(path, state, loader):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np_state = np.random.get_state()
    state["rng"] = {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader.generator.get_state(),
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    temp.replace(path)


def load(path, expected, model, optimizer, scheduler, loader):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("contract") != expected:
        raise ValueError(
            "Recovery checkpoint differs in data/config/source/device/software. "
            "Use its original environment or a new run directory."
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng = state["rng"]
    random.setstate(rng["python"])
    n = rng["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    loader.generator.set_state(rng["loader"])
    return state
