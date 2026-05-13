from __future__ import annotations

from copy import deepcopy
import math

import torch


def resolve_device(name):
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else torch.device(name)
    )


class EMA:
    def __init__(self, model, decay=0.999):
        self.model, self.decay = deepcopy(model).eval(), decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            dest = self.model.state_dict()[name]
            if dest.is_floating_point():
                dest.mul_(self.decay).add_(value, alpha=1 - self.decay)
            else:
                dest.copy_(value)


def learning_rate(epoch, options):
    lr, minimum = float(options["lr"]), float(options.get("min_lr", 0))
    warmup, total = options.get("warmup_epochs", 0), options["epochs"]
    if epoch < warmup:
        start = float(options.get("start_lr", lr / 100))
        return start + (lr - start) * epoch / max(1, warmup - 1)
    progress = (epoch - warmup) / max(1, total - warmup - 1)
    return minimum + 0.5 * (lr - minimum) * (1 + math.cos(math.pi * min(1, progress)))


def load_checkpoint_model(path, stage, device="cpu"):
    from sern_mil.config import AppConfig, build_model
    from sern_mil.data.cache import load_tensor_cache
    from pathlib import Path

    ckpt = load_tensor_cache(path)
    if ckpt.get("stage") != stage:
        raise ValueError(f"Expected {stage} checkpoint, found {ckpt.get('stage')}")
    config = AppConfig(ckpt["config"], Path(path), Path(path).parent)
    model = build_model(config, stage)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval(), ckpt
