from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F

from sern_mil.config import build_model
from sern_mil.core.seeding import seed_everything
from sern_mil.data.cache import load_manifest, load_tensor_cache, to_device, validate_splits
from sern_mil.models.survival.cox_head import fit_baseline
from sern_mil.training.loops import EMA, learning_rate, resolve_device
from sern_mil.training.losses import SERLoss, cox_efron_loss, distillation_loss

MANIFESTS = {
    "ser": "paths.ser_manifest",
    "student": "paths.student_manifest",
    "mil": "paths.mil_manifest",
    "survival": "paths.survival_manifest",
}


@dataclass
class Trainer:
    cfg: object
    stage: str = "mil"

    def _items(self, rows, training=False):
        order = list(range(len(rows)))
        if training:
            order = torch.randperm(len(rows)).tolist()
            if self.stage == "ser":
                boundary = [i for i, r in enumerate(rows) if r.get("has_boundary", False)]
                ordinary = [i for i in order if i not in boundary]
                if boundary and ordinary:
                    p = self.options.get("boundary_sampling_probability", 0.7)
                    weights = torch.tensor(
                        [
                            p / len(boundary) if i in boundary else (1 - p) / len(ordinary)
                            for i in range(len(rows))
                        ]
                    )
                    order = torch.multinomial(weights, len(rows), replacement=True).tolist()
        for i in order:
            row = rows[i]
            item = load_tensor_cache(row["cache_path"], self.cfg.root_dir)
            if self.stage == "student":
                size = self.options.get("batch_size", 1024)
                n = len(item["embeddings"])
                permutation = torch.randperm(n) if training else torch.arange(n)
                for start in range(0, n, size):
                    idx = permutation[start : start + size]
                    yield {k: item[k][idx] for k in ("embeddings", "neighborhoods", "targets")}
            else:
                item["label"] = item.get("label", row.get("label"))
                yield item

    def _loss(self, model, item, epoch):
        if self.stage == "ser":
            output = model(item["tokens"].unsqueeze(0), item["shallow_features"].unsqueeze(0))
            loss = SERLoss()(
                output["tumor_logits"],
                output["boundary_logits"],
                item["tumor_target"].unsqueeze(0),
                item["boundary_target"].unsqueeze(0),
                epoch,
            )
            return loss, 1, None, None
        if self.stage == "student":
            predicted = model(item["embeddings"], item["neighborhoods"])
            return distillation_loss(predicted, item["targets"]), len(predicted), None, None
        gp = item["label"]
        if gp not in (3, 4, 5):
            raise ValueError("Regional bag labels must be GP3, GP4, or GP5")
        if "features" in item and not self.cfg.get("training.mil.pretrained_nfi"):
            raise ValueError("Final cached instance features require training.mil.pretrained_nfi")
        output = model(item)
        target = torch.tensor([gp - 3], device=self.device)
        loss = F.cross_entropy(output["logits"], target)
        return loss, 1, gp, int(output["logits"].argmax(-1)) + 3

    def _epoch(self, model, rows, epoch, optimizer=None, ema=None):
        training = optimizer is not None
        model.train(training)
        loss_sum, count, truth, pred = 0.0, 0, [], []
        accumulation = self.options.get("accumulation_steps", 1) if training else 1
        if training:
            optimizer.zero_grad(set_to_none=True)
        pending = 0

        def step(remainder):
            self.scaler.unscale_(optimizer)
            if remainder < accumulation:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(accumulation / remainder)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.cfg.get("training.grad_clip_norm", 1.0)
            )
            previous_scale = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= previous_scale:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        for item in self._items(rows, training):
            item = to_device(item, self.device)
            with (
                torch.set_grad_enabled(training),
                torch.autocast(device_type=self.device.type, enabled=self.amp),
            ):
                loss, weight, label, guess = self._loss(model, item, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training/validation loss")
            loss_sum += float(loss.detach()) * weight
            count += weight
            if label is not None:
                truth.append(label)
                pred.append(guess)
            if training:
                self.scaler.scale(loss / accumulation).backward()
                pending += 1
                if pending == accumulation:
                    step(pending)
                    pending = 0
        if training and pending:
            step(pending)
        if count == 0:
            raise ValueError("No usable samples in the selected split")
        result = {"loss": loss_sum / count}
        if truth:
            result["macro_f1"] = float(
                f1_score(truth, pred, labels=[3, 4, 5], average="macro", zero_division=0)
            )
        return result

    def _save(self, model, epoch, history, path, **extra):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stage": self.stage,
                "config": self.cfg.raw,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "history": history,
                **extra,
            },
            path,
        )

    def fit(self, seed=None):
        seed = self.cfg.get("project.seed", 42) if seed is None else seed
        seed_everything(seed)
        self.options = self.cfg.get(f"training.{self.stage}", {})
        if self.options.get("epochs", 0) < 1:
            raise ValueError(f"training.{self.stage}.epochs must be positive")
        rows = load_manifest(self.cfg, MANIFESTS[self.stage])
        validate_splits(rows, grading=self.stage != "survival")
        train = [r for r in rows if r["split"] == "train"]
        validation = [r for r in rows if r["split"] == "val"]
        if not train or (self.stage != "survival" and not validation):
            raise ValueError("Training and validation rows are required for grading/student stages")
        self.device = resolve_device(self.cfg.get("training.device", "auto"))
        self.amp = self.device.type == "cuda" and self.cfg.get("training.mixed_precision", True)
        # cuda.amp remains available in the manuscript's PyTorch 2.2 runtime.
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp)
            if hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        model = build_model(self.cfg, self.stage).to(self.device)
        if self.stage == "mil" and self.cfg.get("training.mil.pretrained_nfi"):
            checkpoint = load_tensor_cache(self.cfg.path("training.mil.pretrained_nfi"))
            state = {
                k.removeprefix("nfi."): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("nfi.")
            }
            model.nfi.load_state_dict(state, strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.options["lr"]),
            weight_decay=float(self.options.get("weight_decay", 0)),
        )
        ema = EMA(
            model, 0.0 if self.stage == "student" else self.cfg.get("training.ema_decay", 0.999)
        )
        if self.stage == "survival":
            return self._fit_survival(model, optimizer, ema, train, seed)
        path = self.cfg.path("paths.checkpoints_dir") / f"{self.stage}_best.pt"
        best, bad, history = -float("inf"), 0, []
        for epoch in range(self.options["epochs"]):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch, self.options)
            training = self._epoch(model, train, epoch, optimizer, ema)
            evaluated = self._epoch(ema.model, validation, epoch)
            score = evaluated["macro_f1"] if self.stage == "mil" else -evaluated["loss"]
            history.append(
                {
                    "epoch": epoch + 1,
                    "train": training,
                    "val": evaluated,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            if score > best:
                best, bad = score, 0
                self._save(ema.model, epoch + 1, history, path, seed=seed)
            else:
                bad += 1
            if bad >= self.options.get("patience", self.options["epochs"]):
                break
        return {
            "stage": self.stage,
            "checkpoint": str(path),
            "epochs": len(history),
            "best_validation_score": best,
            "seed": seed,
        }

    def _fit_survival(self, model, optimizer, ema, rows, seed):
        patients_ids = [(r["dataset"], str(r["patient_id"])) for r in rows]
        if len(set(patients_ids)) != len(patients_ids):
            raise ValueError("BCR training requires one record per patient")
        patients = [load_tensor_cache(row["cache_path"], self.cfg.root_dir) for row in rows]
        if any(len(p["morphology"]) != 14 for p in patients):
            raise ValueError("BCR requires the study's supplied 14-D morphology vector")
        x = torch.stack([torch.cat([p["pooled"], p["morphology"]]) for p in patients]).to(
            self.device
        )
        time = torch.tensor([float(p["time"]) for p in patients], device=self.device)
        event = torch.tensor([int(p["event"]) for p in patients], device=self.device)
        if not ((event == 0) | (event == 1)).all() or not event.sum():
            raise ValueError("BCR training needs binary event indicators and at least one event")
        history = []
        for epoch in range(self.options["epochs"]):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch, self.options)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = cox_efron_loss(model(x), time, event)
            regularization = self.options.get("l2_lambda", 0.0)
            if regularization:
                loss = loss + regularization * sum(p.square().sum() for p in model.parameters())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Cox loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.cfg.get("training.grad_clip_norm", 1.0)
            )
            optimizer.step()
            ema.update(model)
            history.append({"epoch": epoch + 1, "loss": float(loss.detach())})
        with torch.no_grad():
            baseline = fit_baseline(ema.model(x).detach().cpu(), time.cpu(), event.cpu())
        path = self.cfg.path("paths.checkpoints_dir") / f"survival_seed{seed}.pt"
        self._save(ema.model, len(history), history, path, baseline=baseline, seed=seed)
        return {"stage": "survival", "checkpoint": str(path), "epochs": len(history), "seed": seed}
