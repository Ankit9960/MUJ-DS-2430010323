"""Training and inference for one experiment configuration."""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src import config as C
from src.evaluate import evaluate
from src.losses import build_loss
from src.model import build_model, count_parameters
from src.utils import set_seed, get_device, Timer


@dataclass
class ExperimentConfig:
    """One point in the ablation grid."""
    name: str = "run"
    protocol: str = "inter"        # inter | intra          (ablation axis 1)
    loss: str = "focal"            # ce | weighted_ce | focal (axis 2)
    arch: str = "multiscale"       # baseline | multiscale   (axis 3)
    leads: str = "dual"            # single | dual           (axis 4)
    use_rr: bool = True
    seed: int = 0
    epochs: int = C.DEFAULTS["epochs"]
    batch_size: int = C.DEFAULTS["batch_size"]
    lr: float = C.DEFAULTS["lr"]
    weight_decay: float = C.DEFAULTS["weight_decay"]
    patience: int = C.DEFAULTS["patience"]
    dropout: float = C.DEFAULTS["dropout"]
    focal_gamma: float = C.DEFAULTS["focal_gamma"]
    cb_beta: float = C.DEFAULTS["cb_beta"]
    synthetic: bool = False
    notes: str = ""
    tags: list = field(default_factory=list)

    @property
    def in_channels(self) -> int:
        return 1 if self.leads == "single" else 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["in_channels"] = self.in_channels
        return d


# --------------------------------------------------------------------------
# Tensor preparation
# --------------------------------------------------------------------------
def _select_leads(x: np.ndarray, leads: str) -> np.ndarray:
    """(n, L, 2) -> (n, C, L) with C = 1 or 2."""
    x = x[:, :, :1] if leads == "single" else x[:, :, :2]
    return np.ascontiguousarray(x.transpose(0, 2, 1))


def make_loaders(train: dict, val: dict, test: dict, cfg: ExperimentConfig):
    """Build DataLoaders. RR features are standardised with TRAIN statistics
    only -- computing them over the full set would leak test information."""
    rr_mu = train["rr"].mean(axis=0, keepdims=True)
    rr_sd = np.where(train["rr"].std(axis=0, keepdims=True) < 1e-8, 1.0,
                     train["rr"].std(axis=0, keepdims=True))

    def to_tensors(d: dict):
        return (
            torch.from_numpy(_select_leads(d["beats"], cfg.leads)).float(),
            torch.from_numpy(_select_leads(d["context"], cfg.leads)).float(),
            torch.from_numpy(((d["rr"] - rr_mu) / rr_sd).astype(np.float32)),
            torch.from_numpy(d["labels"]).long(),
        )

    ds_train, ds_val, ds_test = (TensorDataset(*to_tensors(d)) for d in (train, val, test))
    return (
        DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=False),
        DataLoader(ds_val, batch_size=1024, shuffle=False),
        DataLoader(ds_test, batch_size=1024, shuffle=False),
    )


# --------------------------------------------------------------------------
# Train / predict
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for beat, ctx, rr, y in loader:
        logits = model(beat.to(device), ctx.to(device), rr.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def train_one(train: dict, val: dict, test: dict,
              cfg: ExperimentConfig, verbose: bool = True,
              save_to=None) -> dict:
    """Train a single configuration and return its metrics + history.

    `save_to` writes the selected (best-validation) checkpoint to that path as a
    self-describing dict -- weights plus everything needed to rebuild the model
    -- so the file can be loaded without consulting this script.
    """
    set_seed(cfg.seed)
    device = get_device()

    loader_tr, loader_va, loader_te = make_loaders(train, val, test, cfg)
    model = build_model(cfg.arch, cfg.in_channels, len(C.AAMI_CLASSES),
                        cfg.dropout, use_rr=cfg.use_rr).to(device)
    criterion = build_loss(cfg.loss, train["labels"], len(C.AAMI_CLASSES),
                           cfg.focal_gamma, cfg.cb_beta).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=3)

    best_score, best_state, bad_epochs = -np.inf, None, 0
    history = []

    with Timer() as timer:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running, n_seen = 0.0, 0
            for beat, ctx, rr, y in loader_tr:
                beat, ctx, rr, y = (t.to(device) for t in (beat, ctx, rr, y))
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(model(beat, ctx, rr), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()
                running += loss.item() * len(y)
                n_seen += len(y)

            y_true, y_pred = predict(model, loader_va, device)
            val_metrics = evaluate(y_true, y_pred)
            score = val_metrics["macro_f1"]      # selection criterion
            scheduler.step(score)

            history.append(dict(epoch=epoch,
                                train_loss=running / max(n_seen, 1),
                                val_macro_f1=score,
                                val_accuracy=val_metrics["accuracy"]))
            if verbose:
                print(f"    epoch {epoch:>3}/{cfg.epochs}  "
                      f"loss={running / max(n_seen, 1):.4f}  "
                      f"val_macroF1={score:.4f}  val_acc={val_metrics['accuracy']:.4f}")

            if score > best_score + 1e-5:
                best_score, bad_epochs = score, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    if verbose:
                        print(f"    early stop at epoch {epoch} "
                              f"(no val gain for {cfg.patience} epochs)")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_to is not None:
        from pathlib import Path as _Path
        save_to = _Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best_state,
            "arch": cfg.arch,
            "in_channels": cfg.in_channels,
            "n_classes": len(C.AAMI_CLASSES),
            "dropout": cfg.dropout,
            "use_rr": cfg.use_rr,
            "classes": C.AAMI_CLASSES,
            "config": cfg.to_dict(),
            "best_val_macro_f1": float(best_score),
        }, save_to)
        if verbose:
            print(f"    saved checkpoint -> {save_to}")

    y_true, y_pred = predict(model, loader_te, device)
    metrics = evaluate(y_true, y_pred)
    metrics.update(
        config=cfg.to_dict(),
        best_val_macro_f1=float(best_score),
        epochs_run=len(history),
        train_seconds=round(timer.elapsed, 1),
        n_parameters=count_parameters(model),
        history=history,
        n_train_beats=int(len(train["labels"])),
        n_val_beats=int(len(val["labels"])),
    )
    return metrics


# --------------------------------------------------------------------------
# Command-line entry point
# --------------------------------------------------------------------------
def main():
    """Train one configuration and report its test metrics.

        python src/train.py                         # the proposed model
        python src/train.py --arch baseline --loss ce --leads single --no-rr
        python src/train.py --protocol intra        # the leaky split
        python src/train.py --save models/my_model.pt

    For the full nine-configuration grid use scripts/run_ablation.py.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Train an ECG arrhythmia classifier.")
    ap.add_argument("--protocol", default="inter", choices=["inter", "intra"])
    ap.add_argument("--arch", default="multiscale", choices=["baseline", "multiscale"])
    ap.add_argument("--loss", default="focal", choices=["ce", "weighted_ce", "focal"])
    ap.add_argument("--leads", default="dual", choices=["single", "dual"])
    ap.add_argument("--no-rr", action="store_true", help="drop the RR-interval features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=C.DEFAULTS["epochs"])
    ap.add_argument("--save", default=None, help="write the checkpoint to this path")
    args = ap.parse_args()

    from src.splits import get_split
    from src.evaluate import format_confusion_matrix

    cfg = ExperimentConfig(
        name=f"{args.protocol}_{args.arch}_{args.loss}_{args.leads}",
        protocol=args.protocol, arch=args.arch, loss=args.loss,
        leads=args.leads, use_rr=not args.no_rr, seed=args.seed, epochs=args.epochs)

    print("=" * 74)
    print(f"TRAINING  {cfg.name}  (seed {cfg.seed})")
    print("=" * 74)
    for k, v in cfg.to_dict().items():
        if k not in ("tags", "notes"):
            print(f"  {k:<16} {v}")

    print("\nLoading data...")
    train, val, test = get_split(cfg.protocol, cfg.seed)
    print(f"  train {len(train['labels']):,} | val {len(val['labels']):,} "
          f"| test {len(test['labels']):,} beats")

    print("\nTraining...")
    m = train_one(train, val, test, cfg, verbose=True, save_to=args.save)

    print("\n" + "-" * 74)
    print(f"  accuracy    {m['accuracy']:.4f}")
    print(f"  macro F1    {m['macro_f1']:.4f}")
    print(f"  Cohen kappa {m['kappa']:.4f}")
    print(f"  parameters  {m['n_parameters']:,}   "
          f"time {m['train_seconds']}s   epochs {m['epochs_run']}")
    print(f"\n  {'class':<7}{'Se':>9}{'P+':>9}{'F1':>9}{'support':>10}")
    for cls, pc in m["per_class"].items():
        print(f"  {cls:<7}{pc['sensitivity']:>9.4f}{pc['positive_predictivity']:>9.4f}"
              f"{pc['f1']:>9.4f}{pc['support']:>10,}")
    print("\n  Confusion matrix (counts)")
    print("  " + format_confusion_matrix(m["confusion_matrix"]).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
