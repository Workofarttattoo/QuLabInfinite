"""Fine-tune a binary damage-classification head on top of frozen CLIP embeddings.

Architecture:
    CLIP ViT-L/14 image encoder (frozen weights)
    → 768-dim embedding
    → MLP head: Linear(768, 256) → GELU → Dropout(0.3) → Linear(256, 64) → GELU → Linear(64, 1)
    → Sigmoid → damage probability

Why frozen encoder:
    CLIP already learned rich visual representations. Fine-tuning it on ~10k chips
    risks destroying that (catastrophic forgetting) and requires a GPU. The head
    alone trains on a CPU in 10–20 minutes with the chip counts xView2 provides.

Why CLIP specifically (vs. a ResNet trained from scratch):
    CLIP embeddings capture semantic visual concepts ("damaged roof", "debris", "exposed
    wood") that transfer across resolutions, which matters because xView2 is 0.3 m/px
    and our Sentinel-2 chips are 10 m/px.

Usage:
    # Step 1: extract chips (do this first)
    python -m roof_hunter.xview2_chip_extractor \\
        --root ~/Downloads/xview2 \\
        --out  roof_hunter/chips/xview2 \\
        --mode scene

    # Step 2: train (outputs to roof_hunter/models/clip_damage_head.pt)
    python -m roof_hunter.train_clip_damage_head \\
        --chips roof_hunter/chips/xview2 \\
        --epochs 20 \\
        --batch-size 64

    # Step 3: evaluate on held-out split
    python -m roof_hunter.train_clip_damage_head \\
        --chips roof_hunter/chips/xview2 \\
        --eval-only \\
        --model roof_hunter/models/clip_damage_head.pt

Install:
    pip install transformers torch torchvision Pillow scikit-learn
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_MODEL_OUT_DEFAULT = Path("roof_hunter/models/clip_damage_head.pt")
_METRICS_OUT_DEFAULT = Path("roof_hunter/output/clip_damage_training_metrics.json")

# ── dataset ───────────────────────────────────────────────────────────────────

def _gather_chip_paths(chip_dir: Path) -> Tuple[List[Path], List[int]]:
    """Return (paths, labels) from chip_dir/0/*.jpg and chip_dir/1/*.jpg."""
    paths: List[Path] = []
    labels: List[int] = []
    for cls in (0, 1):
        d = chip_dir / str(cls)
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jpg")):
            paths.append(p)
            labels.append(cls)
    if not paths:
        raise FileNotFoundError(
            f"No chips found in {chip_dir}/0/ or {chip_dir}/1/. "
            "Run xview2_chip_extractor.py first."
        )
    return paths, labels


def _train_val_split(
    paths: List[Path],
    labels: List[int],
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    """Stratified train/val split."""
    rng = random.Random(seed)
    idx_0 = [i for i, l in enumerate(labels) if l == 0]
    idx_1 = [i for i, l in enumerate(labels) if l == 1]
    rng.shuffle(idx_0)
    rng.shuffle(idx_1)

    def split(idx: List[int]) -> Tuple[List[int], List[int]]:
        n_val = max(1, int(len(idx) * val_frac))
        return idx[n_val:], idx[:n_val]

    tr0, va0 = split(idx_0)
    tr1, va1 = split(idx_1)
    train_idx = tr0 + tr1
    val_idx   = va0 + va1
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (
        [paths[i] for i in train_idx], [labels[i] for i in train_idx],
        [paths[i] for i in val_idx],   [labels[i] for i in val_idx],
    )


# ── CLIP feature extraction ───────────────────────────────────────────────────

def _load_clip_processor_and_model():
    from transformers import CLIPModel, CLIPProcessor
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    return proc, model


def _extract_features(
    paths: List[Path],
    processor: Any,
    clip_model: Any,
    batch_size: int = 32,
) -> "np.ndarray":
    """Return (N, 768) float32 feature matrix from chip images."""
    import torch
    import numpy as np
    from PIL import Image

    all_feats: List[Any] = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))

        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            # L2-normalise (CLIP convention)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"    Embedded {min(i + batch_size, len(paths))}/{len(paths)}…", flush=True)

    return np.vstack(all_feats).astype("float32")


# ── MLP head ──────────────────────────────────────────────────────────────────

def _build_head(in_dim: int = 768) -> "torch.nn.Module":
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.GELU(),
        nn.Dropout(0.30),
        nn.Linear(256, 64),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(64, 1),
    )


# ── training loop ─────────────────────────────────────────────────────────────

def _pos_weight(labels: List[int]) -> float:
    """Class balancing weight for BCEWithLogitsLoss: n_neg / n_pos."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def train(
    chip_dir: Path,
    out_model: Path,
    out_metrics: Path,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    print(f"Gathering chips from {chip_dir}…", flush=True)
    paths, labels = _gather_chip_paths(chip_dir)
    print(f"  Total chips: {len(paths)}  (damaged={sum(labels)}, intact={len(labels)-sum(labels)})", flush=True)

    tr_paths, tr_labels, va_paths, va_labels = _train_val_split(paths, labels, val_frac, seed)
    print(f"  Train: {len(tr_paths)}  Val: {len(va_paths)}", flush=True)

    print("Loading CLIP encoder (openai/clip-vit-large-patch14)…", flush=True)
    processor, clip_model = _load_clip_processor_and_model()

    print("Extracting CLIP features for train set…", flush=True)
    X_train = _extract_features(tr_paths, processor, clip_model, batch_size)
    print("Extracting CLIP features for val set…", flush=True)
    X_val   = _extract_features(va_paths, processor, clip_model, batch_size)

    # Release CLIP from memory (we only need the head now)
    del clip_model
    try:
        import gc; gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    y_train = np.array(tr_labels, dtype="float32")
    y_val   = np.array(va_labels, dtype="float32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MLP head on {device}…", flush=True)

    X_tr_t = torch.from_numpy(X_train).to(device)
    y_tr_t = torch.from_numpy(y_train).unsqueeze(1).to(device)
    X_va_t = torch.from_numpy(X_val).to(device)
    y_va_t = torch.from_numpy(y_val).unsqueeze(1).to(device)

    in_dim = X_train.shape[1]
    head = _build_head(in_dim).to(device)

    pos_w = torch.tensor([_pos_weight(tr_labels)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -1.0
    best_state: Optional[Dict] = None
    history: List[Dict[str, float]] = []

    n = len(X_train)
    n_batches = max(1, math.ceil(n / batch_size))

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            logits = head(X_tr_t[idx])
            loss   = criterion(logits, y_tr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Validation
        head.eval()
        with torch.no_grad():
            val_logits = head(X_va_t)
            val_loss   = criterion(val_logits, y_va_t).item()
            val_probs  = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_preds  = (val_probs >= 0.5).astype(int)

        try:
            val_auc = float(roc_auc_score(y_val, val_probs))
        except ValueError:
            val_auc = float("nan")
        val_f1  = float(f1_score(y_val, val_preds, zero_division=0))
        val_acc = float(accuracy_score(y_val, val_preds))

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss / n_batches, 5),
            "val_loss": round(val_loss, 5),
            "val_auc": round(val_auc, 4),
            "val_f1":  round(val_f1, 4),
            "val_acc": round(val_acc, 4),
        })
        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train_loss={epoch_loss/n_batches:.4f}  "
            f"val_auc={val_auc:.4f}  val_f1={val_f1:.4f}",
            flush=True,
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    # Save best model
    if best_state is not None:
        head.load_state_dict(best_state)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "in_dim": in_dim,
            "clip_model_id": "openai/clip-vit-large-patch14",
            "best_val_auc": best_val_auc,
            "training_chips": len(tr_paths),
            "val_chips": len(va_paths),
        },
        out_model,
    )
    print(f"\nSaved model → {out_model}  (best val_auc={best_val_auc:.4f})", flush=True)

    metrics = {
        "best_val_auc": round(best_val_auc, 4),
        "train_chips": len(tr_paths),
        "val_chips": len(va_paths),
        "class_balance": {
            "train_damaged": int(sum(tr_labels)),
            "train_intact": int(len(tr_labels) - sum(tr_labels)),
        },
        "history": history,
    }
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ── eval only ─────────────────────────────────────────────────────────────────

def evaluate(chip_dir: Path, model_path: Path) -> None:
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    )

    print(f"Loading model from {model_path}…", flush=True)
    ckpt = torch.load(model_path, map_location="cpu")
    in_dim = ckpt.get("in_dim", 768)
    head = _build_head(in_dim)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    print("Loading CLIP encoder…", flush=True)
    processor, clip_model = _load_clip_processor_and_model()

    paths, labels = _gather_chip_paths(chip_dir)
    print(f"Evaluating on {len(paths)} chips…", flush=True)

    X = _extract_features(paths, processor, clip_model)
    del clip_model

    X_t = torch.from_numpy(X)
    with torch.no_grad():
        probs = torch.sigmoid(head(X_t)).numpy().flatten()
    preds = (probs >= 0.5).astype(int)
    y = np.array(labels)

    auc = roc_auc_score(y, probs)
    print(f"\nROC-AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy_score(y, preds):.4f}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y, preds))
    print("\nClassification report:")
    print(classification_report(y, preds, target_names=["intact", "damaged"]))


# ── import guard ──────────────────────────────────────────────────────────────

try:
    import math
except ImportError:
    pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import math as _math
    global math
    math = _math

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chips", type=Path,
                    default=Path("roof_hunter/chips/xview2"),
                    help="Chip directory (output of xview2_chip_extractor.py)")
    ap.add_argument("--model", type=Path, default=_MODEL_OUT_DEFAULT)
    ap.add_argument("--metrics", type=Path, default=_METRICS_OUT_DEFAULT)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training; evaluate an existing model on the chip directory")
    args = ap.parse_args()

    if args.eval_only:
        evaluate(args.chips.expanduser().resolve(), args.model.expanduser().resolve())
    else:
        train(
            chip_dir=args.chips.expanduser().resolve(),
            out_model=args.model.expanduser().resolve(),
            out_metrics=args.metrics.expanduser().resolve(),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
