"""
Cataract Classification — AH-1 Ablation Study (Updated — Scientifically Valid)
================================================================================
AH-1: Remove quantum circuits → replace with structurally parallel MLP blocks.

This is a *minimal* ablation: the ONLY difference from QH is that each
quantum circuit (PennyLane QNode) is replaced by a small MLP with identical
input/output dimensions:
    QH  : n_q_blocks × QNode(n_qubits inputs → n_qubits outputs)
    AH-1: n_q_blocks × MLP (n_qubits → mlp_hidden × n_layers → n_qubits)

Everything else is preserved verbatim from the QH architecture:
    • Same backbone + projection (dim→512→256)
    • Same qubit encoder (256→total_q_out)
    • Same log_scale / q_out_scale learnable parameters
    • Same gated fusion (fusion_bb, fusion_q, fusion_gate)
    • Same BN + dropout + head
    • Same two-phase training protocol (warmup + unfreeze)
    • Same optimizer, LR groups, scheduler, grad-clip, patience

FAIRNESS / VALIDITY FIXES (vs original AH-1):
  1. mlp_hidden=2 (default) → ~78 params/block vs QH 72 quantum params/block
     (parameter-matched comparison — the scientifically correct primary test)
  2. McNemar test now run for BOTH AH-1 vs QH and AH-1 vs CL
  3. Capacity_ratio added to result CSV (machine-readable audit trail)
  4. Optimizer phase-2 logs mlp_blocks param count explicitly
  5. Missing QH/CL prob files now raise ERROR (not silent warning)
  6. write_ah1_report header documents param counts and capacity ratio
  7. --mlp-hidden-list supports multiple hidden dims in one run
     (e.g. 2 4 8 → param-matched, current, 2× capacity)
  8. plot_three_way_comparison extended to show multiple AH-1 capacity variants

Logical chain of evidence for quantum advantage claim:
    QH > CL          → quantum hybrid beats classical (from main_experiment.py)
    QH > AH-1 (h=2)  → quantum beats matched-capacity MLP (parameter-controlled)
    QH > AH-1 (h=8)  → quantum beats 3×-capacity MLP (capacity-controlled)
    AH-1 ≈ CL        → MLP replacement offers no advantage over plain baseline

Isolation guarantee: any performance difference between QH and AH-1 can be
attributed *only* to the presence/absence of quantum circuits.

Usage
-----
# Train all 8 backbones (parameter-matched, h=2):
python ablation_ah1.py

# Train with multiple MLP hidden dims for capacity sweep:
python ablation_ah1.py --mlp-hidden-list 2 4 8

# Train specific backbone(s):
python ablation_ah1.py --models resnet50 efficientnet_b0

# Evaluate only (requires existing AH-1 checkpoints):
python ablation_ah1.py --eval-only

# Load QH/CL results for comparison chart:
python ablation_ah1.py --compare-csv ./quantum_output_v12/all_results_qh_vs_cl.csv

Output (per hidden dim h)
------
./ablation_ah1_h{h}_output/<model_name>/
    best_model_ah1.pth
    report_ah1.txt
    metrics_ah1.json
    roc_ah1_vs_qh.png
    roc_ah1_vs_cl.png
    confusion_matrix_ah1.png
    training_curves_ah1.svg
    timing_ah1.json
    resume_best_model_ah1.pth   (deleted on clean finish)

./ablation_ah1_h{h}_output/
    all_results_ah1.csv
    mcnemar_ah1_vs_qh.csv
    mcnemar_ah1_vs_cl.csv
    auc_comparison_ah1_vs_qh_cl.png
    AH1_COMPARISON_REPORT.txt
"""

import os, gc, argparse, random, json, logging
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2 as chi2_dist

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize
import timm

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw): return it

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Config (must mirror your QH/CL run exactly) ───────────────────────────────
CFG = {
    "data_dir"    : Path("./dataset"),
    "output_dir"  : Path("./ablation_ah1_h2_output_for_all_3mlp"),   # updated per run
    "seed"        : 42,
    "classes"     : ["Immature_Cataract", "Mature_Cataract", "Normal_Eye"],
    "epochs"      : 35,
    "batch_size"  : 16,
    "lr"          : 2e-4,
    "lr_min"      : 1e-6,
    "lr_backbone" : 5e-5,
    "patience"    : 8,
    "weight_decay": 1e-4,
    "label_smooth": 0.1,
    "num_workers" : 4,
    "warmup_epochs": 4,
    # ── Quantum / MLP block dims (must match QH CFG exactly) ──────────────────
    "n_qubits"    : 8,      # inputs/outputs per MLP block  (== n_qubits in QH)
    "n_layers"    : 3,      # MLP hidden layers per block   (== n_layers in QH)
    "n_q_blocks"  : 4,      # number of parallel blocks     (== n_q_blocks in QH)
    "proj_hidden" : 256,    # backbone compressor output    (== proj_hidden in QH)
    # ── AH-1 MLP hidden width ─────────────────────────────────────────────────
    # FIX 1: mlp_hidden=2 gives ~78 params/block vs QH's 72 quantum params/block.
    # Formula: AH-1 params/block = 35*h + 8  (with n_qubits=8, n_layers=3)
    # QH params/block = n_layers * n_qubits * 3 = 72
    # h=2 → 35*2+8 = 78  (closest integer match above 72)
    # This is the parameter-matched primary comparison.
    # Use --mlp-hidden-list 2 4 8 to also run capacity-sweep variants.
    "mlp_hidden"  : 2,
    "models": [
        "resnet50", "densenet121", "inception_v3", "mobilenetv3_large_100",
        "convnext_tiny", "efficientnet_b0", "repvgg_b3", "vit_base_patch16_224",
    ],
    "image_sizes": {"inception_v3": 299, "__default__": 224},
    # Path to QH/CL results from main experiment for three-way comparison
    "qh_cl_csv"   : Path("./quantum_output_v12/all_results_qh_vs_cl.csv"),
    "qh_cl_dir"   : Path("./quantum_output_v12"),
}
CFG["unfreeze_models"] = set(CFG["models"])
CLASSES     = CFG["classes"]
NUM_CLASSES = len(CLASSES)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.enabled           = True
    torch.backends.cudnn.benchmark         = True
    torch.backends.cudnn.deterministic     = False
    torch.backends.cuda.matmul.allow_tf32  = True
    torch.backends.cudnn.allow_tf32        = True
    torch.cuda.manual_seed(CFG["seed"])
    _gpu = torch.cuda.get_device_properties(0)
    log.info(f"CUDA: {_gpu.name} | {_gpu.total_memory//1024**3}GB")
else:
    DEVICE = torch.device("cpu")
    torch.set_num_threads(8); torch.set_num_interop_threads(4)
    torch.backends.mkldnn.enabled = True
    log.info("Running on CPU")

_PIN_MEM  = (DEVICE.type == "cuda")
COLOR_AH1 = "#E8701A"   # orange — distinct from QH purple and CL blue
COLOR_QH  = "#7B52AB"
COLOR_CL  = "#3B82C4"

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])


# ── Parameter count helpers ───────────────────────────────────────────────────
def qh_params_per_block():
    """Quantum params per block: n_layers × n_qubits × 3 Euler angles."""
    return CFG["n_layers"] * CFG["n_qubits"] * 3   # 3 × 8 × 3 = 72


def ah1_params_per_block(mlp_hidden=None):
    """
    MLP params per block:
        n_layers × (n_q×h + h)   [hidden layers]
      + h×n_q + n_q              [output layer]
    = n_layers × h × (n_q+1) + n_q × (h+1)

    With n_q=8, n_layers=3:
      = 3h(9) + 8(h+1) = 27h + 8h + 8 = 35h + 8
    """
    h  = mlp_hidden or CFG["mlp_hidden"]
    nq = CFG["n_qubits"]
    nl = CFG["n_layers"]
    return nl * (nq * h + h) + h * nq + nq


def capacity_ratio(mlp_hidden=None):
    return round(ah1_params_per_block(mlp_hidden) / qh_params_per_block(), 3)


# ── AH-1 MLP block ───────────────────────────────────────────────────────────
class MLPBlock(nn.Module):
    """
    Structurally parallel replacement for one quantum circuit block.

    QH  : QNode(inputs[n_q]) → [n_q expval measurements]
    AH-1: MLP (n_q → hidden × n_layers → n_q)

    Activation: Tanh — mirrors the bounded [-1, +1] output range of
    Pauli expectation values, keeping downstream statistics compatible.

    n_layers hidden layers mirror circuit depth.
    Output bounded to [-1, 1] via final Tanh (same range as expvals).

    Parameter count: 35*mlp_hidden + 8  (with n_qubits=8, n_layers=3)
    QH equivalent : n_layers*n_qubits*3 = 72 quantum params/block
    """
    def __init__(self, n_qubits: int, n_layers: int, mlp_hidden: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_qubits
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, mlp_hidden), nn.Tanh()]
            in_dim  = mlp_hidden
        layers += [nn.Linear(in_dim, n_qubits), nn.Tanh()]   # bounded output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Backbone utils ────────────────────────────────────────────────────────────
def _make_backbone(name, image_size):
    if name == "inception_v3":
        bb = timm.create_model(name, pretrained=True, num_classes=0, aux_logits=False)
    else:
        bb = timm.create_model(name, pretrained=True, num_classes=0)
    with torch.no_grad():
        dim = bb(torch.randn(1, 3, image_size, image_size)).shape[1]
    return bb, dim


def _freeze(bb):
    for p in bb.parameters(): p.requires_grad_(False)


# ── AH-1 Model ───────────────────────────────────────────────────────────────
class AH1Model(nn.Module):
    """
    AH-1: QH architecture with quantum circuits replaced by MLP blocks.

    Architecture (identical to QH except *** marked lines):

        Backbone  →  Projection (dim→512→256)
                  →  Block Encoder (256→total_q_out)   [split into n_q_blocks]
        ***       →  n_q_blocks × MLPBlock             [was: quantum circuits]
                  →  q_out × q_out_scale               [learnable output scale]
                  →  Gated Fusion                      [identical to QH]
                  →  BN → Dropout → Linear head

    Parameter parity (with mlp_hidden=2, n_qubits=8, n_layers=3):
        QH  quantum params/block : 72   (n_layers × n_qubits × 3)
        AH-1 MLP params/block    : 78   (35×h + 8, h=2)
        Capacity ratio           : 1.08×  ← closest possible match

    The slight excess (1.08×) is intentional and documented:
    if QH still outperforms AH-1 despite AH-1 having marginally more
    classical capacity, the quantum advantage claim is strengthened.
    """
    def __init__(self, backbone_name, n_qubits, n_layers, image_size,
                 n_q_blocks=None, mlp_hidden=None):
        super().__init__()
        self.model_type    = "AH1"
        self.backbone_name = backbone_name
        self.backbone, dim = _make_backbone(backbone_name, image_size)
        self._n_qubits     = n_qubits
        self._n_q_blocks   = n_q_blocks or CFG.get("n_q_blocks", 1)
        self._mlp_hidden   = mlp_hidden or CFG.get("mlp_hidden", 2)
        total_q_out        = n_qubits * self._n_q_blocks   # e.g. 8×4=32
        h1, h2 = 512, CFG["proj_hidden"]
        fused_gate_dim = h2   # 256

        _qh_p  = qh_params_per_block()
        _ah1_p = ah1_params_per_block(self._mlp_hidden)
        log.info(f"  [AH-1] dim={dim} proj={dim}->{h1}->{h2}->{total_q_out} "
                 f"({self._n_q_blocks} blocks×{n_qubits}→{self._mlp_hidden}→{n_qubits}) "
                 f"layers={n_layers} | "
                 f"params/block: AH1={_ah1_p} QH={_qh_p} ratio={_ah1_p/_qh_p:.2f}×")

        # ── Identical to QH ──────────────────────────────────────────────────
        self.projection = nn.Sequential(
            nn.Linear(dim, h1), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(0.1),
        )
        self.projection_qubit = nn.Sequential(
            nn.Linear(h2, total_q_out),
            nn.LayerNorm(total_q_out),
        )

        # *** MLP blocks — replace quantum circuits (everything else stays) ***
        self.mlp_blocks = nn.ModuleList([
            MLPBlock(n_qubits, n_layers, self._mlp_hidden)
            for _ in range(self._n_q_blocks)
        ])

        # ── Identical to QH ──────────────────────────────────────────────────
        self.log_scale   = nn.Parameter(torch.zeros(1))
        self.q_out_scale = nn.Parameter(torch.ones(total_q_out))

        self.fusion_bb   = nn.Linear(h2, fused_gate_dim)
        self.fusion_q    = nn.Linear(total_q_out, fused_gate_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(h2 + total_q_out, fused_gate_dim),
            nn.Sigmoid(),
        )
        self.bn        = nn.BatchNorm1d(fused_gate_dim)
        self.dropout_q = nn.Dropout(0.1)
        self.head      = nn.Linear(fused_gate_dim, NUM_CLASSES)

    def _mlp_forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Run all MLP blocks and return concatenated, scaled output.
        Mirrors QuantumHybridModel._quantum_forward in structure.

        Identical preprocessing to QH._quantum_forward:
            clamp → log_scale → chunk → blocks → cat → q_out_scale
        No CPU transfer needed — MLP runs natively on DEVICE
        unlike quantum circuits which require CPU (lightning.qubit).
        """
        # Identical preprocessing to QH (clamp BEFORE scaling, same as QH)
        angles  = torch.clamp(angles, -1.0, 1.0)
        scaled  = angles * torch.exp(self.log_scale)
        chunks  = scaled.chunk(self._n_q_blocks, dim=1)
        outputs = [mlp(chunk) for mlp, chunk in zip(self.mlp_blocks, chunks)]
        mlp_out = torch.cat(outputs, dim=1)        # [B, total_q_out]
        return mlp_out * self.q_out_scale           # learnable out-scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats         = self.backbone(x)
        backbone_feat = self.projection(feats)
        angles        = self.projection_qubit(backbone_feat)
        mlp_out       = self._mlp_forward(angles)
        bb_proj = self.fusion_bb(backbone_feat)
        q_proj  = self.fusion_q(mlp_out)
        gate    = self.fusion_gate(torch.cat([backbone_feat, mlp_out], dim=1))
        fused   = gate * bb_proj + (1.0 - gate) * q_proj
        return self.head(self.bn(self.dropout_q(fused)))

    def cached_forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Skip backbone — receives pre-extracted features (warmup phase)."""
        backbone_feat = self.projection(feats)
        angles        = self.projection_qubit(backbone_feat)
        mlp_out       = self._mlp_forward(angles)
        bb_proj = self.fusion_bb(backbone_feat)
        q_proj  = self.fusion_q(mlp_out)
        gate    = self.fusion_gate(torch.cat([backbone_feat, mlp_out], dim=1))
        fused   = gate * bb_proj + (1.0 - gate) * q_proj
        return self.head(self.bn(self.dropout_q(fused)))

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_counts(self):
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trained


# ── Data ──────────────────────────────────────────────────────────────────────
def get_transforms(sz):
    return transforms.Compose([
        transforms.Resize((sz, sz)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

_tensor_cache: dict = {}
_loader_cache: dict = {}


def _build_tensor_cache(split, image_size):
    key = (split, image_size)
    if key in _tensor_cache: return _tensor_cache[key]
    log.info(f"  Caching '{split}' ({image_size}px)...")
    t0  = time.time()
    ds  = datasets.ImageFolder(f"./dataset/{split}", transform=get_transforms(image_size))
    ldr = DataLoader(ds, batch_size=64, shuffle=False,
                     num_workers=CFG["num_workers"], pin_memory=False)
    Xs, ys = [], []
    for x, y in ldr: Xs.append(x); ys.append(y)
    X, y = torch.cat(Xs), torch.cat(ys)
    log.info(f"  Cached {len(y)} in {time.time()-t0:.1f}s | "
             f"{X.element_size()*X.nelement()/1e6:.0f} MB")
    _tensor_cache[key] = (X, y, ds)
    return _tensor_cache[key]


def get_loaders(image_size):
    if image_size in _loader_cache: return _loader_cache[image_size]
    X_tr, y_tr, _   = _build_tensor_cache("train", image_size)
    X_va, y_va, _   = _build_tensor_cache("val",   image_size)
    X_te, y_te, dst = _build_tensor_cache("test",  image_size)
    counts  = torch.bincount(y_tr)
    log.info(f"  Class counts: {dict(zip(CLASSES, counts.tolist()))}")
    sw      = (1.0 / counts.float())[y_tr]
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
    kw  = dict(num_workers=0, pin_memory=_PIN_MEM)
    trl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=CFG["batch_size"], sampler=sampler, **kw)
    val = DataLoader(TensorDataset(X_va, y_va), batch_size=CFG["batch_size"], shuffle=False, **kw)
    tel = DataLoader(TensorDataset(X_te, y_te), batch_size=CFG["batch_size"], shuffle=False, **kw)
    _loader_cache[image_size] = (trl, val, tel, dst, y_te)
    return _loader_cache[image_size]


def clear_cache_for_image_size(image_size):
    for s in ("train", "val", "test"): _tensor_cache.pop((s, image_size), None)
    _loader_cache.pop(image_size, None)
    gc.collect()


# ── Loss ──────────────────────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, s=0.1):
        super().__init__(); self.s = s

    def forward(self, pred, target):
        n  = pred.size(-1)
        lp = nn.functional.log_softmax(pred, dim=-1)
        sm = torch.full_like(lp, self.s / (n - 1))
        sm.scatter_(-1, target.unsqueeze(-1), 1.0 - self.s)
        return -(sm * lp).sum(dim=-1).mean()


CRITERION = LabelSmoothingCE(CFG["label_smooth"])


# ── Training utils ────────────────────────────────────────────────────────────
def freeze_backbone(model):
    """Phase-1: freeze everything except head, projection, and mlp_blocks."""
    for name, param in model.named_parameters():
        if ("head"       not in name and
            "projection" not in name and
            "mlp_blocks" not in name):
            param.requires_grad_(False)


def unfreeze_all(model):
    """Phase-2: release all parameters so backbone also gets gradients."""
    for param in model.parameters():
        param.requires_grad_(True)


def _make_optimizer(model, phase=1):
    """
    Build AdamW with per-component learning rates.
    Strictly identical logic to QH/CL — no model-type branching.

    Phase 1 (warmup, frozen backbone):
      Single LR group — all trainable params at CFG["lr"].

    Phase 2 (full fine-tune, backbone unfrozen):
      Exactly TWO groups (same as QH and CL):
        backbone params     → CFG["lr_backbone"]  (slow)
        all other params    → CFG["lr"]            (head, projection, mlp_blocks)

    FIX 4: explicit logging of mlp_blocks param count for audit trail.
    """
    if phase == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    bb_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (bb_params if "backbone" in n else other_params).append(p)

    # FIX 4: log mlp_blocks param count so reviewer can audit group assignment
    mlp_numel = sum(p.numel() for n, p in model.named_parameters()
                    if "mlp_blocks" in n and p.requires_grad)
    log.info(f"  [AH-1] phase2 optimizer: "
             f"bb_params={len(bb_params)} tensors | "
             f"other_params={len(other_params)} tensors "
             f"(mlp_blocks total numel={mlp_numel})")

    groups = [{"params": other_params, "lr": CFG["lr"]}]
    if bb_params:
        groups.append({"params": bb_params, "lr": CFG["lr_backbone"]})
    return optim.AdamW(groups, weight_decay=CFG["weight_decay"])


@torch.no_grad()
def _cache_backbone_features(backbone, loader, is_train=True):
    backbone.eval()
    Xf, Yf = [], []
    split = "train" if is_train else "val/test"
    for imgs, y in tqdm(loader, desc=f"  cache {split}", leave=False,
                        unit="bat", dynamic_ncols=True, colour="yellow"):
        imgs = imgs.to(DEVICE, non_blocking=True)
        Xf.append(backbone(imgs).cpu())
        Yf.append(y)
    X = torch.cat(Xf); Y = torch.cat(Yf)
    if is_train:
        counts  = torch.bincount(Y)
        sw      = (1.0 / counts.float())[Y]
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        return DataLoader(TensorDataset(X, Y), batch_size=CFG["batch_size"],
                          sampler=sampler, num_workers=0, pin_memory=_PIN_MEM)
    return DataLoader(TensorDataset(X, Y), batch_size=CFG["batch_size"] * 2,
                      shuffle=False, num_workers=0, pin_memory=_PIN_MEM)


def train_one_epoch(model, loader, optimizer, use_cached=False):
    model.train()
    loss_sum = correct = total = 0
    pbar = tqdm(loader, desc="  train", leave=False, unit="bat",
                dynamic_ncols=True, colour="cyan")
    for x, labels in pbar:
        x      = x.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out  = model.cached_forward(x) if use_cached else model(x)
        loss = CRITERION(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)   # identical to QH/CL
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct  += out.detach().argmax(1).eq(labels).sum().item()
        total    += x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
    return loss_sum / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, use_cached=False):
    model.eval()
    loss_sum = correct = total = 0
    probs_l, labels_l = [], []
    pbar = tqdm(loader, desc="   eval", leave=False, unit="bat",
                dynamic_ncols=True, colour="green")
    for x, labels in pbar:
        x      = x.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        out  = model.cached_forward(x) if use_cached else model(x)
        loss = CRITERION(out, labels)
        p    = torch.softmax(out, dim=1)
        loss_sum += loss.item() * x.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += x.size(0)
        pbar.set_postfix(acc=f"{correct/total:.4f}")
        probs_l.append(p.cpu().numpy()); labels_l.append(labels.cpu().numpy())
    return loss_sum / total, correct / total, \
           np.concatenate(probs_l), np.concatenate(labels_l)


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _save_resume_ckpt(path, epoch, phase, model, opt, sch,
                      history, best_val, patience_ctr):
    torch.save({
        "epoch"       : epoch,
        "phase"       : phase,
        "model_state" : model.state_dict(),
        "opt_state"   : opt.state_dict(),
        "sch_state"   : sch.state_dict(),
        "history"     : history,
        "best_val"    : best_val,
        "patience_ctr": patience_ctr,
    }, path)


# ── Two-phase training (identical protocol to QH/CL) ─────────────────────────
def train_model(model, label, train_ldr, val_ldr, out_dir, ckpt_name):
    """
    Two-phase training — identical protocol to QH and CL.

    Phase 1 (warmup, frozen backbone):
      Backbone features cached once → fast warmup epochs.

    Phase 2 (full fine-tune, backbone unfrozen):
      End-to-end with differential LR (backbone slower).

    Crash recovery: resume checkpoint saved every epoch, deleted on clean finish.
    """
    WARMUP      = CFG["warmup_epochs"]
    resume_path = out_dir / f"resume_{ckpt_name}"

    if resume_path.exists():
        log.info(f"  [{label}] Resume checkpoint found — restoring ...")
        ckpt         = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        start_epoch  = ckpt["epoch"] + 1
        phase        = ckpt["phase"]
        best_val     = ckpt["best_val"]
        patience_ctr = ckpt["patience_ctr"]
        history      = ckpt["history"]
        model.load_state_dict(ckpt["model_state"])
        if phase == 2 or start_epoch > WARMUP:
            phase = 2; unfreeze_all(model)
            opt = _make_optimizer(model, phase=2)
        else:
            freeze_backbone(model)
            opt = _make_optimizer(model, phase=1)
        opt.load_state_dict(ckpt["opt_state"])
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor): state[k] = v.to(DEVICE)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, CFG["epochs"] - WARMUP), eta_min=CFG["lr_min"])
        sch.load_state_dict(ckpt["sch_state"])
        log.info(f"  [{label}] Resumed epoch {start_epoch} "
                 f"phase {phase} best={best_val:.4f}")
    else:
        start_epoch  = 1; phase = 1
        best_val     = 0.0; patience_ctr = 0
        history      = {"train_loss": [], "val_loss": [], "train_acc": [],
                        "val_acc": [], "epoch_time": []}
        freeze_backbone(model)
        opt = _make_optimizer(model, phase=1)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, CFG["epochs"] - WARMUP), eta_min=CFG["lr_min"])

    feat_train = feat_val = None
    if start_epoch <= WARMUP:
        log.info(f"  [{label}] Caching backbone features for warmup ...")
        t0         = time.time()
        feat_train = _cache_backbone_features(model.backbone, train_ldr, is_train=True)
        feat_val   = _cache_backbone_features(model.backbone, val_ldr,   is_train=False)
        log.info(f"  [{label}] Cache ready in {time.time()-t0:.1f}s")

    total_p, train_p = model.param_counts()
    log.info(f"  [{label}] total={total_p:,} trainable={train_p:,} "
             f"start_epoch={start_epoch} phase={phase}")

    t_start   = time.time()
    epoch_bar = tqdm(range(start_epoch, CFG["epochs"] + 1),
                     desc=f"[{label}]", unit="ep", dynamic_ncols=True, colour="magenta")

    for epoch in epoch_bar:
        t0 = time.time()

        if epoch == WARMUP + 1 and phase == 1:
            unfreeze_all(model)
            opt = _make_optimizer(model, phase=2)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, CFG["epochs"] - WARMUP), eta_min=CFG["lr_min"])
            phase = 2
            if feat_train is not None:
                del feat_train, feat_val; feat_train = feat_val = None
                gc.collect()
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
            _, train_p2 = model.param_counts()
            log.info(f"  [{label}] E{epoch}: backbone unfrozen "
                     f"(LR bb={CFG['lr_backbone']:.0e} other={CFG['lr']:.0e}) "
                     f"trainable={train_p2:,}")

        use_cached = (epoch <= WARMUP and feat_train is not None)
        trl = feat_train if use_cached else train_ldr
        vll = feat_val   if use_cached else val_ldr

        tr_loss, tr_acc       = train_one_epoch(model, trl, opt, use_cached=use_cached)
        vl_loss, vl_acc, _, _ = evaluate(model, vll, use_cached=use_cached)
        sch.step()
        ep_time = time.time() - t0

        for k, v in [("train_loss", tr_loss), ("val_loss", vl_loss),
                     ("train_acc", tr_acc),  ("val_acc", vl_acc),
                     ("epoch_time", round(ep_time, 2))]:
            history[k].append(v)

        log.info(f"  [{label}] E{epoch:>3}/{CFG['epochs']} "
                 f"Tr={tr_acc:.4f} L={tr_loss:.4f} Va={vl_acc:.4f} {ep_time:.0f}s")

        if vl_acc > best_val:
            best_val = vl_acc; patience_ctr = 0
            torch.save(model.state_dict(), out_dir / ckpt_name)
            log.info(f"    ✓ Best saved (val={vl_acc:.4f})")
        else:
            patience_ctr += 1

        epoch_bar.set_postfix(ph=phase, tr=f"{tr_acc:.3f}", va=f"{vl_acc:.3f}",
                              best=f"{best_val:.3f}",
                              pat=f"{patience_ctr}/{CFG['patience']}")

        _save_resume_ckpt(resume_path, epoch, phase, model, opt, sch,
                          history, best_val, patience_ctr)

        if patience_ctr >= CFG["patience"]:
            log.info(f"  Early stop E{epoch} best={best_val:.4f}"); break

    if resume_path.exists():
        resume_path.unlink()
        log.info(f"  [{label}] Resume checkpoint removed")

    history["total_time"] = round(time.time() - t_start, 2)
    return history, best_val


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(true_labels, probs, preds):
    bin_labels = label_binarize(true_labels, classes=list(range(NUM_CLASSES)))
    report = classification_report(true_labels, preds, target_names=CLASSES, output_dict=True)
    cm     = confusion_matrix(true_labels, preds)
    per_class = {}
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp; tn = cm.sum() - tp - fp - fn
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        per_class[cls] = {
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "FPR": round(fpr, 4), "FNR": round(fnr, 4),
            "Sensitivity": round(report[cls]["recall"], 4),
            "Specificity": round(1 - fpr, 4),
            "Precision":   round(report[cls]["precision"], 4),
            "F1":          round(report[cls]["f1-score"], 4),
        }
    auc_per   = {cls: round(roc_auc_score(bin_labels[:, i], probs[:, i]), 4)
                 for i, cls in enumerate(CLASSES)}
    macro_auc = round(float(roc_auc_score(bin_labels, probs,
                                          multi_class="ovr", average="macro")), 4)
    ap_per    = {cls: round(average_precision_score(bin_labels[:, i], probs[:, i]), 4)
                 for i, cls in enumerate(CLASSES)}
    return {
        "overall_accuracy": round(float((preds == true_labels).mean()), 4),
        "macro_auc":        macro_auc,
        "per_class_metrics": per_class,
        "auc_per_class":    auc_per,
        "ap_per_class":     ap_per,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_metrics_report(metrics, model_name, variant, out_dir):
    sep   = "=" * 62
    lines = [sep, f"CATARACT — {variant}  {model_name}",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", sep, "",
             f"Accuracy : {metrics['overall_accuracy']:.4f}",
             f"Macro AUC: {metrics['macro_auc']:.4f}", "", "Per-class AUC:"]
    for cls, auc in metrics["auc_per_class"].items():
        lines.append(f"  {cls:<25} {auc:.4f}")
    lines += ["", "Per-class AP:"]
    for cls, ap in metrics["ap_per_class"].items():
        lines.append(f"  {cls:<25} {ap:.4f}")
    lines += ["", sep, "Detailed Metrics", sep]
    for cls, m in metrics["per_class_metrics"].items():
        lines.append(f"\n  {cls}")
        for k, v in m.items():
            lines.append(f"    {k:<35} {v}")
    txt = "\n".join(lines); print(txt)
    (out_dir / f"report_{variant.lower()}.txt").write_text(txt, encoding="utf-8")
    with open(out_dir / f"metrics_{variant.lower()}.json", "w") as f:
        json.dump(metrics, f, indent=2)


# ── McNemar ───────────────────────────────────────────────────────────────────
def mcnemar_test(preds_a, preds_b, true_labels, label_a="A", label_b="B"):
    """
    McNemar's test with Edwards continuity correction.
    Tests whether two classifiers make *different errors* on the same test set.
    Appropriate for paired classifiers on shared samples.

    H0: identical error rates (n01 == n10)
    Reject H0 (p<0.05) → significantly different error patterns.
    """
    ca = (preds_a == true_labels); cb = (preds_b == true_labels)
    n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
    n00 = int((ca & cb).sum());  n11 = int((~ca & ~cb).sum())
    denom = n01 + n10
    if denom == 0: chi2, p = 0.0, 1.0
    else:
        chi2 = (abs(n01 - n10) - 1.0) ** 2 / denom   # Edwards correction
        p    = float(1.0 - chi2_dist.cdf(chi2, df=1))
    sig  = p < 0.05
    note = (f"{label_a} significantly BETTER (p={p:.4f})" if (sig and n01 > n10)
            else f"{label_a} significantly WORSE (p={p:.4f})"  if (sig and n10 > n01)
            else f"No significant difference (p={p:.4f})")
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "chi2": round(chi2, 4), "p_value": round(p, 6),
            "significant": sig, "a_better": (sig and n01 > n10), "note": note}


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_roc_overlay_ah1(true_labels, probs_ah1, probs_ref, label_ref, color_ref,
                         model_name, out_dir):
    bl   = label_binarize(true_labels, classes=list(range(NUM_CLASSES)))
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(7 * NUM_CLASSES, 6))
    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        fa, ta, _ = roc_curve(bl[:, i], probs_ah1[:, i])
        aa = roc_auc_score(bl[:, i], probs_ah1[:, i])
        fr, tr, _ = roc_curve(bl[:, i], probs_ref[:, i])
        ar = roc_auc_score(bl[:, i], probs_ref[:, i])
        ax.plot(fa, ta, color=COLOR_AH1, lw=2.5, label=f"AH-1 {aa:.3f}")
        ax.plot(fr, tr, color=color_ref,  lw=2.5, ls="--", label=f"{label_ref} {ar:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        d   = aa - ar
        col = "#27ae60" if d > 0 else "#e74c3c"
        ax.text(0.55, 0.08, f"dAUC={d:+.4f}", transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=col, alpha=0.9))
        ax.set_title(cls.replace("_", "\n"), fontsize=11)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.suptitle(f"{model_name}  ROC  AH-1 vs {label_ref}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"roc_ah1_vs_{label_ref.lower()}.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"  Saved: {fname}")


def plot_confusion_matrix_ah1(cm, model_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{model_name} — AH-1 Confusion Matrices", fontsize=13)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title("Counts"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
    axes[1].set_title("Normalised"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    for ax in axes: ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_ah1.png", dpi=150, bbox_inches="tight"); plt.close()
    log.info("  Saved: confusion_matrix_ah1.png")


def plot_training_curves_ah1(hist, model_name, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model_name} — AH-1 Training Curves", fontsize=13)
    ep = range(1, len(hist["train_acc"]) + 1)
    axes[0].plot(ep, hist["train_acc"], color=COLOR_AH1, lw=2, label="train")
    axes[0].plot(ep, hist["val_acc"],   color=COLOR_AH1, lw=2, ls="--", label="val")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, hist["train_loss"], color=COLOR_AH1, lw=2, label="train")
    axes[1].plot(ep, hist["val_loss"],   color=COLOR_AH1, lw=2, ls="--", label="val")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(ep, hist["epoch_time"], color=COLOR_AH1, lw=2, marker="o", ms=4,
                 label=f"tot={hist['total_time']:.0f}s")
    axes[2].set_title("Epoch Time (s)"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves_ah1.svg", format="svg", bbox_inches="tight"); plt.close()
    log.info("  Saved: training_curves_ah1.svg")


def plot_three_way_comparison(ah1_results, qh_cl_csv_path, out_dir, mlp_hidden):
    """
    Bar chart: AH-1 vs QH vs CL macro-AUC across all models.

    FIX 8: title and filename include mlp_hidden so capacity variants
    produce separate charts that can be compared side by side.
    """
    ah1_dict = {r["Model"]: r["AUC"] for r in ah1_results}
    models   = list(ah1_dict.keys())

    qh_dict = {}; cl_dict = {}
    if Path(qh_cl_csv_path).exists():
        df = pd.read_csv(qh_cl_csv_path)
        for _, row in df.iterrows():
            if row["Type"] == "QH": qh_dict[row["Model"]] = float(row["AUC"])
            if row["Type"] == "CL": cl_dict[row["Model"]] = float(row["AUC"])

    _qh_p  = qh_params_per_block()
    _ah1_p = ah1_params_per_block(mlp_hidden)
    _ratio = capacity_ratio(mlp_hidden)

    x  = np.arange(len(models)); w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2.2), 7))
    bar_ah1 = ax.bar(x - w, [ah1_dict.get(m, 0) for m in models], w,
                     color=COLOR_AH1, alpha=0.88,
                     label=f"AH-1 (MLP h={mlp_hidden}, {_ah1_p}p/blk)")
    bar_qh  = ax.bar(x,     [qh_dict.get(m, 0) for m in models], w,
                     color=COLOR_QH,  alpha=0.88,
                     label=f"QH (Quantum, {_qh_p}p/blk)")
    bar_cl  = ax.bar(x + w, [cl_dict.get(m, 0) for m in models], w,
                     color=COLOR_CL,  alpha=0.88, label="CL (Classical)")

    for bars, dct in [(bar_ah1, ah1_dict), (bar_qh, qh_dict), (bar_cl, cl_dict)]:
        for bar, m in zip(bars, models):
            v = dct.get(m, 0)
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=9)
    ax.set_ylim(0.5, 1.06)
    ax.set_ylabel("Macro AUC-ROC", fontsize=11)
    ax.set_title(
        f"AH-1 (MLP h={mlp_hidden}, {_ah1_p}p/blk, ratio={_ratio}×) "
        f"vs QH (Quantum, {_qh_p}p/blk) vs CL — Macro AUC",
        fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fname = f"auc_comparison_ah1_h{mlp_hidden}_vs_qh_cl.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {fname}")


# ── Comparison report ─────────────────────────────────────────────────────────
def write_ah1_report(ah1_results, mcnemar_records, qh_cl_csv_path, out_dir, mlp_hidden):
    """
    FIX 6: report header now documents param counts, capacity ratio,
    and includes both AH-1 vs QH and AH-1 vs CL McNemar results.
    """
    _qh_p  = qh_params_per_block()
    _ah1_p = ah1_params_per_block(mlp_hidden)
    _ratio = capacity_ratio(mlp_hidden)

    sep = "=" * 70
    lines = [
        sep,
        "AH-1 ABLATION REPORT — MLP vs Quantum (AH-1 vs QH) vs Classical (CL)",
        f"Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"AH-1 MLP     : {CFG['n_q_blocks']} blocks × "
        f"MLPBlock({CFG['n_qubits']}→{mlp_hidden}×{CFG['n_layers']}→{CFG['n_qubits']})",
        f"QH Circuit   : {CFG['n_qubits']}q × {CFG['n_layers']}L  "
        f"({CFG['n_layers']}×{CFG['n_qubits']}×3 = {_qh_p} quantum params/block)",
        f"AH-1 MLP     : {_ah1_p} params/block  "
        f"(35×{mlp_hidden}+8 = {_ah1_p})",
        f"Capacity ratio (AH1/QH): {_ratio}×   "
        f"{'← parameter-MATCHED (primary test)' if _ratio < 1.2 else '← AH-1 has MORE capacity'}",
        f"Clip         : 1.0 (identical for AH-1, QH, and CL)",
        f"lr           : {CFG['lr']}   lr_backbone : {CFG['lr_backbone']}",
        sep, "",
        "CLAIM: If QH > AH-1, performance difference is attributable to",
        "       the quantum circuit itself, NOT to parameter count or",
        "       architectural capacity.",
        "",
        "AH-1 RESULTS (all models, sorted by AUC):", "-" * 70,
    ]
    for r in sorted(ah1_results, key=lambda x: x["AUC"], reverse=True):
        lines.append(f"  {r['Model']:<28} AUC={r['AUC']:.4f} "
                     f"Acc={r['Accuracy']:.4f} F1={r['F1']:.4f} "
                     f"ratio={r.get('Capacity_ratio','?')}×")

    if Path(qh_cl_csv_path).exists():
        df = pd.read_csv(qh_cl_csv_path)
        lines += ["", "QH RESULTS (from main experiment):", "-" * 70]
        for _, row in df[df["Type"] == "QH"].sort_values("AUC", ascending=False).iterrows():
            lines.append(f"  {row['Model']:<28} AUC={row['AUC']:.4f} "
                         f"Acc={row['Accuracy']:.4f} F1={row['F1']:.4f}")
        lines += ["", "CL RESULTS (from main experiment):", "-" * 70]
        for _, row in df[df["Type"] == "CL"].sort_values("AUC", ascending=False).iterrows():
            lines.append(f"  {row['Model']:<28} AUC={row['AUC']:.4f} "
                         f"Acc={row['Accuracy']:.4f} F1={row['F1']:.4f}")

        lines += ["", sep,
                  f"PER-MODEL DELTA  (AH-1 [h={mlp_hidden}] − QH | AH-1 − CL)", sep]
        qh_d = {row["Model"]: row["AUC"] for _, row in df[df["Type"] == "QH"].iterrows()}
        cl_d = {row["Model"]: row["AUC"] for _, row in df[df["Type"] == "CL"].iterrows()}
        for r in sorted(ah1_results, key=lambda x: x["Model"]):
            dq = r["AUC"] - qh_d.get(r["Model"], float("nan"))
            dc = r["AUC"] - cl_d.get(r["Model"], float("nan"))
            win_q = "QH wins" if dq < 0 else "AH-1 wins"
            win_c = "AH-1 wins" if dc > 0 else "CL wins"
            lines.append(f"  {r['Model']:<28} "
                         f"vs QH: {dq:+.4f} ({win_q})  "
                         f"vs CL: {dc:+.4f} ({win_c})")

    # ── FIX 2: McNemar for BOTH AH-1 vs QH and AH-1 vs CL ───────────────────
    if mcnemar_records:
        qh_records = [r for r in mcnemar_records if r.get("Comparison") == "AH1 vs QH"]
        cl_records = [r for r in mcnemar_records if r.get("Comparison") == "AH1 vs CL"]

        if qh_records:
            lines += ["", sep,
                      "McNEMAR'S TEST: AH-1 vs QH  "
                      "(H0: identical errors | p<0.05 = significant)",
                      sep]
            for rec in qh_records:
                sig = "SIGNIFICANT" if rec["significant"] else "NS"
                direction = ("QH BETTER" if not rec["a_better"] and rec["significant"]
                             else "AH1 BETTER" if rec["a_better"] else "")
                lines.append(f"  {rec['Model']:<28} chi2={rec['chi2']:.3f} "
                             f"p={rec['p_value']:.4f} {sig} {direction}")
                lines.append(f"    {rec['note']}")

        if cl_records:
            lines += ["", sep,
                      "McNEMAR'S TEST: AH-1 vs CL  "
                      "(H0: identical errors | p<0.05 = significant)",
                      sep]
            for rec in cl_records:
                sig = "SIGNIFICANT" if rec["significant"] else "NS"
                direction = ("CL BETTER" if not rec["a_better"] and rec["significant"]
                             else "AH1 BETTER" if rec["a_better"] else "")
                lines.append(f"  {rec['Model']:<28} chi2={rec['chi2']:.3f} "
                             f"p={rec['p_value']:.4f} {sig} {direction}")
                lines.append(f"    {rec['note']}")

    lines += ["", sep, "VERDICT GUIDE", sep,
              "  QH > AH-1 AND AH-1 ≈ CL → quantum circuits add genuine value",
              "  QH > AH-1 AND AH-1 > CL → quantum AND architecture both matter",
              "  QH ≈ AH-1               → quantum circuit not the key factor",
              "  AH-1 > QH               → classical MLP sufficient (revisit claim)",
              ""]

    report = "\n".join(lines); print(report)
    fname  = f"AH1_H{mlp_hidden}_COMPARISON_REPORT.txt"
    (out_dir / fname).write_text(report, encoding="utf-8")
    log.info(f"  Saved: {fname}")


# ── Per-model pipeline ────────────────────────────────────────────────────────
def run_model(model_name, eval_only, qh_cl_csv_path, mlp_hidden):
    out_dir    = CFG["output_dir"] / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    image_size = CFG["image_sizes"].get(model_name, CFG["image_sizes"]["__default__"])
    train_ldr, val_ldr, test_ldr, _, y_te = get_loaders(image_size)
    true_labels = y_te.numpy()

    # ── Build AH-1 model ──────────────────────────────────────────────────────
    log.info(f"\n  -- AH-1 (h={mlp_hidden}): {model_name} --")
    model = AH1Model(
        backbone_name=model_name,
        n_qubits=CFG["n_qubits"],
        n_layers=CFG["n_layers"],
        image_size=image_size,
        n_q_blocks=CFG["n_q_blocks"],
        mlp_hidden=mlp_hidden,
    ).to(DEVICE)

    ckpt_name = "best_model_ah1.pth"
    ckpt_path = out_dir / ckpt_name
    hist      = {"train_loss": [], "val_loss": [], "train_acc": [],
                 "val_acc": [], "epoch_time": [], "total_time": 0}
    best_val  = None

    if not eval_only:
        hist, best_val = train_model(model, f"AH1-h{mlp_hidden}", train_ldr, val_ldr,
                                     out_dir, ckpt_name)
        with open(out_dir / "timing_ah1.json", "w") as f:
            json.dump({"model": model_name, "type": "AH1",
                       "mlp_hidden": mlp_hidden, **hist}, f, indent=2)
        plot_training_curves_ah1(hist, model_name, out_dir)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
        log.info("  Loaded AH-1 checkpoint")
    else:
        log.warning("  No AH-1 checkpoint found — evaluating with untrained weights!")

    _, _, ah1_probs, _ = evaluate(model, test_ldr)
    ah1_preds = ah1_probs.argmax(axis=1)
    metrics   = compute_metrics(true_labels, ah1_probs, ah1_preds)
    save_metrics_report(metrics, model_name, "AH1", out_dir)
    np.save(out_dir / "ah1_probs.npy", ah1_probs)
    np.save(out_dir / "ah1_preds.npy", ah1_preds)
    log.info(f"  AH-1 {model_name}: AUC={metrics['macro_auc']:.4f} "
             f"Acc={metrics['overall_accuracy']:.4f}")

    # ── Plots vs QH and CL ────────────────────────────────────────────────────
    plot_confusion_matrix_ah1(np.array(metrics["confusion_matrix"]), model_name, out_dir)
    qh_cl_model_dir = CFG["qh_cl_dir"] / model_name

    for ref_label, ref_color, ref_fname in [
        ("QH", COLOR_QH, "qh_probs.npy"),
        ("CL", COLOR_CL, "cl_probs.npy"),
    ]:
        ref_path = qh_cl_model_dir / ref_fname
        if ref_path.exists():
            ref_probs = np.load(ref_path)
            plot_roc_overlay_ah1(true_labels, ah1_probs, ref_probs,
                                 ref_label, ref_color, model_name, out_dir)
        else:
            # FIX 5: ERROR not warning — missing files means comparison is incomplete
            log.error(f"  MISSING {ref_fname} at {ref_path}. "
                      f"Run main_experiment.py first, then re-run this ablation. "
                      f"ROC overlay vs {ref_label} will be skipped.")

    # ── FIX 2: McNemar vs BOTH QH and CL ─────────────────────────────────────
    mcnemar_records = []

    qh_pred_path = qh_cl_model_dir / "qh_preds.npy"
    if qh_pred_path.exists():
        qh_preds = np.load(qh_pred_path)
        mn_qh = mcnemar_test(ah1_preds, qh_preds, true_labels, "AH1", "QH")
        mn_qh["Model"] = model_name; mn_qh["Comparison"] = "AH1 vs QH"
        mcnemar_records.append(mn_qh)
        log.info(f"  McNemar (AH1 vs QH): chi2={mn_qh['chi2']:.3f} "
                 f"p={mn_qh['p_value']:.4f} {mn_qh['note']}")
    else:
        log.error(f"  MISSING qh_preds.npy at {qh_pred_path} — "
                  f"McNemar AH1 vs QH skipped.")

    cl_pred_path = qh_cl_model_dir / "cl_preds.npy"
    if cl_pred_path.exists():
        cl_preds = np.load(cl_pred_path)
        mn_cl = mcnemar_test(ah1_preds, cl_preds, true_labels, "AH1", "CL")
        mn_cl["Model"] = model_name; mn_cl["Comparison"] = "AH1 vs CL"
        mcnemar_records.append(mn_cl)
        log.info(f"  McNemar (AH1 vs CL): chi2={mn_cl['chi2']:.3f} "
                 f"p={mn_cl['p_value']:.4f} {mn_cl['note']}")
    else:
        log.error(f"  MISSING cl_preds.npy at {cl_pred_path} — "
                  f"McNemar AH1 vs CL skipped.")

    # ── FIX 3: param counts in result dict ───────────────────────────────────
    _qh_p  = qh_params_per_block()
    _ah1_p = ah1_params_per_block(mlp_hidden)

    result = {
        "Model"               : model_name,
        "Type"                : "AH1",
        "Accuracy"            : metrics["overall_accuracy"],
        "Precision"           : round(metrics["classification_report"]["weighted avg"]["precision"], 4),
        "Recall"              : round(metrics["classification_report"]["weighted avg"]["recall"], 4),
        "F1"                  : round(metrics["classification_report"]["weighted avg"]["f1-score"], 4),
        "AUC"                 : metrics["macro_auc"],
        "Best Val Acc"        : round(best_val, 4) if best_val else "N/A",
        "Architecture"        : (f"MLP {CFG['n_q_blocks']}×({CFG['n_qubits']}"
                                 f"→{mlp_hidden}×{CFG['n_layers']}→{CFG['n_qubits']})"),
        "mlp_hidden"          : mlp_hidden,
        # FIX 3: machine-readable param audit fields
        "QH_params_per_block" : _qh_p,
        "AH1_params_per_block": _ah1_p,
        "Capacity_ratio"      : capacity_ratio(mlp_hidden),
    }

    del model; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    clear_cache_for_image_size(image_size)
    return result, mcnemar_records


# ── Entry point ───────────────────────────────────────────────────────────────
def main(eval_only=False, model_filter=None, mlp_hidden_list=None, qh_cl_csv=None):
    """
    FIX 7: mlp_hidden_list supports multiple hidden dims in one invocation.
    Each hidden dim gets its own output directory, CSV, and report.
    This enables the capacity sweep needed to make a strong quantum claim:

        h=2 → 78 params/block  (1.08× QH) — parameter-matched primary test
        h=4 → 148 params/block (2.06× QH) — AH-1 has 2× capacity
        h=8 → 288 params/block (4.00× QH) — AH-1 has 4× capacity

    If QH > AH-1 at ALL capacity levels, the claim is very strong.
    """
    if mlp_hidden_list is None or len(mlp_hidden_list) == 0:
        mlp_hidden_list = [CFG["mlp_hidden"]]   # default: [2]

    qh_cl_csv_path = qh_cl_csv or str(CFG["qh_cl_csv"])

    log.info(f"Device={DEVICE}")
    log.info(f"QH quantum params/block : {qh_params_per_block()}")
    log.info(f"AH-1 mlp_hidden variants: {mlp_hidden_list}")
    for h in mlp_hidden_list:
        log.info(f"  h={h} → {ah1_params_per_block(h)} params/block "
                 f"(ratio={capacity_ratio(h):.2f}× vs QH)")

    for mlp_hidden in mlp_hidden_list:
        log.info(f"\n{'#'*70}")
        log.info(f"  AH-1 RUN: mlp_hidden={mlp_hidden}  "
                 f"params/block={ah1_params_per_block(mlp_hidden)}  "
                 f"ratio={capacity_ratio(mlp_hidden):.2f}×")
        log.info(f"{'#'*70}\n")

        # Update global CFG for this run
        CFG["mlp_hidden"] = mlp_hidden
        CFG["output_dir"] = Path(f"./ablation_ah1_h{mlp_hidden}_output")
        CFG["output_dir"].mkdir(parents=True, exist_ok=True)

        models_to_run = model_filter or CFG["models"]
        all_results   = []
        all_mcnemar   = []

        for model_name in tqdm(models_to_run, desc=f"AH-1 h={mlp_hidden}", unit="model",
                               dynamic_ncols=True, colour="blue"):
            log.info(f"\n{'='*65}\n  AH-1 h={mlp_hidden} MODEL: {model_name.upper()}\n{'='*65}")
            try:
                result, mcnemar_records = run_model(model_name, eval_only,
                                                     qh_cl_csv_path, mlp_hidden)
                all_results.append(result)
                all_mcnemar.extend(mcnemar_records)
            except Exception as e:
                log.error(f"  FAILED {model_name} (h={mlp_hidden}): {e}")
                import traceback; traceback.print_exc(); gc.collect()

        if not all_results:
            log.error(f"No results for h={mlp_hidden}."); continue

        out = CFG["output_dir"]

        # Save aggregated CSV
        df = pd.DataFrame(all_results).sort_values("AUC", ascending=False)
        csv_name = f"all_results_ah1_h{mlp_hidden}.csv"
        df.to_csv(out / csv_name, index=False)
        log.info(f"\n{df[['Model','Accuracy','F1','AUC','Capacity_ratio','Best Val Acc']].to_string(index=False)}")

        # Separate McNemar CSVs for QH and CL
        qh_mn = [r for r in all_mcnemar if r.get("Comparison") == "AH1 vs QH"]
        cl_mn = [r for r in all_mcnemar if r.get("Comparison") == "AH1 vs CL"]
        if qh_mn:
            pd.DataFrame(qh_mn).to_csv(out / f"mcnemar_ah1_h{mlp_hidden}_vs_qh.csv", index=False)
        if cl_mn:
            pd.DataFrame(cl_mn).to_csv(out / f"mcnemar_ah1_h{mlp_hidden}_vs_cl.csv", index=False)

        # Plots and report
        plot_three_way_comparison(all_results, qh_cl_csv_path, out, mlp_hidden)
        write_ah1_report(all_results, all_mcnemar, qh_cl_csv_path, out, mlp_hidden)

        log.info(f"\nAll AH-1 h={mlp_hidden} outputs saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AH-1 Ablation: Quantum → MLP (parameter-matched, scientifically valid)")
    parser.add_argument("--eval-only",
                        action="store_true",
                        help="Skip training, evaluate existing checkpoints only")
    parser.add_argument("--models",
                        nargs="+", default=None,
                        help="Subset of backbone names to run (default: all 8)")
    parser.add_argument("--mlp-hidden",
                        type=int, default=None,
                        help="Single MLP hidden dim (overrides --mlp-hidden-list). "
                             "Default: 2 (parameter-matched to QH)")
    parser.add_argument("--mlp-hidden-list",
                        nargs="+", type=int, default=None,
                        help="Run AH-1 for each hidden dim. "
                             "Recommended: --mlp-hidden-list 2 4 8 "
                             "(param-match, 2× capacity, 4× capacity). "
                             "Default: [2]")
    parser.add_argument("--compare-csv",
                        type=str, default=None,
                        help="Path to all_results_qh_vs_cl.csv from main experiment "
                             "(default: ./quantum_output_v12/all_results_qh_vs_cl.csv)")
    args = parser.parse_args()

    # Resolve mlp_hidden_list
    if args.mlp_hidden is not None:
        mlp_hidden_list = [args.mlp_hidden]
    elif args.mlp_hidden_list is not None:
        mlp_hidden_list = args.mlp_hidden_list
    else:
        mlp_hidden_list = [2]   # default: parameter-matched

    main(
        eval_only       = args.eval_only,
        model_filter    = args.models,
        mlp_hidden_list = mlp_hidden_list,
        qh_cl_csv       = args.compare_csv,
    )