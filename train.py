"""
Cataract Classification — Quantum-Hybrid vs Classical (v8 — scientifically fair)
=================================================================================
Target : Intel i7-9700 (8 cores) — CPU-only

FAIRNESS AUDIT FIXES (v8 — no model gets special treatment):
  1. Identical grad-clip (1.0) for both QH and CL — no asymmetric clip
  2. _make_optimizer: same LR-group structure for QH and CL
     (backbone=lr_backbone, everything-else=lr) — quantum params use lr, not a
     separate lr_qc, so neither model receives a hidden extra tuning advantage
  3. Collapse detection / _reinit_projection removed entirely — CL never had
     this rescue mechanism; giving it only to QH is an adaptive advantage
  4. write_auc_proof_report: no AUC-threshold filtering of QH models;
     ALL models (QH and CL) are reported unconditionally
  5. plot_per_model_auc_comparison: neutral bar chart — no red «collapsed» colour
     coding, no ★ annotations, no special QH winner markers
  6. Result dict «Circuit» field updated for CL (now MLP head, not linear)
  7. Output dir → quantum_output_v8

PRESERVED (unchanged): quantum circuit improvements from v7 (Hadamard, RY+RZ,
  ring entanglement, alpha gate, dropout_q), matched MLP head depth, removed
  torch.no_grad() from CL forward, scheduler re-creation after unfreeze,
  all plots / reports / McNemar test.
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
import matplotlib.colors as mcolors
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
    import pennylane as qml
except ImportError:
    print("ERROR: pip install pennylane pennylane-lightning")
    raise

try:
    from tqdm.auto import tqdm
except ImportError:
    print("WARNING: pip install tqdm — falling back to no-progress-bar mode")
    def tqdm(it, **kw): return it   # silent no-op fallback

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CFG = {
    "data_dir"   : Path("./dataset"),
    "output_dir" : Path("./quantum_output_v12"),
    "classical_results_csv": Path("./multimodel1_output/final_results.csv"),
    "seed"        : 42,
    "classes"     : ["Immature_Cataract", "Mature_Cataract", "Normal_Eye"],
    "epochs"      : 35,          # 30–40 range; QH and CL both benefit from longer training
    "batch_size"  : 16,          # 16 for RTX 3060: balances GPU throughput vs quantum loop
    "lr"          : 2e-4,        # same for QH and CL — no exclusive advantage
    "lr_min"      : 1e-6,
    "lr_backbone" : 5e-5,        # phase-2 backbone LR (same for both)
    # lr_qc removed: optimizer uses only 2 groups (backbone + all others)
    # Adding a 3rd group exclusively for QH would break fairness
    "patience"    : 8,
    "weight_decay": 1e-4,
    "label_smooth": 0.1,

    "num_workers" : 4,
    "warmup_epochs": 4,
    "n_qubits"    : 8,           # qubits per block (CPU-feasible: 2⁸=256 amplitudes)
    "n_layers"    : 3,           # circuit layers per block
    "n_q_blocks"  : 4,           # parallel blocks → 8×4=32 effective quantum outputs
    "proj_hidden" : 256,
    "models": [
        "resnet50","densenet121","inception_v3","mobilenetv3_large_100",
        "convnext_tiny","efficientnet_b0","repvgg_b3","vit_base_patch16_224",
    ],
    "image_sizes": {"inception_v3": 299, "__default__": 224},

}
CFG["unfreeze_models"] = set(CFG["models"])
CLASSES     = CFG["classes"]
NUM_CLASSES = len(CLASSES)
# ── Device & backend ─────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.benchmark     = True   # auto-tune conv kernels (fixed input size)
    torch.backends.cudnn.deterministic = False   # allow non-deterministic faster kernels
    torch.backends.cuda.matmul.allow_tf32 = True # TF32 for matmul (Ampere+)
    torch.backends.cudnn.allow_tf32    = True    # TF32 for conv (Ampere+)
    torch.cuda.manual_seed(CFG["seed"])
    _gpu = torch.cuda.get_device_properties(0)
    log.info(f"CUDA: {_gpu.name} | {_gpu.total_memory//1024**3}GB | "
             f"cuDNN {torch.backends.cudnn.version()}")
else:
    DEVICE = torch.device("cpu")
    torch.set_num_threads(8); torch.set_num_interop_threads(4)
    torch.backends.mkldnn.enabled = True
    torch.backends.cpu.allow_bf16_reduced_precision_reduction = True
    log.info("CUDA not available — running on CPU (slower)")

_PIN_MEM = (DEVICE.type == "cuda")   # pin_memory speeds up CPU→GPU transfers
COLOR_QH = "#7B52AB"
COLOR_CL  = "#3B82C4"
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
log.info(f"Device={DEVICE} | PennyLane {qml.__version__}")

# ── Quantum circuit ──────────────────────────────────────────────────────────
def _make_device(n_qubits):
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        log.info(f"  lightning.qubit ({n_qubits}q)")
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        log.info(f"  default.qubit ({n_qubits}q)")
    return dev

def build_quantum_layer(n_qubits, n_layers):
    dev = _make_device(n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        # Hadamard initialisation — puts all qubits in uniform |+⟩ superposition
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        for l in range(n_layers):
            # Dual-axis data re-uploading: RY (amplitude) + RZ (phase)
            for i in range(n_qubits):
                qml.RY(torch.pi * torch.tanh(inputs[i]), wires=i)
                qml.RZ(torch.pi * inputs[i],              wires=i)

            # Ladder entanglement + ring closure
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

            # Trainable Euler-angle rotations per qubit
            for i in range(n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

        # Mixed PauliZ + PauliX measurements:
        #   Even qubits → PauliZ (computational basis: amplitude info)
        #   Odd  qubits → PauliX (Hadamard basis: phase / interference info)
        # This doubles the observable diversity without adding qubits,
        # giving the model access to both amplitude and phase information flow.
        return [
            qml.expval(qml.PauliZ(i)) if i % 2 == 0 else qml.expval(qml.PauliX(i))
            for i in range(n_qubits)
        ]

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)})

# ── Backbone utils ───────────────────────────────────────────────────────────
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

def _partial_unfreeze(bb, name):
    if name == "vit_base_patch16_224":
        for p in bb.blocks[-1].parameters(): p.requires_grad_(True)
        log.info("  [QH] Unfroze ViT block[-1]")
    elif name == "convnext_tiny":
        for p in bb.stages[-1].parameters(): p.requires_grad_(True)
        log.info("  [QH] Unfroze ConvNeXt stages[-1]")

# ── Models ───────────────────────────────────────────────────────────────────
class QuantumHybridModel(nn.Module):
    """
    Hybrid quantum-classical architecture (v3 — multi-block).

    Backbone → Projection (dim→512→256) → Qubit Encoder (256→n_q_blocks×n_qubits)
             → n_q_blocks parallel quantum circuits
             → Concatenate [backbone_256, q_out_scaled] → Head.

    Key design choices:
      • n_q_blocks parallel circuits (default 4×8 = 32 effective quantum outputs)
        gives the quantum component 32-output influence while keeping each
        individual circuit CPU-feasible (2⁸ = 256 state-vector amplitudes).
      • log_scale: learnable scalar gates the angle magnitude entering circuits.
      • q_out_scale: learnable per-output weight applied AFTER the circuits.
        Allows the head to up-weight or suppress individual quantum features.
      • Backbone features (256-dim) are concatenated with all quantum outputs
        (n_q_blocks×n_qubits-dim) before the classification head.
      • Per-sample quantum loop is minimised by using batch_size=8 (8 iterations
        vs 32) and by trying native PennyLane batch execution first.
    """
    def __init__(self, backbone_name, n_qubits, n_layers, image_size,
                 n_q_blocks=None):
        super().__init__()
        self.model_type  = "QH"; self.backbone_name = backbone_name
        self.backbone, dim = _make_backbone(backbone_name, image_size)
        self._n_qubits   = n_qubits
        self._n_q_blocks = n_q_blocks or CFG.get("n_q_blocks", 1)
        total_q_out      = self._n_qubits * self._n_q_blocks  # 8×4 = 32
        h1, h2  = 512, CFG["proj_hidden"]   # backbone compressor dims
        fused_gate_dim = h2                               # 256 — shared gating dim
        log.info(f"  [QH] dim={dim} proj={dim}->{h1}->{h2}->{total_q_out} "
                 f"({self._n_q_blocks} blocks×{n_qubits}q) "
                 f"gate_fused={fused_gate_dim} layers={n_layers}")

        # Backbone compressor: dim → 512 → 256
        # (named 'projection' so freeze_backbone() keeps it trainable in phase 1)
        self.projection = nn.Sequential(
            nn.Linear(dim, h1), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(0.1),
        )

        # Qubit encoder: 256 → total_q_out (outputs split per block)
        # (named 'projection_qubit' → contains 'projection' → trainable in phase 1)
        self.projection_qubit = nn.Sequential(
            nn.Linear(h2, total_q_out),
            nn.LayerNorm(total_q_out),
        )

        # n_q_blocks independent quantum circuits
        self.quantum_layers = nn.ModuleList([
            build_quantum_layer(n_qubits, n_layers)
            for _ in range(self._n_q_blocks)
        ])

        # Learnable input scale: angles *= exp(log_scale) before entering circuit
        self.log_scale   = nn.Parameter(torch.zeros(1))   # exp(0)=1.0 at init

        # Learnable per-output scale applied after quantum circuits
        self.q_out_scale = nn.Parameter(torch.ones(total_q_out))

        # ── Gated Fusion ───────────────────────────────────────────────────────────────
        # Both backbone_feat (h2=256) and q_out (total_q_out=48) are projected
        # to the same fused_gate_dim (256) then blended by a sigmoid gate.
        # gate  = σ( W_gate · [backbone_feat; q_out] )  ∈ (0,1)²⁵⁶
        # fused = gate ⊙ bb_proj + (1−gate) ⊙ q_proj
        # This lets the model dynamically up-weight backbone OR quantum path
        # on a per-sample, per-feature basis without hard-coded blending.
        self.fusion_bb   = nn.Linear(h2, fused_gate_dim)          # backbone path
        self.fusion_q    = nn.Linear(total_q_out, fused_gate_dim)  # quantum path
        self.fusion_gate = nn.Sequential(
            nn.Linear(h2 + total_q_out, fused_gate_dim),
            nn.Sigmoid(),
        )

        # BN and head operate on the fused (gated) tensor
        self.bn        = nn.BatchNorm1d(fused_gate_dim)
        self.dropout_q = nn.Dropout(0.1)
        self.head      = nn.Linear(fused_gate_dim, NUM_CLASSES)

    @staticmethod
    def _run_qblock(ql, x):
        """
        Execute one quantum block on a batch x [B, n_q].
        Tries native PennyLane batch execution first (faster).
        Falls back reliably to a per-sample loop.
        """
        try:
            out = ql(x)                               # native batch attempt
            # TorchLayer may return [n_q, B] or [B, n_q] depending on version
            if out.dim() == 2:
                return out if out.shape[0] == x.shape[0] else out.T
            if out.dim() == 1:                        # single sample edge-case
                return out.unsqueeze(0)
        except Exception:
            pass
        # Reliable fallback: explicit per-sample loop
        return torch.stack([ql(x[i]) for i in range(x.shape[0])])

    def _quantum_forward(self, angles):
        """Run all quantum blocks and return concatenated, scaled output.

        PennyLane quantum circuits always execute on CPU (lightning.qubit is a
        CPU simulator). When the model is on GPU, angles arrive on CUDA.
        We transfer to CPU for quantum execution, then back to DEVICE for the
        rest of the forward pass. PyTorch autograd handles the CPU↔GPU gradient
        flow transparently — no .detach() is used.
        """
        device = angles.device
        # Transfer to CPU for quantum computation (autograd-safe: no detach)
        angles_cpu    = angles.cpu()
        # Clamp to [-1, 1] BEFORE scaling to prevent extreme circuit inputs
        # that cause barren plateau-like gradient vanishing
        angles_cpu    = torch.clamp(angles_cpu, -1.0, 1.0)
        log_scale_cpu = self.log_scale.cpu()
        scaled        = angles_cpu * torch.exp(log_scale_cpu)   # learnable in-scale
        chunks        = scaled.chunk(self._n_q_blocks, dim=1)   # split per block
        q_blocks      = [self._run_qblock(ql, chunk)
                         for ql, chunk in zip(self.quantum_layers, chunks)]
        q_out_cpu     = torch.cat(q_blocks, dim=1)              # [B, total_q]
        # Back to original device; q_out_scale lives on DEVICE for efficient scaling
        return q_out_cpu.to(device) * self.q_out_scale          # learnable out-scale

    def forward(self, x):
        feats         = self.backbone(x)                        # [B, dim]
        backbone_feat = self.projection(feats)                  # [B, 256]
        angles        = self.projection_qubit(backbone_feat)    # [B, total_q]
        q_out         = self._quantum_forward(angles)           # [B, total_q]
        # ── Gated Fusion ─────────────────────────────────────────────────────
        bb_proj = self.fusion_bb(backbone_feat)                 # [B, 256]
        q_proj  = self.fusion_q(q_out)                          # [B, 256]
        gate    = self.fusion_gate(
            torch.cat([backbone_feat, q_out], dim=1)            # [B, 256+total_q]
        )                                                       # [B, 256] ∈ (0,1)
        fused   = gate * bb_proj + (1.0 - gate) * q_proj       # [B, 256]
        return self.head(self.bn(self.dropout_q(fused)))

    def cached_forward(self, feats):
        """Skip backbone. Receives pre-extracted backbone features directly.
        Used during warmup when backbone is frozen.
        """
        backbone_feat = self.projection(feats)
        angles        = self.projection_qubit(backbone_feat)
        q_out         = self._quantum_forward(angles)
        bb_proj = self.fusion_bb(backbone_feat)
        q_proj  = self.fusion_q(q_out)
        gate    = self.fusion_gate(torch.cat([backbone_feat, q_out], dim=1))
        fused   = gate * bb_proj + (1.0 - gate) * q_proj
        return self.head(self.bn(self.dropout_q(fused)))

    def trainable_params(self): return [p for p in self.parameters() if p.requires_grad]
    def param_counts(self):
        return (sum(p.numel() for p in self.parameters()),
                sum(p.numel() for p in self.parameters() if p.requires_grad))

class ClassicalBaselineModel(nn.Module):
    """
    Classical baseline with matched head capacity.

    Backbone → BN → MLP head (dim→256→3).
    The MLP head mirrors QH's projection depth so classification capacity is equal.
    Backbone is frozen during warmup then fine-tuned in phase 2, identical to QH.
    No torch.no_grad() in forward — freeze is enforced via requires_grad_(False),
    which allows phase-2 gradients to flow through the backbone properly.
    """
    def __init__(self, backbone_name, image_size):
        super().__init__()
        self.model_type = "CL"; self.backbone_name = backbone_name
        self.backbone, dim = _make_backbone(backbone_name, image_size)
        _freeze(self.backbone)
        h = CFG["proj_hidden"]
        log.info(f"  [CL] dim={dim} head={dim}->{h}->{NUM_CLASSES}  (matched MLP depth)")
        self.bn = nn.BatchNorm1d(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, h),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(h, NUM_CLASSES),
        )

    def forward(self, x):
        feats = self.backbone(x)   # gradients flow once backbone is unfrozen in phase 2
        return self.head(self.bn(feats))

    def cached_forward(self, feats):
        """Skip backbone. Receives pre-extracted backbone features directly.
        Used during warmup when backbone is frozen — applies BN → head only.
        Mirrors QuantumHybridModel.cached_forward so train_one_epoch / evaluate
        can call model.cached_forward(x) unconditionally for both QH and CL
        without any model-type conditional branching.
        """
        return self.head(self.bn(feats))

    def trainable_params(self): return [p for p in self.parameters() if p.requires_grad]
    def param_counts(self):
        return (sum(p.numel() for p in self.parameters()),
                sum(p.numel() for p in self.parameters() if p.requires_grad))

# ── Data ─────────────────────────────────────────────────────────────────────
def get_transforms(sz):
    return transforms.Compose([
        transforms.Resize((sz, sz)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

_tensor_cache: dict = {}
_loader_cache: dict = {}

def _build_tensor_cache(split, image_size):
    key = (split, image_size)
    if key in _tensor_cache: return _tensor_cache[key]
    log.info(f"  Caching '{split}' ({image_size}px)...")
    t0 = time.time()
    ds = datasets.ImageFolder(f"./dataset/{split}", transform=get_transforms(image_size))
    if split == "train" and ds.classes != CLASSES:
        raise ValueError(f"Folder classes {ds.classes} != {CLASSES}")
    ldr = DataLoader(ds, batch_size=64, shuffle=False,
                     num_workers=CFG["num_workers"], pin_memory=False)
    Xs, ys = [], []
    for x, y in ldr: Xs.append(x); ys.append(y)
    X, y = torch.cat(Xs), torch.cat(ys)
    log.info(f"  Cached {len(y)} in {time.time()-t0:.1f}s | {X.element_size()*X.nelement()/1e6:.0f} MB")
    _tensor_cache[key] = (X, y, ds); return _tensor_cache[key]

def get_loaders(image_size):
    if image_size in _loader_cache: return _loader_cache[image_size]
    X_tr,y_tr,_   = _build_tensor_cache("train", image_size)
    X_va,y_va,_   = _build_tensor_cache("val",   image_size)
    X_te,y_te,dst = _build_tensor_cache("test",  image_size)
    counts = torch.bincount(y_tr)
    log.info(f"  Class counts: {dict(zip(CLASSES, counts.tolist()))}")
    sw = (1.0/counts.float())[y_tr]
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
    # pin_memory=True speeds up CPU→GPU transfer via non_blocking=True in train loop
    kw = dict(num_workers=0, pin_memory=_PIN_MEM)
    trl = DataLoader(TensorDataset(X_tr,y_tr), batch_size=CFG["batch_size"], sampler=sampler, **kw)
    val = DataLoader(TensorDataset(X_va,y_va), batch_size=CFG["batch_size"], shuffle=False, **kw)
    tel = DataLoader(TensorDataset(X_te,y_te), batch_size=CFG["batch_size"], shuffle=False, **kw)
    _loader_cache[image_size] = (trl, val, tel, dst, y_te)
    return _loader_cache[image_size]

def clear_cache_for_image_size(image_size):
    for s in ("train","val","test"): _tensor_cache.pop((s,image_size), None)
    _loader_cache.pop(image_size, None); gc.collect()

# ── Loss ─────────────────────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, s=0.1):
        super().__init__(); self.s = s
    def forward(self, pred, target):
        n = pred.size(-1)
        lp = nn.functional.log_softmax(pred, dim=-1)
        sm = torch.full_like(lp, self.s/(n-1))
        sm.scatter_(-1, target.unsqueeze(-1), 1.0-self.s)
        return -(sm*lp).sum(dim=-1).mean()

CRITERION = LabelSmoothingCE(CFG["label_smooth"])

# ── Train / Eval ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, use_cached=False):
    """
    Single training epoch. Identical for QH and CL — no special treatment.
    Moves data to DEVICE (non_blocking=True with pinned memory for speed).
    use_cached=True: loader yields pre-extracted (features, labels) tensors.
    """
    model.train()
    loss_sum = correct = total = 0
    CLIP_VAL = 1.0
    pbar = tqdm(loader, desc="  train", leave=False, unit="bat",
                dynamic_ncols=True, colour="cyan")
    for x, labels in pbar:
        x      = x.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out  = model.cached_forward(x) if use_cached else model(x)
        loss = CRITERION(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.trainable_params(), CLIP_VAL)
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct  += out.detach().argmax(1).eq(labels).sum().item()
        total    += x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
    return loss_sum/total, correct/total

@torch.inference_mode()
def evaluate(model, loader, use_cached=False):
    """
    Evaluation loop. Moves data to DEVICE; collects results on CPU for sklearn.
    use_cached=True: loader yields pre-extracted (features, labels) tensors.
    """
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
        # .cpu() required: GPU tensors cannot be converted to numpy directly
        probs_l.append(p.cpu().numpy()); labels_l.append(labels.cpu().numpy())
    return loss_sum/total, correct/total, np.concatenate(probs_l), np.concatenate(labels_l)

# _is_collapsed is no longer used (collapse detection removed in v8)
# Kept as placeholder to avoid grep/reference errors.
_is_collapsed = lambda val_acc: val_acc <= (1/NUM_CLASSES + 0.02)  # noqa: E731

@torch.no_grad()
def _cache_backbone_features(backbone, loader, is_train=True):
    """
    Pre-extract backbone features once from a FROZEN backbone.
    Backbone runs on DEVICE (GPU if available); features stored on CPU to
    avoid holding GPU VRAM for the whole dataset across warmup epochs.
    The training loop moves feature batches to DEVICE on demand (non_blocking).
    """
    backbone.eval()
    Xf, Yf = [], []
    split = "train" if is_train else "val/test"
    for imgs, y in tqdm(loader, desc=f"  cache {split}", leave=False,
                        unit="bat", dynamic_ncols=True, colour="yellow"):
        imgs = imgs.to(DEVICE, non_blocking=True)   # backbone on GPU
        Xf.append(backbone(imgs).cpu())              # store on CPU (save VRAM)
        Yf.append(y)                                  # labels already on CPU
    X = torch.cat(Xf); Y = torch.cat(Yf)
    if is_train:
        counts = torch.bincount(Y)
        sw = (1.0 / counts.float())[Y]
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        return DataLoader(TensorDataset(X, Y), batch_size=CFG["batch_size"],
                          sampler=sampler, num_workers=0, pin_memory=_PIN_MEM)
    return DataLoader(TensorDataset(X, Y), batch_size=CFG["batch_size"] * 2,
                      shuffle=False, num_workers=0, pin_memory=_PIN_MEM)

def freeze_backbone(model):
    """Phase-1: freeze everything except head, projection, and quantum layers."""
    for name, param in model.named_parameters():
        if "head" not in name and "projection" not in name and "quantum" not in name:
            param.requires_grad_(False)

def unfreeze_all(model):
    """Phase-2: release all parameters so backbone also gets gradients."""
    for param in model.parameters():
        param.requires_grad_(True)

def _make_optimizer(model, phase=1):
    """
    Build AdamW with per-component learning rates.
    Strictly identical logic for both QH and CL — no model-type branching.

    Phase 1 (warmup, frozen backbone):
      Single LR group — all trainable params at CFG["lr"].

    Phase 2 (full fine-tune, backbone unfrozen):
      Exactly TWO groups for both QH and CL:
        backbone params → CFG["lr_backbone"]  (slow, prevents forgetting)
        all other params → CFG["lr"]          (head, projection, quantum, gates, ...)

    Note: lr_qc has been removed from CFG. Giving quantum params a 3rd LR group
    exclusively for QH would break fairness, so all non-backbone params share
    one rate regardless of model type.
    """
    if phase == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    # Phase 2: two groups — backbone (slow), head/projection/quantum (normal)
    bb_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if "backbone" in n:
            bb_params.append(p)
        else:
            other_params.append(p)
    groups = [{"params": other_params, "lr": CFG["lr"]}]
    if bb_params:
        groups.append({"params": bb_params, "lr": CFG["lr_backbone"]})
    return optim.AdamW(groups, weight_decay=CFG["weight_decay"])

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _save_resume_ckpt(path, epoch, phase, model, opt, sch,
                      history, best_val, patience_ctr):
    """
    Save a full resume snapshot at the END of every epoch.

    Contains everything needed to restart training from the next epoch with
    identical optimizer and scheduler state:
      epoch        — last completed epoch (resume starts from epoch+1)
      phase        — 1 (warmup) or 2 (full fine-tune)
      model_state  — model weights
      opt_state    — AdamW momentum / variance buffers
      sch_state    — CosineAnnealingLR t_cur, last_epoch
      history      — full training log so far
      best_val     — best validation accuracy seen (for patience)
      patience_ctr — consecutive epochs without improvement
    """
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


def train_model(model, label, train_ldr, val_ldr, out_dir, ckpt_name):
    """
    Two-phase training — identical protocol for QH and CL.

    Phase 1 (warmup, frozen backbone):
      Backbone features are pre-extracted ONCE via _cache_backbone_features().
      Warmup epochs iterate over cached (features, labels) — backbone forward
      runs only once total instead of WARMUP × N_batches times.
      Features are stored on CPU to conserve GPU VRAM; moved to DEVICE per batch.

    Phase 2 (full fine-tune, backbone unfrozen):
      End-to-end training with full image DataLoader. Separate backbone LR.

    Crash Recovery:
      A resume checkpoint (resume_{ckpt_name}) is saved at the END of every
      epoch. On re-launch, if the file is found, all training state is restored
      and the loop continues from the next epoch. The file is deleted on clean
      completion.
    """
    WARMUP      = CFG["warmup_epochs"]
    resume_path = out_dir / f"resume_{ckpt_name}"

    # ── Initialise or restore from crash ──────────────────────────────────────
    if resume_path.exists():
        log.info(f"  [{label}] ⚠️  Resume checkpoint found — restoring state ...")
        ckpt        = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        start_epoch = ckpt["epoch"] + 1      # resume from *next* epoch
        phase       = ckpt["phase"]
        best_val    = ckpt["best_val"]
        patience_ctr= ckpt["patience_ctr"]
        history     = ckpt["history"]
        model.load_state_dict(ckpt["model_state"])
        # Rebuild correct optimizer & scheduler for the saved phase BEFORE
        # loading state_dicts (param group count must match).
        if phase == 2 or start_epoch > WARMUP:
            phase = 2; unfreeze_all(model)
            opt = _make_optimizer(model, phase=2)
        else:
            freeze_backbone(model)
            opt = _make_optimizer(model, phase=1)
        opt.load_state_dict(ckpt["opt_state"])
        # Move Adam buffers to DEVICE (they are saved on CPU by default)
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, CFG["epochs"] - WARMUP), eta_min=CFG["lr_min"]
        )
        sch.load_state_dict(ckpt["sch_state"])
        log.info(f"  [{label}] Resumed from epoch {start_epoch} "
                 f"(phase {phase}, best_val={best_val:.4f})")
    else:
        # ── Fresh start ────────────────────────────────────────────────────────
        start_epoch  = 1; phase = 1
        best_val     = 0.0; patience_ctr = 0
        history      = {"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[],"epoch_time":[]}
        freeze_backbone(model)
        opt = _make_optimizer(model, phase=1)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, CFG["epochs"] - WARMUP), eta_min=CFG["lr_min"]
        )

    # ── Feature cache for warmup (only needed if warmup epochs remain) ────────
    feat_train = feat_val = None
    if start_epoch <= WARMUP:
        log.info(f"  [{label}] Pre-caching backbone features for warmup epochs "
                 f"{start_epoch}–{WARMUP} ...")
        t_cache = time.time()
        feat_train = _cache_backbone_features(model.backbone, train_ldr, is_train=True)
        feat_val   = _cache_backbone_features(model.backbone, val_ldr,   is_train=False)
        log.info(f"  [{label}] Feature cache ready in {time.time()-t_cache:.1f}s")

    total_p, train_p = model.param_counts()
    log.info(f"  [{label}] total={total_p:,} trainable={train_p:,} "
             f"start_epoch={start_epoch} phase={phase}")

    t_start = time.time()

    # ── Training loop ──────────────────────────────────────────────────────
    epoch_range = range(start_epoch, CFG["epochs"] + 1)
    epoch_bar   = tqdm(epoch_range, desc=f"[{label}]", unit="ep",
                       dynamic_ncols=True, colour="magenta")
    for epoch in epoch_bar:
        t0 = time.time()

        if epoch == WARMUP + 1 and phase == 1:
            # ── Transition: warmup → full fine-tune ────────────────────────────
            unfreeze_all(model)
            opt = _make_optimizer(model, phase=2)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, CFG["epochs"] - WARMUP), eta_min=CFG["lr_min"]
            )
            phase = 2
            if feat_train is not None:
                del feat_train, feat_val; feat_train = feat_val = None
                gc.collect()
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
            _, train_p2 = model.param_counts()
            log.info(f"  [{label}] E{epoch}: backbone unfrozen "
                     f"(LR bb={CFG['lr_backbone']:.0e} other={CFG['lr']:.0e}) "
                     f"trainable={train_p2:,}")

        if epoch <= WARMUP and feat_train is not None:
            tr_loss, tr_acc       = train_one_epoch(model, feat_train, opt, use_cached=True)
            vl_loss, vl_acc, _, _ = evaluate(model, feat_val, use_cached=True)
        else:
            tr_loss, tr_acc       = train_one_epoch(model, train_ldr, opt, use_cached=False)
            vl_loss, vl_acc, _, _ = evaluate(model, val_ldr, use_cached=False)

        sch.step()
        ep_time = time.time() - t0
        for k, v in [("train_loss",tr_loss),("val_loss",vl_loss),
                     ("train_acc",tr_acc),("val_acc",vl_acc),("epoch_time",round(ep_time,2))]:
            history[k].append(v)
        log.info(f"  [{label}] E{epoch:>3}/{CFG['epochs']} Tr={tr_acc:.4f} L={tr_loss:.4f} "
                 f"Va={vl_acc:.4f} {ep_time:.0f}s")

        if vl_acc > best_val:
            best_val = vl_acc; patience_ctr = 0
            torch.save(model.state_dict(), out_dir/ckpt_name)
            log.info(f"    ✓ Best model saved (val={vl_acc:.4f})")
        else:
            patience_ctr += 1

        # Update epoch bar with live summary metrics
        epoch_bar.set_postfix(
            ph=phase,
            tr=f"{tr_acc:.3f}", va=f"{vl_acc:.3f}",
            best=f"{best_val:.3f}", pat=f"{patience_ctr}/{CFG['patience']}"
        )

        # ── Save resume checkpoint (overwrite each epoch) ───────────────────────
        _save_resume_ckpt(resume_path, epoch, phase, model, opt, sch,
                          history, best_val, patience_ctr)

        if patience_ctr >= CFG["patience"]:
            log.info(f"  Early stop E{epoch} best={best_val:.4f}"); break

    # Clean up the resume checkpoint on successful completion
    if resume_path.exists():
        resume_path.unlink()
        log.info(f"  [{label}] Resume checkpoint removed (training complete)")

    history["total_time"] = round(time.time()-t_start, 2)
    return history, best_val

# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(true_labels, probs, preds):
    bin_labels = label_binarize(true_labels, classes=list(range(NUM_CLASSES)))
    report = classification_report(true_labels, preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(true_labels, preds)
    per_class = {}
    for i,cls in enumerate(CLASSES):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp; tn=cm.sum()-tp-fp-fn
        fpr=fp/(fp+tn) if (fp+tn)>0 else 0.0; fnr=fn/(fn+tp) if (fn+tp)>0 else 0.0
        per_class[cls]={"TP":int(tp),"FP":int(fp),"FN":int(fn),"TN":int(tn),
            "FPR":round(fpr,4),"FNR":round(fnr,4),
            "Sensitivity":round(report[cls]["recall"],4),"Specificity":round(1-fpr,4),
            "Precision":round(report[cls]["precision"],4),"F1":round(report[cls]["f1-score"],4)}
    auc_per={cls:round(roc_auc_score(bin_labels[:,i],probs[:,i]),4) for i,cls in enumerate(CLASSES)}
    macro_auc=round(float(roc_auc_score(bin_labels,probs,multi_class="ovr",average="macro")),4)
    ap_per={cls:round(average_precision_score(bin_labels[:,i],probs[:,i]),4) for i,cls in enumerate(CLASSES)}
    return {"overall_accuracy":round(float((preds==true_labels).mean()),4),"macro_auc":macro_auc,
            "per_class_metrics":per_class,"auc_per_class":auc_per,"ap_per_class":ap_per,
            "confusion_matrix":cm.tolist(),"classification_report":report}

# ── McNemar ──────────────────────────────────────────────────────────────────
def mcnemar_test(preds_a, preds_b, true_labels, label_a="A", label_b="B"):
    ca=(preds_a==true_labels); cb=(preds_b==true_labels)
    n01=int((ca&~cb).sum()); n10=int((~ca&cb).sum())
    n00=int((ca&cb).sum()); n11=int((~ca&~cb).sum())
    denom=n01+n10
    if denom==0: chi2,p=0.0,1.0
    else:
        chi2=(abs(n01-n10)-1.0)**2/denom; p=float(1.0-chi2_dist.cdf(chi2,df=1))
    if p<0.05:
        note=(f"{label_a} significantly BETTER (p={p:.4f})" if n01>n10
              else f"{label_a} significantly WORSE (p={p:.4f})")
    else: note=f"No significant difference (p={p:.4f})"
    return {"n00":n00,"n01":n01,"n10":n10,"n11":n11,"chi2":round(chi2,4),"p_value":round(p,6),
            "significant":p<0.05,"a_better":(p<0.05 and n01>n10),"note":note}

# ── Plots ────────────────────────────────────────────────────────────────────
def plot_training_curves(hq, hc, name, out_dir):
    fig,axes=plt.subplots(1,3,figsize=(20,5)); fig.suptitle(f"{name} - Training Curves",fontsize=13)
    eq=range(1,len(hq["train_acc"])+1); ec=range(1,len(hc["train_acc"])+1)
    for ax,key,title in zip(axes[:2],["train_acc","train_loss"],["Accuracy","Loss"]):
        vk=key.replace("train_","val_")
        ax.plot(eq,hq[key],color=COLOR_QH,lw=2,label="QH train")
        ax.plot(eq,hq[vk],color=COLOR_QH,lw=2,ls="--",label="QH val")
        ax.plot(ec,hc[key],color=COLOR_CL,lw=2,label="CL train")
        ax.plot(ec,hc[vk],color=COLOR_CL,lw=2,ls="--",label="CL val")
        ax.set_title(title);ax.set_xlabel("Epoch");ax.legend(fontsize=8);ax.grid(True,alpha=0.3)
    axes[2].plot(eq,hq["epoch_time"],color=COLOR_QH,lw=2,marker="o",ms=4,label=f"QH tot={hq['total_time']:.0f}s")
    axes[2].plot(ec,hc["epoch_time"],color=COLOR_CL,lw=2,marker="s",ms=4,label=f"CL tot={hc['total_time']:.0f}s")
    axes[2].set_title("Epoch Time (s)");axes[2].legend(fontsize=8);axes[2].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir/"training_curves_qh_vs_cl.svg",format="svg",bbox_inches="tight"); plt.close()
    log.info("  Saved: training_curves_qh_vs_cl.svg")

def plot_roc_overlay(true_labels, probs_qh, probs_cl, name, out_dir):
    bl=label_binarize(true_labels,classes=list(range(NUM_CLASSES)))
    fig,axes=plt.subplots(1,NUM_CLASSES,figsize=(7*NUM_CLASSES,6))
    auc_q,auc_c=[],[]
    for i,cls in enumerate(CLASSES):
        ax=axes[i]
        fq,tq,_=roc_curve(bl[:,i],probs_qh[:,i]); aq=roc_auc_score(bl[:,i],probs_qh[:,i])
        fc,tc,_=roc_curve(bl[:,i],probs_cl[:,i]); ac=roc_auc_score(bl[:,i],probs_cl[:,i])
        auc_q.append(aq); auc_c.append(ac)
        ax.plot(fq,tq,color=COLOR_QH,lw=2.5,label=f"QH {aq:.3f}")
        ax.plot(fc,tc,color=COLOR_CL,lw=2.5,ls="--",label=f"CL {ac:.3f}")
        ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
        d=aq-ac; col="#27ae60" if d>0 else "#e74c3c"
        ax.text(0.55,0.08,f"dAUC={d:+.4f}",transform=ax.transAxes,fontsize=11,fontweight="bold",color=col,
                bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor=col,alpha=0.9))
        ax.set_title(cls.replace("_","\n"),fontsize=11);ax.set_xlabel("FPR");ax.set_ylabel("TPR")
        ax.legend(fontsize=9);ax.grid(True,alpha=0.25)
    mq=round(float(np.mean(auc_q)),4); mc=round(float(np.mean(auc_c)),4)
    fig.suptitle(f"{name}  ROC  QH={mq:.4f}  CL={mc:.4f}  d={mq-mc:+.4f}",fontsize=13,fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir/"roc_overlay_qh_vs_cl.png",dpi=150,bbox_inches="tight"); plt.close()
    log.info("  Saved: roc_overlay_qh_vs_cl.png"); return mq,mc

def plot_confusion_matrices(cm_qh, cm_cl, name, out_dir):
    fig,axes=plt.subplots(2,2,figsize=(16,12)); fig.suptitle(f"{name} - Confusion Matrices",fontsize=13)
    pairs=[(cm_qh,"QH - Counts","Purples",axes[0,0]),(cm_qh,"QH - Norm","Purples",axes[0,1]),
           (cm_cl,"CL - Counts","Blues",axes[1,0]),(cm_cl,"CL - Norm","Blues",axes[1,1])]
    for idx,(cm,title,cmap,ax) in enumerate(pairs):
        if idx%2==0: sns.heatmap(cm,annot=True,fmt="d",cmap=cmap,xticklabels=CLASSES,yticklabels=CLASSES,ax=ax)
        else:
            cmn=cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-8)
            sns.heatmap(cmn,annot=True,fmt=".2f",cmap=cmap,xticklabels=CLASSES,yticklabels=CLASSES,ax=ax)
        ax.set_title(title,fontsize=10);ax.set_xlabel("Predicted");ax.set_ylabel("Actual");ax.tick_params(axis="x",rotation=30)
    plt.tight_layout(); plt.savefig(out_dir/"confusion_matrix_qh_vs_cl.png",dpi=150,bbox_inches="tight"); plt.close()
    log.info("  Saved: confusion_matrix_qh_vs_cl.png")

def plot_metrics_delta(mq, mc, name, out_dir):
    aq=mq["overall_accuracy"]; ac=mc["overall_accuracy"]
    fq=mq["classification_report"]["weighted avg"]["f1-score"]; fc=mc["classification_report"]["weighted avg"]["f1-score"]
    labels=["Accuracy","F1 (weighted)","Macro AUC"]; deltas=[aq-ac,fq-fc,mq["macro_auc"]-mc["macro_auc"]]
    colors=["#27ae60" if d>0 else "#e74c3c" for d in deltas]
    fig,ax=plt.subplots(figsize=(8,5))
    bars=ax.bar(labels,deltas,color=colors,alpha=0.85,edgecolor="white",linewidth=1.5)
    ax.axhline(0,color="black",linewidth=1)
    for bar,val in zip(bars,deltas):
        ax.text(bar.get_x()+bar.get_width()/2,val+(0.001 if val>=0 else -0.002),f"{val:+.4f}",
                ha="center",va="bottom" if val>=0 else "top",fontsize=11,fontweight="bold",
                color="#27ae60" if val>0 else "#e74c3c")
    ax.set_ylabel("QH - CL"); ax.set_title(f"{name} - Delta QH vs CL")
    ax.grid(True,alpha=0.25,axis="y"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out_dir/"metrics_delta_qh_vs_cl.png",dpi=150,bbox_inches="tight"); plt.close()
    log.info("  Saved: metrics_delta_qh_vs_cl.png")

def plot_per_model_auc_comparison(all_results, out_dir):
    """
    Neutral AUC bar chart — no model receives special colour coding,
    no ★ annotations, no cherry-picking labels. Best-CL reference line
    is shown as a neutral dashed line for readability only.
    """
    qh_rows=[r for r in all_results if r["Type"]=="QH"]
    cl_rows=[r for r in all_results if r["Type"]=="CL"]
    models=[r["Model"] for r in qh_rows]; qh_aucs=[r["AUC"] for r in qh_rows]
    cl_local={r["Model"]:r["AUC"] for r in cl_rows}; ext_cl={}
    if CFG["classical_results_csv"].exists():
        cdf=pd.read_csv(CFG["classical_results_csv"])
        for _,row in cdf.iterrows(): ext_cl[row["Model"]]=float(row["AUC"])
    all_cl=list(cl_local.values())+list(ext_cl.values()); best_cl=max(all_cl) if all_cl else None
    x=np.arange(len(models)); w=0.35
    fig,ax=plt.subplots(figsize=(max(10,len(models)*1.8),7))
    # Uniform colour for all QH bars — no special treatment based on AUC level
    bq=ax.bar(x-w/2,qh_aucs,w,color=COLOR_QH,alpha=0.88,label="QH")
    cl_vals=[cl_local.get(m,0.0) for m in models]
    bc=ax.bar(x+w/2,cl_vals,w,color=COLOR_CL,alpha=0.88,label="CL")
    for bar,val in zip(bq,qh_aucs):
        ax.text(bar.get_x()+bar.get_width()/2,val+0.003,f"{val:.4f}",ha="center",va="bottom",fontsize=8,fontweight="bold")
    for bar,val in zip(bc,cl_vals):
        ax.text(bar.get_x()+bar.get_width()/2,val+0.003,f"{val:.4f}",ha="center",va="bottom",fontsize=8,fontweight="bold",color=COLOR_CL)
    if best_cl:
        # Reference line only — no winner annotations
        ax.axhline(best_cl,color="grey",lw=1.5,ls="--",label=f"Best CL={best_cl:.4f}")
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_","\n") for m in models],fontsize=9)
    ax.set_ylim(0.5,1.05); ax.set_ylabel("Macro AUC-ROC",fontsize=11)
    ax.set_title("AUC-ROC: QH vs CL — All models, unfiltered",fontsize=12)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.2,axis="y"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out_dir/"auc_comparison_chart.png",dpi=150,bbox_inches="tight"); plt.close()
    log.info("  Saved: auc_comparison_chart.png"); return best_cl

def plot_mcnemar_heatmap(records, out_dir):
    if not records: return
    df=pd.DataFrame(records); models=df["Model"].unique().tolist(); comps=df["Comparison"].unique().tolist()
    pm=np.ones((len(models),len(comps))); om=np.zeros_like(pm)
    for i,m in enumerate(models):
        for j,c in enumerate(comps):
            row=df[(df["Model"]==m)&(df["Comparison"]==c)]
            if len(row):
                p=float(row["p_value"].values[0]); ab=bool(row["a_better"].values[0]); sig=bool(row["significant"].values[0])
                pm[i,j]=p; om[i,j]=(1 if (sig and ab) else (-1 if (sig and not ab) else 0))
    fig,axes=plt.subplots(1,2,figsize=(12,max(4,len(models)*0.9+2)))
    sns.heatmap(pm,annot=True,fmt=".4f",cmap="RdYlGn_r",xticklabels=comps,yticklabels=models,vmin=0,vmax=0.1,ax=axes[0],linewidths=0.5)
    axes[0].set_title("p-values"); axes[0].tick_params(axis="x",rotation=15)
    cmap2=mcolors.ListedColormap(["#e74c3c","#f0f0f0","#27ae60"])
    sns.heatmap(om,cmap=cmap2,vmin=-1,vmax=1,xticklabels=comps,yticklabels=models,ax=axes[1],linewidths=0.5,annot=False)
    axes[1].set_title("green=QH+ red=QH- grey=NS")
    for i in range(len(models)):
        for j in range(len(comps)):
            sym="+" if om[i,j]>0 else ("-" if om[i,j]<0 else "="); col="white" if om[i,j]!=0 else "#555"
            axes[1].text(j+0.5,i+0.5,sym,ha="center",va="center",color=col,fontsize=14,fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir/"mcnemar_heatmap.png",dpi=150,bbox_inches="tight"); plt.close()
    log.info("  Saved: mcnemar_heatmap.png")

def save_metrics_report(metrics, model_name, variant, out_dir):
    sep="="*62
    lines=[sep,f"CATARACT - {variant}  {model_name}",f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",sep,"",
           f"Accuracy : {metrics['overall_accuracy']:.4f}",f"Macro AUC: {metrics['macro_auc']:.4f}","","Per-class AUC:"]
    for cls,auc in metrics["auc_per_class"].items(): lines.append(f"  {cls:<25} {auc:.4f}")
    lines+=["","Per-class AP:"]
    for cls,ap in metrics["ap_per_class"].items(): lines.append(f"  {cls:<25} {ap:.4f}")
    lines+=[""]+[sep,"Detailed Metrics",sep]
    for cls,m in metrics["per_class_metrics"].items():
        lines.append(f"\n  {cls}")
        for k,v in m.items(): lines.append(f"    {k:<35} {v}")
    txt="\n".join(lines); print(txt)
    (out_dir/f"report_{variant.lower()}.txt").write_text(txt,encoding="utf-8")
    with open(out_dir/f"metrics_{variant.lower()}.json","w") as f: json.dump(metrics,f,indent=2)

def write_auc_proof_report(all_results, mcnemar_records, best_cl_auc, out_dir):
    """
    Unbiased comparison report.
    ALL models (QH and CL) are reported — no AUC-threshold filtering.
    The verdict is based on the best observed QH vs the best observed CL,
    both selected from the same pool without exclusions.
    """
    qh_rows=[r for r in all_results if r["Type"]=="QH"]
    cl_rows=[r for r in all_results if r["Type"]=="CL"]
    if not qh_rows:
        log.error("No QH results to evaluate."); return False
    if not cl_rows:
        log.error("No CL results to evaluate."); return False
    best_qh=max(qh_rows, key=lambda r:r["AUC"])
    best_cl_local=max(cl_rows, key=lambda r:r["AUC"])
    sep="="*70
    lines=[sep,"QUANTUM HYBRID vs CLASSICAL — AUC COMPARISON REPORT (v8)",
           f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
           f"Circuit   : {CFG['n_qubits']}q x {CFG['n_layers']}L  ({CFG['n_layers']*CFG['n_qubits']*3} quantum params)",
           f"Training  : batch={CFG['batch_size']} lr={CFG['lr']} lr_backbone={CFG['lr_backbone']} "
           f"AdamW warmup={CFG['warmup_epochs']}ep epochs={CFG['epochs']}ep patience={CFG['patience']}",
           f"Clip      : 1.0 (identical for QH and CL)",
           sep,"",
           "CLAIM: Best QH Macro AUC > Best CL Macro AUC",
           "  (all models included — no threshold filtering)",""]
    lines+=["-"*70,"QH RESULTS — all models, sorted by AUC:","-"*70]
    for r in sorted(qh_rows, key=lambda x:x["AUC"], reverse=True):
        tag="  <- BEST QH" if r["Model"]==best_qh["Model"] else ""
        lines.append(f"  {r['Model']:<28} AUC={r['AUC']:.4f} Acc={r['Accuracy']:.4f} F1={r['F1']:.4f}{tag}")
    lines+=[""]+["-"*70,"CL RESULTS — all models, sorted by AUC:","-"*70]
    for r in sorted(cl_rows, key=lambda x:x["AUC"], reverse=True):
        tag="  <- BEST CL" if r["Model"]==best_cl_local["Model"] else ""
        lines.append(f"  {r['Model']:<28} AUC={r['AUC']:.4f} Acc={r['Accuracy']:.4f} F1={r['F1']:.4f}{tag}")
    ext_best=None
    if CFG["classical_results_csv"].exists():
        cdf=pd.read_csv(CFG["classical_results_csv"])
        lines+=[""]+["-"*70,"EXTERNAL CL (multimodel1_output — fully classical pipeline):","-"*70]
        for _,row in cdf.sort_values("AUC",ascending=False).iterrows():
            lines.append(f"  {row['Model']:<28} AUC={row['AUC']:.4f}")
        ext_best=(cdf.loc[cdf["AUC"].idxmax(),"Model"],float(cdf["AUC"].max()))
    lines+=[""]+[sep,"VERDICT",sep,""]
    lines.append(f"  Best QH (this run) : {best_qh['Model']:<28} AUC={best_qh['AUC']:.4f}")
    lines.append(f"  Best CL (this run) : {best_cl_local['Model']:<28} AUC={best_cl_local['AUC']:.4f}")
    if ext_best: lines.append(f"  Best CL (external) : {ext_best[0]:<28} AUC={ext_best[1]:.4f}")
    if best_cl_auc: lines.append(f"  Best CL (overall)  : AUC={best_cl_auc:.4f}")
    lines.append("")
    claim=best_qh["AUC"]>(best_cl_auc or 0)
    if claim:
        lines+=["  CLAIM SUPPORTED ✓",
                f"    QH ({best_qh['Model']}) AUC={best_qh['AUC']:.4f} > best CL AUC={best_cl_auc:.4f}",
                f"    delta=+{best_qh['AUC']-best_cl_auc:.4f}"]
    else:
        lines+=["  CLAIM NOT SUPPORTED ✗",
                f"    Best QH AUC={best_qh['AUC']:.4f}  Best CL AUC={best_cl_auc:.4f}",
                f"    delta={best_qh['AUC']-(best_cl_auc or 0):.4f}"]
    if mcnemar_records:
        lines+=[""]+[sep,"McNEMAR'S TEST (H0: identical error rates | p<0.05 = significant)",sep,""]
        for rec in mcnemar_records:
            sig="SIGNIFICANT" if rec["significant"] else "NS"
            lines.append(f"  {rec['Model']:<28} chi2={rec['chi2']:.3f} p={rec['p_value']:.4f} {sig}")
            lines.append(f"    {rec['note']}")
    lines+=[""]; report="\n".join(lines); print(report)
    (out_dir/"AUC_COMPARISON_REPORT.txt").write_text(report,encoding="utf-8")
    log.info("  Saved: AUC_COMPARISON_REPORT.txt"); return claim

# ── Per-model pipeline ───────────────────────────────────────────────────────
def run_model(model_name, n_qubits, n_layers, eval_only):
    out_dir=CFG["output_dir"]/model_name; out_dir.mkdir(parents=True,exist_ok=True)
    image_size=CFG["image_sizes"].get(model_name,CFG["image_sizes"]["__default__"])
    train_ldr,val_ldr,test_ldr,_,y_te=get_loaders(image_size)
    true_labels=y_te.numpy(); timing_list=[]; mcnemar_list=[]

    # QH
    log.info(f"\n  -- QH: {model_name} --")
    qh_model=QuantumHybridModel(model_name,n_qubits,n_layers,image_size,
                               n_q_blocks=CFG.get("n_q_blocks",1)).to(DEVICE)
    if DEVICE.type=="cuda": log.info(f"  [QH] model on {torch.cuda.get_device_name(0)}")
    qh_ckpt=out_dir/"best_model_qh.pth"
    hist_qh={"train_acc":[],"val_acc":[],"train_loss":[],"val_loss":[],"epoch_time":[],"total_time":0}
    best_val_qh=None
    if not eval_only:
        hist_qh,best_val_qh=train_model(qh_model,"QH",train_ldr,val_ldr,out_dir,"best_model_qh.pth")
        timing_list.append({"Model":model_name,"Type":"QH","TotalTime":hist_qh["total_time"]})
        with open(out_dir/"timing_qh.json","w") as f: json.dump({"model":model_name,"type":"QH",**hist_qh},f,indent=2)
    if qh_ckpt.exists():
        qh_model.load_state_dict(torch.load(qh_ckpt,map_location=DEVICE,weights_only=False))
        log.info("  Loaded QH checkpoint")
    else: log.warning("  No QH checkpoint")
    _,_,qh_probs,_=evaluate(qh_model,test_ldr)
    qh_preds=qh_probs.argmax(axis=1); metrics_qh=compute_metrics(true_labels,qh_probs,qh_preds)
    save_metrics_report(metrics_qh,model_name,"QH",out_dir)
    np.save(out_dir/"qh_probs.npy",qh_probs); np.save(out_dir/"qh_preds.npy",qh_preds)
    log.info(f"  QH {model_name}: AUC={metrics_qh['macro_auc']:.4f} Acc={metrics_qh['overall_accuracy']:.4f}")
    qh_result={"Model":model_name,"Type":"QH","Accuracy":metrics_qh["overall_accuracy"],
               "Precision":round(metrics_qh["classification_report"]["weighted avg"]["precision"],4),
               "Recall":round(metrics_qh["classification_report"]["weighted avg"]["recall"],4),
               "F1":round(metrics_qh["classification_report"]["weighted avg"]["f1-score"],4),
               "AUC":metrics_qh["macro_auc"],"Best Val Acc":round(best_val_qh,4) if best_val_qh else "N/A",
               "Circuit":f"{n_qubits}q x {n_layers}L"}
    del qh_model; gc.collect()

    # CL
    log.info(f"\n  -- CL: {model_name} --")
    cl_model=ClassicalBaselineModel(model_name,image_size).to(DEVICE)
    if DEVICE.type=="cuda": log.info(f"  [CL] model on {torch.cuda.get_device_name(0)}")
    cl_ckpt=out_dir/"best_model_cl.pth"
    hist_cl={"train_acc":[],"val_acc":[],"train_loss":[],"val_loss":[],"epoch_time":[],"total_time":0}
    best_val_cl=None
    if not eval_only:
        hist_cl,best_val_cl=train_model(cl_model,"CL",train_ldr,val_ldr,out_dir,"best_model_cl.pth")
        timing_list.append({"Model":model_name,"Type":"CL","TotalTime":hist_cl["total_time"]})
        with open(out_dir/"timing_cl.json","w") as f: json.dump({"model":model_name,"type":"CL",**hist_cl},f,indent=2)
    if cl_ckpt.exists():
        cl_model.load_state_dict(torch.load(cl_ckpt,map_location=DEVICE,weights_only=False))
        log.info("  Loaded CL checkpoint")
    _,_,cl_probs,_=evaluate(cl_model,test_ldr)
    cl_preds=cl_probs.argmax(axis=1); metrics_cl=compute_metrics(true_labels,cl_probs,cl_preds)
    save_metrics_report(metrics_cl,model_name,"CL",out_dir)
    np.save(out_dir/"cl_probs.npy",cl_probs); np.save(out_dir/"cl_preds.npy",cl_preds)
    cl_result={"Model":model_name,"Type":"CL","Accuracy":metrics_cl["overall_accuracy"],
               "Precision":round(metrics_cl["classification_report"]["weighted avg"]["precision"],4),
               "Recall":round(metrics_cl["classification_report"]["weighted avg"]["recall"],4),
               "F1":round(metrics_cl["classification_report"]["weighted avg"]["f1-score"],4),
               "AUC":metrics_cl["macro_auc"],"Best Val Acc":round(best_val_cl,4) if best_val_cl else "N/A",
               "Circuit":"MLP head (dim->256->3)"}
    del cl_model; gc.collect()

    qhp=np.load(out_dir/"qh_probs.npy"); clp=np.load(out_dir/"cl_probs.npy")
    qhd=np.load(out_dir/"qh_preds.npy"); cld=np.load(out_dir/"cl_preds.npy")
    if not eval_only: plot_training_curves(hist_qh,hist_cl,model_name,out_dir)
    plot_roc_overlay(true_labels,qhp,clp,model_name,out_dir)
    plot_confusion_matrices(np.array(metrics_qh["confusion_matrix"]),np.array(metrics_cl["confusion_matrix"]),model_name,out_dir)
    plot_metrics_delta(metrics_qh,metrics_cl,model_name,out_dir)
    mn=mcnemar_test(qhd,cld,true_labels,label_a="QH",label_b="CL")
    mn["Model"]=model_name; mn["Comparison"]="QH vs CL"; mcnemar_list.append(mn)
    log.info(f"  McNemar: chi2={mn['chi2']:.3f} p={mn['p_value']:.4f} {mn['note']}")
    clear_cache_for_image_size(image_size)
    log.info(f"  Outputs: {out_dir}")
    return qh_result,cl_result,mcnemar_list,timing_list

# ── Entry point ──────────────────────────────────────────────────────────────
def main(eval_only=False, model_filter=None, n_qubits=None, n_layers=None):
    if n_qubits: CFG["n_qubits"]=n_qubits
    if n_layers: CFG["n_layers"]=n_layers
    nq,nl=CFG["n_qubits"],CFG["n_layers"]
    log.info(f"Device={DEVICE} qubits={nq} layers={nl} q-params={nl*nq*3} batch={CFG['batch_size']}")
    log.info(f"lr={CFG['lr']}")
    CFG["output_dir"].mkdir(parents=True,exist_ok=True)
    models_to_run=model_filter or CFG["models"]
    all_results=[]; all_mcnemar=[]; all_timing=[]

    for model_name in tqdm(models_to_run, desc="Models", unit="model",
                           dynamic_ncols=True, colour="blue"):
        log.info(f"\n{'='*65}\n  MODEL: {model_name.upper()}\n{'='*65}")
        try:
            qh_res,cl_res,mn_list,tm_list=run_model(model_name,nq,nl,eval_only)
            all_results.extend([qh_res,cl_res]); all_mcnemar.extend(mn_list); all_timing.extend(tm_list)
        except Exception as e:
            log.error(f"  FAILED {model_name}: {e}")
            import traceback; traceback.print_exc(); gc.collect()

    if not all_results: log.error("No results."); return
    out=CFG["output_dir"]
    df=pd.DataFrame(all_results).sort_values(["Type","AUC"],ascending=[True,False])
    df.to_csv(out/"all_results_qh_vs_cl.csv",index=False)
    best_cl=plot_per_model_auc_comparison(all_results,out)
    if all_mcnemar:
        pd.DataFrame(all_mcnemar).to_csv(out/"mcnemar_results.csv",index=False)
        plot_mcnemar_heatmap(all_mcnemar,out)
    claim=write_auc_proof_report(all_results,all_mcnemar,best_cl,out)
    log.info("\n"+df[["Model","Type","Accuracy","F1","AUC","Best Val Acc"]].to_string(index=False))
    log.info(f"\nCLAIM: {'SUPPORTED' if claim else 'NOT YET SUPPORTED'}")
    log.info(f"Outputs: {out}")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="QH vs CL Cataract (v8 — scientifically fair)")
    parser.add_argument("--eval-only",action="store_true")
    parser.add_argument("--models",nargs="+",default=None)
    parser.add_argument("--qubits",type=int,default=None)
    parser.add_argument("--layers",type=int,default=None)
    args=parser.parse_args()
    main(eval_only=args.eval_only,model_filter=args.models,n_qubits=args.qubits,n_layers=args.layers)
