"""
eureka_benchmark.py
====================
EUREKA (Edge Utility for REwiring and Knowledge-Aware graph representation)
benchmark, built on the same harness as benchmark_rewiring_v15.py
(CVGAE-HetHom) — data loading, graph-quality metrics, matched-density
subsampling, semi-supervised label masking, downstream classifiers,
resumable JSONL results, progress tracking, tables and plots are all in the
same style and (where model-agnostic) the same code.

Only the representation/rewiring model changed, and it is reproduced
VERBATIM from sweep_eureka.py: FUSE(A) (modularity power iteration), the
orthogonal residual-feature branch, the four-view edge-utility scorer f_theta,
the adjacency reweighting A*_uv = A_uv * g(q_uv), and the re-embedding step
S*/H*/E* are all copied unchanged. Nothing about how EUREKA computes its
representation or rewires the graph has been modified here.

--------------------------------------------------------------------------------
WHAT EUREKA DOES  (see sweep_eureka.py for the full write-up)
--------------------------------------------------------------------------------
Given a graph G=(V,E) with adjacency A and node features X:

  1. Structural representation via a FUSE procedure:      S  = FUSE(A)
  2. Complementary (residual) feature representation:     H  = (I - S Sᵀ) X
  3. Combined representation:                             E_rep = [ S | α H ]

  4. A learnable per-edge *utility* q_uv = f_θ(z_uv), where the edge descriptor
     combines four views of the edge (u,v):
         z_uv = [ z_struct , z_res , z_(1) , z_(2) ]
     (the neighbourhood-distribution views are the DHGR-inspired signals).

  5. The adjacency is reweighted by the utility:          A*_uv = A_uv · g(q_uv)
     g(·)=sigmoid maps utility to a retention weight; low-utility edges are
     pruned.  The objective is NOT to maximise homophily — a heterophilic edge
     is retained whenever it is consistent with the representation objective.

  6. The rewired graph is re-embedded:
         S* = FUSE(A*) ,  H* = (I - S* S*ᵀ) X ,  E* = [ S* | α H* ]

E* plays the role the CVGAE embedding Z_lat used to play, and A* plays the
role of the rewired graph.

--------------------------------------------------------------------------------
WHAT THIS FILE EVALUATES
--------------------------------------------------------------------------------
Exactly two graph/feature configurations, each run with several downstream
classifiers, over seeds {0, 1, 2}, on the full dataset suite (real +
synthetic):

  1. "original"                 — original graph      + original node features
  2. "eureka_rewired_learnedE"  — EUREKA-rewired graph + learned representation E*

For every (dataset, method, classifier, seed) we record accuracy, macro-F1 and
runtime (rewiring + classifier training).  Results are appended to a JSONL
file after every single run, so an interrupted sweep resumes exactly where it
left off.  Per-dataset tables, an "average over datasets per classifier" table
and all figures are produced at the end (and refreshed after every dataset).

Usage examples
--------------
  python eureka_benchmark.py
  python eureka_benchmark.py --resume --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

warnings.filterwarnings("ignore")

# ── Unified-benchmark additions ──────────────────────────────────────────────
import os as _os
import random as _random
import sys as _sys

# Make the shared ``common`` package importable no matter where this script is
# launched from (it lives next to this file).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from common import metrics as _metrics          # noqa: E402
from common import progress as _progress        # noqa: E402
from common import checkpoint as _checkpoint     # noqa: E402
from common import reporting as _reporting       # noqa: E402
from common import plotting as _plotting         # noqa: E402


def set_global_seed(seed: int):
    """Seed every RNG we can reach and put cuDNN in deterministic mode."""
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    _os.environ["PYTHONHASHSEED"] = str(seed)

# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR   = Path("eureka_benchmark_results")
PLOTS_DIR = OUT_DIR / "plots"
CKPT_DIR  = OUT_DIR / "checkpoints"
for _d in [OUT_DIR, PLOTS_DIR, CKPT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

QUALITY_CSV  = OUT_DIR / "graph_quality.csv"
RESULTS_FILE = OUT_DIR / "results.jsonl"
SUMMARY_CSV  = OUT_DIR / "summary.csv"


# =============================================================================
#  SECTION 1 — Data loading  (fixed for Actor / Squirrel-F / Chameleon-F)
# =============================================================================

def _safe_extract_masks(d):
    """
    Robustly extract train/val/test masks from a PyG Data object.

    Handles three mask formats found across PyG versions:
      1. 2-D boolean mask  (N, K)  — take column 0
      2. 1-D boolean mask  (N,)
      3. 1-D integer index tensor — convert to boolean
    """
    def _to_bool(m, N):
        if m is None:
            return torch.zeros(N, dtype=torch.bool)
        if m.dtype == torch.bool:
            if m.dim() == 2:
                return m[:, 0]
            return m
        # integer indices
        mask = torch.zeros(N, dtype=torch.bool)
        mask[m] = True
        return mask

    N = d.num_nodes
    return (
        _to_bool(d.train_mask, N),
        _to_bool(d.val_mask,   N),
        _to_bool(d.test_mask,  N),
    )


def _make_ns(d, tm, vm, te):
    return SimpleNamespace(
        x=d.x, y=d.y, edge_index=d.edge_index,
        train_mask=tm, val_mask=vm, test_mask=te,
        num_nodes=d.num_nodes,
        num_classes=int(d.y.max().item()) + 1,
    )


def load_real_dataset(name, root="./data"):
    # ------------------------------------------------------------------
    # Actor  (older PyG: Actor dataset; newer PyG: same but masks differ)
    # ------------------------------------------------------------------
    if name == "Actor":
        from torch_geometric.datasets import Actor as _Actor
        ds = _Actor(root=root)
        d  = ds[0]
        tm, vm, te = _safe_extract_masks(d)
        # Actor sometimes has no split at all in very old versions — fallback
        if int(tm.sum()) == 0:
            N   = d.num_nodes
            rng = torch.Generator(); rng.manual_seed(0)
            perm = torch.randperm(N, generator=rng)
            tm = torch.zeros(N, dtype=torch.bool); tm[perm[:int(0.6*N)]] = True
            vm = torch.zeros(N, dtype=torch.bool); vm[perm[int(0.6*N):int(0.8*N)]] = True
            te = torch.zeros(N, dtype=torch.bool); te[perm[int(0.8*N):]] = True
        return _make_ns(d, tm, vm, te)

    # ------------------------------------------------------------------
    # Squirrel-F / Chameleon-F  (filtered Wikipedia networks)
    # Newer PyG renamed the `geom_gcn_preprocess` flag and changed how
    # masks are stored.  We try multiple strategies.
    # ------------------------------------------------------------------
    _wiki_map = {
        "Squirrel-F":  "squirrel",
        "Chameleon-F": "chameleon",
    }
    if name in _wiki_map:
        wiki_name = _wiki_map[name]
        from torch_geometric.datasets import WikipediaNetwork as _Wiki

        # Try three strategies in order; each fully wrapped so that errors
        # in BOTH the constructor AND ds[0] are caught cleanly.
        # Strategy 1: geom_gcn_preprocess=True  (produces per-split masks)
        # Strategy 2: geom_gcn_preprocess=False (raw graph, we build split)
        # Strategy 3: no keyword               (very old PyG)
        _wiki_d = None
        for _kwargs in [
            {"geom_gcn_preprocess": True},
            {"geom_gcn_preprocess": False},
            {},
        ]:
            try:
                _ds = _Wiki(root=root, name=wiki_name, **_kwargs)
                _d  = _ds[0]
                # Validate that y exists and is non-empty before accepting
                if _d.y is None or _d.y.numel() == 0:
                    continue
                _tm, _vm, _te = _safe_extract_masks(_d)
                if int(_tm.sum()) > 0:
                    return _make_ns(_d, _tm, _vm, _te)
                # Masks empty but graph valid — keep as fallback for manual split
                _wiki_d = _d
            except Exception:
                continue

        if _wiki_d is None:
            raise RuntimeError(
                f"Cannot load {name}: all WikipediaNetwork strategies failed. "
                "Try: pip install torch_geometric --upgrade"
            )

        # Graph loaded but all splits were empty — create a standard 60/20/20 split
        N   = _wiki_d.num_nodes
        rng = torch.Generator(); rng.manual_seed(0)
        perm = torch.randperm(N, generator=rng)
        tm = torch.zeros(N, dtype=torch.bool); tm[perm[:int(0.6*N)]] = True
        vm = torch.zeros(N, dtype=torch.bool); vm[perm[int(0.6*N):int(0.8*N)]] = True
        te = torch.zeros(N, dtype=torch.bool); te[perm[int(0.8*N):]] = True
        return _make_ns(_wiki_d, tm, vm, te)

    # ------------------------------------------------------------------
    # HeterophilousGraphDataset (Roman-empire, Amazon-ratings, Tolokers …)
    # These Yandex .npz files contain pickled objects, so np.load must be
    # called with allow_pickle=True.  Older PyG versions don't pass that
    # flag, so we monkey-patch np.load around the call and always restore
    # the original afterwards.
    # ------------------------------------------------------------------
    import numpy as _np_inner
    _orig_np_load = _np_inner.load

    def _pickle_load(*args, **kwargs):
        kwargs.setdefault("allow_pickle", True)
        return _orig_np_load(*args, **kwargs)

    _np_inner.load = _pickle_load
    try:
        from torch_geometric.datasets import HeterophilousGraphDataset
        ds = HeterophilousGraphDataset(root=root, name=name)
        d  = ds[0]
    finally:
        _np_inner.load = _orig_np_load   # always restore, even on error

    tm, vm, te = _safe_extract_masks(d)
    return _make_ns(d, tm, vm, te)


def generate_synthetic_dataset(name, seed=0):
    import random as _random
    rng = np.random.RandomState(seed)
    N = 20_000

    def balanced_labels(n, C):
        y = np.concatenate([
            np.full(n // C + (1 if i < n % C else 0), i) for i in range(C)
        ])
        rng.shuffle(y); return y

    def make_masks(n):
        idx  = rng.permutation(n)
        n_tr = int(0.20 * n); n_va = int(0.20 * n)
        tm = torch.zeros(n, dtype=torch.bool)
        vm = torch.zeros(n, dtype=torch.bool)
        te = torch.zeros(n, dtype=torch.bool)
        tm[idx[:n_tr]] = True
        vm[idx[n_tr:n_tr+n_va]] = True
        te[idx[n_tr+n_va:]] = True
        return tm, vm, te

    def edge_tensor(edges):
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        E = np.unique(np.array(edges, dtype=np.int64), axis=0)
        E = E[E[:, 0] != E[:, 1]]
        rev = E[:, [1, 0]]
        E = np.unique(np.vstack([E, rev]), axis=0)
        return torch.tensor(E.T, dtype=torch.long)

    configs = {"HSBM-MED": (N, 6), "STRUC-HET": (N, 8), "FEAT-HET": (N, 5), "MIXED-SIG": (N, 7)}
    if name not in configs:
        raise ValueError(f"Unknown synthetic dataset: {name}")

    n, C = configs[name]; y = balanced_labels(n, C)
    groups = [np.where(y == c)[0] for c in range(C)]; edges = []

    if name == "HSBM-MED":
        pin, pout = 0.00025, 0.00125
        for c1 in range(C):
            for c2 in range(c1, C):
                p = pin if c1 == c2 else pout
                m = int(p * len(groups[c1]) * len(groups[c2]))
                uu = rng.choice(groups[c1], m); vv = rng.choice(groups[c2], m)
                edges.extend(zip(uu.tolist(), vv.tolist()))
        centers = rng.randn(C, 32) * 1.1; X = np.zeros((n, 128), dtype=np.float32)
        for i in range(n):
            c = y[i]; alt = rng.choice([k for k in range(C) if k != c])
            X[i, :32] = 0.65 * centers[c] + 0.35 * centers[alt] + rng.randn(32) * 1.35
            X[i, 32:64] = rng.randn(32); X[i, 64:] = rng.randn(64) * 0.8

    elif name == "STRUC-HET":
        deg_budget = np.array([42, 36, 30, 24, 8, 7, 6, 5], dtype=float)
        hub_classes = [0,1,2,3]; peri_classes = [4,5,6,7]
        for c in hub_classes:
            target1 = peri_classes[c % 4]; target2 = peri_classes[(c+1) % 4]
            for u in groups[c]:
                reps = int(deg_budget[c] * 0.35)
                for _ in range(reps):
                    tgt = target1 if rng.rand() < 0.7 else target2
                    edges.append((u, rng.choice(groups[tgt])))
                if rng.rand() < 0.08:
                    v = rng.choice(groups[c])
                    if v != u: edges.append((u, v))
        if len(edges) > 320_000:
            edges = _random.sample(edges, 320_000)
        centers = rng.randn(C, 8) * 0.35; X = np.zeros((n, 128), dtype=np.float32)
        for i in range(n):
            c = y[i]; X[i, :8] = centers[c] + rng.randn(8) * 2.3; X[i, 8:] = rng.randn(120)

    elif name == "FEAT-HET":
        pin, pout = 0.00035, 0.00145
        for c1 in range(C):
            for c2 in range(c1, C):
                p = pin if c1 == c2 else pout
                m = int(p * len(groups[c1]) * len(groups[c2]))
                uu = rng.choice(groups[c1], m); vv = rng.choice(groups[c2], m)
                edges.extend(zip(uu.tolist(), vv.tolist()))
        centers = rng.randn(C, 64) * 1.8; X = np.zeros((n, 128), dtype=np.float32)
        for i in range(n):
            c = y[i]; alt = rng.choice([k for k in range(C) if k != c])
            X[i,:64] = 0.78*centers[c] + 0.22*centers[alt] + rng.randn(64)*0.95
            X[i,64:] = rng.randn(64) * 0.55

    elif name == "MIXED-SIG":
        for c in range(C):
            nxt = (c+1)%C; nxt2 = (c+2)%C; prv = (c-1)%C
            for u in groups[c]:
                for _ in range(2): edges.append((u, rng.choice(groups[nxt])))
                for _ in range(1): edges.append((u, rng.choice(groups[nxt2])))
                if rng.rand() < 0.22: edges.append((u, rng.choice(groups[prv])))
                if rng.rand() < 0.12: edges.append((u, rng.choice(groups[c])))
        centers = rng.randn(C, 20) * 1.0; X = np.zeros((n, 128), dtype=np.float32)
        for i in range(n):
            c = y[i]; alt1 = (c+1)%C; alt2 = (c-1)%C
            mix = rng.choice([0,1,2], p=[0.55,0.25,0.20])
            proto = centers[c] if mix==0 else (centers[alt1] if mix==1 else centers[alt2])
            X[i,:20] = proto + rng.randn(20)*1.4; X[i,20:48] = rng.randn(28); X[i,48:] = rng.randn(80)*0.6

    X = X.astype(np.float32); mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True)+1e-8; X=(X-mu)/sd
    edge_index = edge_tensor(edges); train_mask, val_mask, test_mask = make_masks(n)
    return SimpleNamespace(
        x=torch.tensor(X, dtype=torch.float32), y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        num_nodes=n, num_classes=C,
    )


# =============================================================================
#  SECTION 2 — Graph quality metrics
# =============================================================================

def homophily_ratio(edge_index, labels):
    src, dst = edge_index; mask = src != dst; src, dst = src[mask], dst[mask]
    if len(src) == 0: return 0.0
    return float((labels[src] == labels[dst]).float().mean())

def feature_homophily(edge_index, features):
    src, dst = edge_index; mask = src != dst; src, dst = src[mask], dst[mask]
    if len(src) == 0: return 0.0
    Z = F.normalize(features.float(), dim=-1)
    return float((Z[src] * Z[dst]).sum(dim=-1).clamp(-1.0, 1.0).mean())

def adjusted_homophily(edge_index, labels, num_nodes):
    src, dst = edge_index; mask = src != dst; src, dst = src[mask], dst[mask]
    if len(src) == 0: return 0.0
    C = int(labels.max().item()) + 1; E2 = len(src)
    h_edge = float((labels[src] == labels[dst]).float().mean())
    deg = torch.zeros(num_nodes); deg.scatter_add_(0, src, torch.ones(len(src)))
    sum_sq = sum((float(deg[labels == c].sum()) / E2) ** 2 for c in range(C))
    denom = 1.0 - sum_sq
    return 0.0 if abs(denom) < 1e-12 else float((h_edge - sum_sq) / denom)

def edge_purity(edge_index, labels):
    N = labels.shape[0]; src, dst = edge_index; mask = src != dst; src, dst = src[mask], dst[mask]
    same = (labels[src] == labels[dst]).float()
    deg  = torch.zeros(N).scatter_add_(0, src, torch.ones(len(src))).clamp(min=1)
    pur  = torch.zeros(N).scatter_add_(0, src, same)
    return float((pur / deg).mean())

def mean_same_label_degree(edge_index, labels):
    N = labels.shape[0]; src, dst = edge_index
    mask = (src != dst) & (labels[src] == labels[dst]); src_sl = src[mask]
    sl_deg = torch.zeros(N).scatter_add_(0, src_sl, torch.ones(len(src_sl)))
    return float(sl_deg.mean())

def spectral_smoothness(edge_index, labels, num_nodes):
    C = int(labels.max().item()) + 1; Y = torch.zeros(num_nodes, C)
    Y[torch.arange(num_nodes), labels] = 1.0
    src, dst = edge_index; mask = src != dst; src, dst = src[mask], dst[mask]
    deg = torch.zeros(num_nodes).scatter_add_(0, src, torch.ones(len(src)))
    d_inv_sqrt = deg.clamp(min=1).pow(-0.5)
    weights = d_inv_sqrt[src] * d_inv_sqrt[dst]
    AY = torch.zeros(num_nodes, C)
    AY.scatter_add_(0, src.unsqueeze(1).expand(-1, C), Y[dst] * weights.unsqueeze(1))
    return float(1.0 - (Y * AY).sum() / Y.pow(2).sum().clamp(min=1e-8))

def class_cluster_conductance(edge_index, labels, num_nodes):
    C = int(labels.max().item()) + 1; src, dst = edge_index
    mask = src != dst; src, dst = src[mask], dst[mask]
    deg = torch.zeros(num_nodes).scatter_add_(0, src, torch.ones(len(src))); vals = []
    for c in range(C):
        S = labels == c; cut = float(((S[src]) & (~S[dst])).float().sum())
        vol_S = float(deg[S].sum()); vol_c = float(deg.sum()) - vol_S
        denom = min(vol_S, vol_c)
        if denom < 1: continue
        vals.append(cut / denom)
    return float(np.mean(vals)) if vals else float("nan")

def compute_graph_quality(edge_index, labels, num_nodes, tag, dataset, features=None):
    N_edges = int(edge_index.shape[1])
    density = N_edges / max(num_nodes * (num_nodes - 1), 1)
    rec = {
        "dataset": dataset, "tag": tag, "num_edges": N_edges, "density": density,
        "homophily":      homophily_ratio(edge_index, labels),
        "adj_homophily":  adjusted_homophily(edge_index, labels, num_nodes),
        "feat_homophily": feature_homophily(edge_index, features) if features is not None else float("nan"),
        "edge_purity":    edge_purity(edge_index, labels),
        "mean_sl_degree": mean_same_label_degree(edge_index, labels),
        "spectral_smooth": spectral_smoothness(edge_index, labels, num_nodes),
        "conductance": class_cluster_conductance(edge_index, labels, num_nodes),
    }
    return rec


# =============================================================================
#  SECTION 3 — Matched-density subsampling
# =============================================================================

def subsample_to_density(edge_index, target_num_edges, features_or_labels, train_mask=None):
    from torch_geometric.utils import to_undirected
    src, dst = edge_index; mask = src != dst; src, dst = src[mask], dst[mask]
    E = src.shape[0]
    if E <= target_num_edges: return edge_index
    feat = features_or_labels
    if feat.dim() == 1:
        labels = feat
        if train_mask is None:
            raise ValueError("subsample_to_density: train_mask required when features_or_labels is a label tensor")
        both_train = train_mask[src] & train_mask[dst]
        same_train = (labels[src] == labels[dst]).float()
        score = torch.where(both_train, same_train, torch.zeros_like(same_train))
        N = labels.shape[0]
    else:
        Z = F.normalize(feat.float(), dim=-1)
        score = (Z[src] * Z[dst]).sum(dim=-1).clamp(-1.0, 1.0)
        N = feat.shape[0]
    order = torch.argsort(score * E - torch.arange(E, dtype=score.dtype, device=score.device), descending=True)
    keep = order[:target_num_edges]
    return to_undirected(torch.stack([src[keep], dst[keep]]), num_nodes=N)


# =============================================================================
#  SECTION 3b — Semi-supervised label masking
# =============================================================================

def apply_label_mask(train_mask: torch.Tensor, label_mask_ratio: float, seed: int = 0) -> torch.Tensor:
    if label_mask_ratio == 1.0:
        return train_mask
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    n_train = len(train_indices)
    n_keep  = int(round(label_mask_ratio * n_train))
    rng = torch.Generator(); rng.manual_seed(seed)
    perm     = torch.randperm(n_train, generator=rng)
    keep_idx = train_indices[perm[:n_keep]]
    masked   = torch.zeros_like(train_mask)
    if n_keep > 0:
        masked[keep_idx] = True
    return masked


# =============================================================================
#  SECTION 4 — Graph construction utilities used by EUREKA
#  (copied verbatim from sweep_eureka.py — no changes)
# =============================================================================

def _undirected_edges(edge_index, num_nodes):
    """Return a de-duplicated set of undirected edges (u<v), no self-loops,
    as a [2, M] long tensor on CPU."""
    from torch_geometric.utils import to_undirected, remove_self_loops
    ei, _ = remove_self_loops(edge_index)
    ei = to_undirected(ei, num_nodes=num_nodes).cpu()
    src, dst = ei
    lo = torch.minimum(src, dst); hi = torch.maximum(src, dst)
    key = lo.long() * num_nodes + hi.long()
    uniq = torch.unique(key)
    u = (uniq // num_nodes).long(); v = (uniq % num_nodes).long()
    return torch.stack([u, v])


def build_knn_graph_from_embeddings(Z, k=30, sim_threshold=0.30, mutual=True,
                                    add_loops=True, batch_size=512):
    """kNN graph in embedding space (used for optional candidate-edge addition
    and for the density-matched downstream graph)."""
    from torch_geometric.utils import to_undirected, add_self_loops as asl
    N = Z.shape[0]; Z = F.normalize(Z, dim=-1); rows, cols = [], []
    for start in range(0, N, batch_size):
        end = min(start+batch_size, N); B = end - start
        sim = Z[start:end] @ Z.T; sim[:, start:end].fill_diagonal_(-1.0)
        kk = min(k, N - 1)
        vals, idx = sim.topk(kk, dim=1)
        dist = 1.0 - vals; rho = dist[:, 0:1]; sigma = dist[:, -1:] + 1e-6
        weights = torch.exp(-(dist - rho).clamp(min=0.0) / sigma)
        keep = weights > sim_threshold
        src = torch.arange(start, end, device=Z.device).unsqueeze(1).expand(B, kk)
        rows.append(src[keep]); cols.append(idx[keep])
    row = torch.cat(rows); col = torch.cat(cols); edge_index = torch.stack([row, col])
    if mutual and edge_index.numel() > 0:
        ei  = edge_index
        N_n = int(ei.max().item()) + 1
        fwd = ei[0].long() * N_n + ei[1].long()
        rev = ei[1].long() * N_n + ei[0].long()
        keep = torch.isin(rev, fwd)
        edge_index = ei[:, keep]
    edge_index = to_undirected(edge_index)
    if add_loops: edge_index, _ = asl(edge_index, num_nodes=N)
    return edge_index


# =============================================================================
#  SECTION 5 — EUREKA representation model
#  (copied verbatim from sweep_eureka.py — no changes: the structural FUSE
#  branch, the residual-feature branch, the neighbourhood-distribution views,
#  the four-view edge descriptor and the edge-utility scorer f_theta are all
#  reproduced exactly as in the original EUREKA sweep.)
# =============================================================================

def split_dims(dim, struct_dim=None, feat_dim=None, struct_frac=0.5):
    """Split a total embedding budget into (d_z structural, d_x feature) dims.
    Mirrors fsgb's split_dims: explicit overrides win, else split by struct_frac."""
    if struct_dim is not None and feat_dim is not None:
        return int(struct_dim), int(feat_dim)
    d_z = int(round(dim * struct_frac)) if struct_dim is None else int(struct_dim)
    d_z = max(1, min(d_z, dim))
    d_x = (dim - d_z) if feat_dim is None else int(feat_dim)
    return d_z, max(0, d_x)


def fuse(edge_index, num_nodes, dim, weight=None, device="cpu", iters=100, seed=0):
    """FUSE(A): modularity power iteration.

    Returns S ∈ R^{N×d} whose columns are the leading eigenvectors of the
    modularity matrix  B = A − d dᵀ / (2m)  (d = degree vector, m = #edges),
    with ORTHONORMAL columns (SᵀS = I) — matching your fuser.py contract.

    Computed via a Krylov solver on the modularity LinearOperator (same leading
    subspace as power iteration, but robust); falls back to explicit orthogonal
    (subspace) power iteration, then to a dense solve on tiny graphs.

    ── FSGB HOOK ────────────────────────────────────────────────────────────
    To use your real embedder instead, replace this body with a call to
    `fsgb.embedders.fuse.FUSE(dim=dim, ...).fit_transform(gd)` on a GraphData
    built from `edge_index`/`weight`, and return its (orthonormal) output.
    """
    ei = edge_index.cpu().numpy()
    N  = num_nodes
    w  = (np.ones(ei.shape[1], dtype=np.float64) if weight is None
          else weight.detach().cpu().numpy().astype(np.float64))
    A = sp.coo_matrix((w, (ei[0], ei[1])), shape=(N, N)).tocsr()
    A = A.maximum(A.T)                          # symmetric, no self-loops
    deg = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float64)
    two_m = max(deg.sum(), 1e-12)

    d = int(min(dim, N - 1))
    if d < 1:
        return torch.zeros(N, max(dim, 1), device=device)

    # Modularity operator  B x = A x − d (dᵀx)/2m,  shifted to make the leading
    # (most-positive / community) eigenpairs dominant for the iterative solver.
    shift = float(deg.max()) + 1.0

    def _bmat(M):                              # M: [N, k]
        return A @ M - np.outer(deg, (deg @ M) / two_m) + shift * M

    vecs = None
    try:
        linop = spla.LinearOperator((N, N), matvec=lambda x: _bmat(x.reshape(-1, 1)).reshape(-1),
                                     matmat=_bmat, dtype=np.float64)
        _, vecs = spla.eigsh(linop, k=d, which="LA")
    except Exception:
        try:                                   # explicit orthogonal power iteration
            rng = np.random.RandomState(seed)
            Q, _ = np.linalg.qr(rng.randn(N, d))
            for _ in range(iters):
                Q, _ = np.linalg.qr(_bmat(Q))
            vecs = Q
        except Exception:                      # dense fallback (tiny graphs)
            B = A.toarray() - np.outer(deg, deg) / two_m
            _, v_all = np.linalg.eigh(B)
            vecs = v_all[:, -d:]

    S = torch.tensor(np.ascontiguousarray(vecs), dtype=torch.float32, device=device)
    if S.shape[1] < dim:                       # pad if the graph forced d < dim
        S = F.pad(S, (0, dim - S.shape[1]))
    return S


def residual_augment(S, X, d_x, alpha=1.0, normalize_raw=True, center=True):
    """Residual complementary-feature branch, mirroring fsgb.residual_augment.

        R = (I − S Sᵀ) X            (exact, since S is orthonormal)
        H = top-d_x components of R  (energy-matched to S), scaled by alpha
        E = [ S | α H ]

    Returns (E_rep, H) where H is the reduced+scaled feature branch used both in
    E_rep and as the residual view z_res of the edge descriptor.
    """
    if X is None or d_x <= 0:
        return S, torch.zeros(S.shape[0], 0, device=S.device)

    Xf = X.float()
    if normalize_raw:                          # z-score raw feature columns
        mu = Xf.mean(0, keepdim=True); sd = Xf.std(0, keepdim=True) + 1e-8
        Xf = (Xf - mu) / sd

    R = Xf - S @ (S.T @ Xf)                     # orthogonal residual
    if center:
        R = R - R.mean(0, keepdim=True)

    # reduce R to d_x dimensions via truncated SVD (top-d_x directions)
    k = int(min(d_x, R.shape[1], R.shape[0]))
    if k <= 0:
        return S, torch.zeros(S.shape[0], 0, device=S.device)
    try:
        U, sv, _ = torch.linalg.svd(R, full_matrices=False)
        H = U[:, :k] * sv[:k].unsqueeze(0)
    except Exception:
        H = R[:, :k]
    if H.shape[1] < d_x:                        # pad to requested width
        H = F.pad(H, (0, d_x - H.shape[1]))

    # energy-matched concatenation: scale H so ‖H‖_F ≈ ‖S‖_F, then by alpha
    s_energy = S.norm() + 1e-8
    h_energy = H.norm() + 1e-8
    H = H * (s_energy / h_energy)
    E = torch.cat([S, alpha * H], dim=1)
    return E, alpha * H


def neighbour_distributions(edge_index, num_nodes, X, device):
    """1-hop and 2-hop feature distributions D1 = P X, D2 = P² X, with
    P = D⁻¹A the row-normalised adjacency (DHGR-style)."""
    if X is None:
        return None, None
    from torch_geometric.utils import remove_self_loops
    ei, _ = remove_self_loops(edge_index)
    src, dst = ei.to(device)
    deg = torch.zeros(num_nodes, device=device).scatter_add_(
        0, dst, torch.ones(dst.numel(), device=device)).clamp(min=1.0)

    def _prop(M):
        out = torch.zeros_like(M)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, M.shape[1]), M[src])
        return out / deg.unsqueeze(-1)

    D1 = _prop(X)
    D2 = _prop(D1)
    return D1, D2


def _pair_view_feats(M, u, v):
    """Per-view symmetric pair descriptor: [cos, -‖Δ‖, -mean|Δ|] on
    row-normalised endpoints. Returns [E,3]."""
    if M is None or M.shape[1] == 0:
        return torch.zeros(u.shape[0], 3, device=u.device)
    a = F.normalize(M[u], dim=-1); b = F.normalize(M[v], dim=-1)
    cos  = (a * b).sum(-1, keepdim=True)
    dist = (a - b).norm(dim=-1, keepdim=True)
    absd = (a - b).abs().mean(-1, keepdim=True)
    return torch.cat([cos, -dist, -absd], dim=-1)


def edge_descriptor(u, v, S, H, D1, D2):
    """z_uv = [ z_struct , z_res , z_(1) , z_(2) ]  (4 views × 3 feats = 12)."""
    return torch.cat([
        _pair_view_feats(S,  u, v),   # structural utility
        _pair_view_feats(H,  u, v),   # residual-feature information
        _pair_view_feats(D1, u, v),   # 1-hop neighbourhood distribution
        _pair_view_feats(D2, u, v),   # 2-hop neighbourhood distribution
    ], dim=-1)


class EdgeUtilityScorer(nn.Module):
    """f_θ : z_uv -> q_uv  (scalar edge-utility logit).  g(q)=sigmoid(q)."""
    def __init__(self, in_dim=12, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)


# =============================================================================
#  SECTION 6 — EUREKA training pipeline
#  run_eureka() below is copied verbatim from sweep_eureka.py — no changes.
# =============================================================================

def run_eureka(data, dataset_name, cfg, device, seed, label_mask_ratio=1.0):
    """Train EUREKA and return (E*, rewired_edge_index) on CPU.

    E*             : the final combined representation [ S* | α H* ]
    rewired graph  : unweighted edge set of A* (edges surviving the retention
                     threshold, plus optional high-utility candidate edges).
    """
    from tqdm.auto import tqdm

    torch.manual_seed(seed); np.random.seed(seed)
    N = data.num_nodes
    C = data.num_classes
    labels = data.y.to(device)
    masked_train = apply_label_mask(data.train_mask, label_mask_ratio, seed=seed)
    train_mask = masked_train.to(device)
    n_full, n_mask = int(data.train_mask.sum()), int(masked_train.sum())
    print(f"\n{'='*72}\n[EUREKA] {dataset_name} | N={N} | C={C}\n{'='*72}")
    print(f"  [Semi-sup] label_mask_ratio={label_mask_ratio:.3f}  "
          f"visible_train_labels={n_mask}/{n_full}"
          + ("  (label guidance inactive)" if n_mask == 0 else ""))

    X = data.x.to(device).float() if cfg.use_raw_feats else None
    alpha = float(cfg.alpha)
    d_z, d_x = split_dims(cfg.emb_dim, struct_frac=cfg.struct_frac)
    if X is None:
        d_z, d_x = cfg.emb_dim, 0                # structure-only when featureless

    # ── 1) FUSE(A) and the combined representation on the ORIGINAL graph ──────
    ei0 = _undirected_edges(data.edge_index, N)             # [2, M] canonical u<v
    S = fuse(ei0, N, d_z, weight=None, device=device, seed=seed)
    E_rep, H = residual_augment(S, X, d_x, alpha=alpha,
                                normalize_raw=cfg.normalize_raw,
                                center=cfg.center_residual)
    D1, D2 = neighbour_distributions(data.edge_index, N, X, device)
    print(f"  split_dims: d_z(struct)={S.shape[1]}  d_x(feat)={H.shape[1]}  "
          f"| E_rep={E_rep.shape[1]}D")

    # row-normalised E_rep for the representation-agreement target.
    E_hat = F.normalize(E_rep, dim=-1)

    u_all = ei0[0].to(device); v_all = ei0[1].to(device)
    M = u_all.shape[0]

    # ── 2) Train the edge-utility scorer f_θ ─────────────────────────────────
    scorer = EdgeUtilityScorer(in_dim=12, hidden=cfg.scorer_hidden,
                               dropout=cfg.eureka_dropout).to(device)
    opt = Adam(scorer.parameters(), lr=cfg.eureka_lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    def _target(u, v):
        """Surrogate edge utility: representation agreement mapped to [0,1],
        optionally blended with visible-label agreement."""
        sim = (E_hat[u] * E_hat[v]).sum(-1).clamp(-1.0, 1.0)
        t = (sim + 1.0) * 0.5
        lw = float(cfg.label_guide_weight)
        if lw > 0.0 and n_mask > 0:
            both = train_mask[u] & train_mask[v]
            if both.any():
                same = (labels[u] == labels[v]).float()
                t = torch.where(both, (1 - lw) * t + lw * same, t)
        return t

    pos_target = _target(u_all, v_all)
    n_neg = max(0, int(cfg.neg_samples))

    pbar = tqdm(range(1, cfg.eureka_epochs + 1), desc="EUREKA/f_theta",
                leave=False, dynamic_ncols=True)
    best_loss = float("inf")
    for ep in pbar:
        scorer.train(); opt.zero_grad()

        z_pos = edge_descriptor(u_all, v_all, S, H, D1, D2)
        q_pos = scorer(z_pos)
        loss = bce(q_pos, pos_target)

        if n_neg > 0:
            ru = torch.randint(0, N, (M * n_neg,), device=device)
            rv = torch.randint(0, N, (M * n_neg,), device=device)
            valid = ru != rv
            ru, rv = ru[valid], rv[valid]
            z_neg = edge_descriptor(ru, rv, S, H, D1, D2)
            q_neg = scorer(z_neg)
            loss = loss + bce(q_neg, _target(ru, rv))

        if cfg.sparse_weight > 0:            # encourage pruning
            loss = loss + cfg.sparse_weight * torch.sigmoid(q_pos).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
        opt.step()
        cur = loss.item()
        best_loss = min(best_loss, cur)
        if ep == 1 or ep % max(1, cfg.eureka_epochs // 50) == 0 or ep == cfg.eureka_epochs:
            pbar.set_postfix({"loss": f"{cur:.4f}", "best": f"{best_loss:.4f}"})
    pbar.close()

    # ── 3) Reweight the adjacency:  A*_uv = A_uv · g(q_uv) ───────────────────
    scorer.eval()
    with torch.no_grad():
        q_edges = scorer(edge_descriptor(u_all, v_all, S, H, D1, D2))
        w_edges = torch.sigmoid(q_edges)                    # retention weights

    keep = w_edges >= cfg.prune_thresh
    u_keep, v_keep, w_keep = u_all[keep], v_all[keep], w_edges[keep]
    print(f"  retained {int(keep.sum())}/{M} edges "
          f"(prune_thresh={cfg.prune_thresh}, mean w={float(w_edges.mean()):.3f})")

    # ── optional: add high-utility candidate edges (kNN in E_rep space) ──────
    if int(cfg.add_edges) == 1:
        with torch.no_grad():
            cand = build_knn_graph_from_embeddings(
                E_rep.cpu(), k=cfg.knn_k, sim_threshold=0.30,
                mutual=True, add_loops=False).to(device)
            cu, cv = cand
            m2 = cu != cv
            cu, cv = cu[m2], cv[m2]
            if cu.numel() > 0:
                q_cand = scorer(edge_descriptor(cu, cv, S, H, D1, D2))
                w_cand = torch.sigmoid(q_cand)
                good = w_cand >= cfg.prune_thresh
                u_keep = torch.cat([u_keep, cu[good]])
                v_keep = torch.cat([v_keep, cv[good]])
                w_keep = torch.cat([w_keep, w_cand[good]])
                print(f"  added {int(good.sum())} candidate edges")

    # weighted, symmetric edge list for FUSE(A*)
    if u_keep.numel() == 0:                  # degenerate: fall back to original
        u_keep, v_keep, w_keep = u_all, v_all, w_edges
    star_src = torch.cat([u_keep, v_keep])
    star_dst = torch.cat([v_keep, u_keep])
    star_w   = torch.cat([w_keep, w_keep])
    star_ei  = torch.stack([star_src, star_dst])

    # ── 4) Re-embed the rewired graph:  S* = FUSE(A*), H*, E* ────────────────
    S_star = fuse(star_ei, N, d_z, weight=star_w, device=device, seed=seed)
    E_star, _ = residual_augment(S_star, X, d_x, alpha=alpha,
                                 normalize_raw=cfg.normalize_raw,
                                 center=cfg.center_residual)

    # downstream graph = unweighted rewired edge set (undirected)
    from torch_geometric.utils import to_undirected
    rewired_ei = to_undirected(torch.stack([u_keep, v_keep]).cpu(), num_nodes=N)

    print(f"Done | best_loss={best_loss:.4f} | E*={tuple(E_star.shape)} "
          f"| rewired_edges={rewired_ei.shape[1]} | alpha={alpha} "
          f"add_edges={int(cfg.add_edges)}\n{'='*72}\n")
    return E_star.detach().cpu(), rewired_ei


def rewire_eureka(data, cfg, device, seed):
    """Harness wrapper around run_eureka(), in the same caching style as
    rewire_cvgae_hethom(): checkpoint the raw EUREKA outputs, then build the
    final density-matched downstream graph.

    Returns (edge_index, E_star) so downstream eval can use the learned
    representation as node features.
    """
    ratio_tag = f"_ratio{cfg.label_mask_ratio:.3f}"
    fl_tag    = "_featureless" if cfg.featureless else ""
    ckpt = CKPT_DIR / f"{cfg.dataset}_eureka_seed{seed}{fl_tag}{ratio_tag}.pt"
    if ckpt.exists():
        print(f"    [Resume] cached EUREKA embeddings seed={seed}")
        blob = torch.load(ckpt, map_location="cpu")
        E_star, rewired_ei = blob["E_star"], blob["rewired_ei"]
    else:
        E_star, rewired_ei = run_eureka(data, cfg.dataset, cfg, device, seed,
                                         label_mask_ratio=cfg.label_mask_ratio)
        torch.save({"E_star": E_star, "rewired_ei": rewired_ei}, ckpt)
        print(f"    [Saved] EUREKA -> {ckpt}")
    # Density-match the rewired graph to the original edge count (fair
    # comparison), scoring candidate edges by the learned representation E*.
    ei_matched = subsample_to_density(rewired_ei, data.edge_index.shape[1], E_star)
    return ei_matched, E_star


# =============================================================================
#  SECTION 7 — Downstream classifiers
# =============================================================================

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, LINKX

class NodeClassifier(nn.Module):
    def __init__(self, backbone, in_dim, hid_dim, num_classes, dropout=0.5):
        super().__init__(); self.dropout = dropout
        if backbone == "gcn":
            self.conv1 = GCNConv(in_dim, hid_dim); self.conv2 = GCNConv(hid_dim, num_classes)
        elif backbone == "gat":
            heads = 4
            self.conv1 = GATConv(in_dim, hid_dim//heads, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hid_dim, num_classes, heads=1, concat=False, dropout=dropout)
        elif backbone == "sage":
            self.conv1 = SAGEConv(in_dim, hid_dim); self.conv2 = SAGEConv(hid_dim, num_classes)
        else: raise ValueError(backbone)
        self.bn = nn.BatchNorm1d(hid_dim)
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

class H2GCNClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes, dropout=0.5):
        super().__init__(); self.dropout = dropout
        self.fc_ego = nn.Linear(in_dim, hid_dim); self.fc_h1 = nn.Linear(in_dim, hid_dim)
        self.fc_h2  = nn.Linear(in_dim, hid_dim); self.cls  = nn.Linear(3*hid_dim, num_classes)
        self.bn = nn.BatchNorm1d(3*hid_dim)
    def _agg(self, x, edge_index, N):
        src, dst = edge_index
        deg = torch.zeros(N, device=x.device).scatter_add_(0, dst, torch.ones(len(dst), device=x.device)).clamp(min=1)
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.shape[1]), x[src])
        return out / deg.unsqueeze(-1)
    def forward(self, x, edge_index):
        N = x.shape[0]
        h = torch.cat([F.relu(self.fc_ego(x)),
                       F.relu(self.fc_h1(self._agg(x, edge_index, N))),
                       F.relu(self.fc_h2(self._agg(self._agg(x, edge_index, N), edge_index, N)))], dim=-1)
        return self.cls(F.dropout(self.bn(h), p=self.dropout, training=self.training))

def build_classifier(backbone, in_dim, hid_dim, num_classes, num_nodes=None):
    if backbone == "h2gcn":
        return H2GCNClassifier(in_dim, hid_dim, num_classes)

    if backbone == "linkx":
        if num_nodes is None:
            raise ValueError("num_nodes is required for LINKX")

        return LINKX(
            num_nodes=num_nodes,
            in_channels=in_dim,
            hidden_channels=hid_dim,
            out_channels=num_classes,
            num_layers=2,
            num_edge_layers=1,
            num_node_layers=1,
            dropout=0.5,
        )

    return NodeClassifier(backbone, in_dim, hid_dim, num_classes)


# =============================================================================
#  SECTION 8 — Train / evaluate a downstream classifier
# =============================================================================

def train_and_eval(model, Z, edge_index, data, cfg, device):
    """
    Train and evaluate one downstream classifier.

    Returns:
        best_val, test_acc, test_f1, classification_time_s
    """
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()

    t_cls_start = time.perf_counter()

    model = model.to(device)
    Z_d  = Z.to(device)
    ei_d = edge_index.to(device)

    y  = data.y.to(device)
    tm = data.train_mask.to(device)
    vm = data.val_mask.to(device)
    te = data.test_mask.to(device)

    opt = Adam(
        model.parameters(),
        lr=cfg.clf_lr,
        weight_decay=5e-4,
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cfg.clf_epochs,
        eta_min=1e-5,
    )

    best_val = 0.0
    best_state = None

    for ep in range(cfg.clf_epochs):

        model.train()
        opt.zero_grad()

        logits = model(Z_d, ei_d)

        loss = F.cross_entropy(
            logits[tm],
            y[tm],
        )

        loss.backward()
        opt.step()
        sched.step()

        if (ep + 1) % 10 == 0:
            model.eval()

            with torch.no_grad():
                pred = model(Z_d, ei_d).argmax(-1)

                vacc = (
                    (pred[vm] == y[vm])
                    .float()
                    .mean()
                    .item()
                )

            if vacc > best_val:
                best_val = vacc

                best_state = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        pred = model(Z_d, ei_d).argmax(-1)

        tacc = (
            (pred[te] == y[te])
            .float()
            .mean()
            .item()
        )

    # Macro-F1 on the test set (uses the same predictions as the accuracy).
    from common.metrics import accuracy_and_macro_f1
    _, test_f1 = accuracy_and_macro_f1(pred[te], y[te])

    # Make CUDA timing accurate.
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()

    classification_time_s = time.perf_counter() - t_cls_start

    return best_val, tacc, test_f1, classification_time_s


# =============================================================================
#  SECTION 9 — Results persistence
# =============================================================================

def save_result(rec):
    with open(RESULTS_FILE, "a") as f: f.write(json.dumps(rec) + "\n")

def load_results():
    if not RESULTS_FILE.exists(): return []
    with open(RESULTS_FILE) as f: return [json.loads(l) for l in f if l.strip()]

def result_exists(dataset, method, backbone, seed):
    return any(r.get("dataset") == dataset and r.get("method") == method
               and r.get("backbone") == backbone and r.get("seed") == seed
               for r in load_results())


# =============================================================================
#  SECTION 10 — Unified benchmark loop  (EUREKA-rewired + Learned-E only,
#               plus the original-graph / original-features baseline)
# =============================================================================
#
# This file evaluates exactly two graph/feature configurations, each run with
# several downstream classifiers, over seeds {0, 1, 2}:
#
#   1. "original"                 — original graph      + original node features
#   2. "eureka_rewired_learnedE"  — EUREKA-rewired graph + learned representation E*
#
# For every (dataset, method, classifier, seed) we record accuracy, macro-F1 and
# runtime (rewiring + classifier training).  Results are appended to a JSONL file
# after every single run, so an interrupted sweep resumes exactly where it left
# off.  Per-dataset tables, an "average over datasets per classifier" table and
# all figures are produced at the end (and refreshed after every dataset).

ALL_DATASETS = [
    "Actor", "Squirrel-F", "Chameleon-F",
    "Roman-empire", "Amazon-ratings", "Tolokers",
    "HSBM-MED", "STRUC-HET", "FEAT-HET", "MIXED-SIG",
]

# All downstream classifiers are kept (each is a standard node classifier).
CLASSIFIERS = ["gcn", "gat", "sage", "h2gcn", "linkx"]

# The two configurations compared in this file.
#   name                       edge source        feature source
METHODS = ["original", "eureka_rewired_learnedE"]

SYNTHETIC = {"HSBM-MED", "STRUC-HET", "FEAT-HET", "MIXED-SIG"}

# Output layout (OUT_DIR / CKPT_DIR were created at the top of the file).
RESULTS_JSONL = OUT_DIR / "results.jsonl"
TABLES_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "plots"
KEY_FIELDS = ["dataset", "method", "classifier", "seed"]
GROUP_FIELDS = ["method", "classifier"]


def _refresh_reports(store, datasets):
    """(Re)write summary CSVs, print tables and regenerate figures."""
    records = store.all()
    summary, avg = _reporting.write_all_csvs(
        records, GROUP_FIELDS, TABLES_DIR,
        dataset_order=datasets, make_average=True,
    )
    if summary.empty:
        return
    print("\n" + "=" * 72)
    print("  EUREKA BENCHMARK — SUMMARY SO FAR")
    print("=" * 72)
    _reporting.print_dataset_tables(summary, GROUP_FIELDS, datasets)
    _reporting.print_average_table(avg, GROUP_FIELDS)
    _plotting.generate_all_figures(summary, avg, GROUP_FIELDS, FIG_DIR,
                                   dataset_order=datasets)


def run_benchmark(cfg):
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else torch.device(cfg.device))

    print("\n" + "=" * 72)
    print("  EUREKA Rewiring Benchmark  (Learned-E + Original baseline)")
    print("=" * 72)
    print(f"  Datasets    : {cfg.datasets}")
    print(f"  Methods     : {METHODS}")
    print(f"  Classifiers : {cfg.classifiers}")
    print(f"  Seeds       : {cfg.seeds}")
    print(f"  Device      : {device}")
    print(f"  Results     -> {OUT_DIR.resolve()}")
    print("=" * 72)

    store = _checkpoint.ResultStore(RESULTS_JSONL, key_fields=KEY_FIELDS)
    if store.count():
        print(f"  [resume] found {store.count()} completed run(s); "
              f"they will be skipped.\n")

    total_runs = (len(cfg.datasets) * len(METHODS)
                  * len(cfg.classifiers) * len(cfg.seeds))
    tracker = _progress.ProgressTracker(total_runs, label="eureka").start()

    for dataset in cfg.datasets:
        cfg.dataset = dataset
        print(f"\n{'─'*72}\n  DATASET: {dataset}\n{'─'*72}")

        try:
            data = (generate_synthetic_dataset(dataset, seed=0)
                    if dataset in SYNTHETIC
                    else load_real_dataset(dataset, root=cfg.data_root))
        except Exception as e:
            print(f"  [SKIP dataset] {dataset}: {e}")
            # still tick the tracker so ETA stays meaningful
            tracker.tick_skipped(len(METHODS) * len(cfg.classifiers) * len(cfg.seeds))
            continue

        X_raw = data.x.cpu()
        N, C = data.num_nodes, data.num_classes
        print(f"  N={N}  C={C}  edges={data.edge_index.shape[1]}  "
              f"train={int(data.train_mask.sum())}  "
              f"val={int(data.val_mask.sum())}  "
              f"test={int(data.test_mask.sum())}")

        # Lazily-computed, per-seed EUREKA rewiring (cached to disk by
        # rewire_eureka, so resume is cheap).
        eureka_cache = {}   # seed -> (edge_index, E_star, rewire_time_s)

        def get_eureka(seed):
            if seed in eureka_cache:
                return eureka_cache[seed]
            set_global_seed(seed)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            ei, E = rewire_eureka(data, cfg, device, seed)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            rt = time.perf_counter() - t0
            eureka_cache[seed] = (ei.cpu(), E.cpu(), rt)
            return eureka_cache[seed]

        for method in METHODS:
            if method not in cfg.methods:
                # method disabled on the CLI — account for its runs and skip
                tracker.tick_skipped(len(cfg.classifiers) * len(cfg.seeds))
                continue
            for clf_name in cfg.classifiers:
                for seed in cfg.seeds:
                    if cfg.resume and store.exists(
                        dataset=dataset, method=method,
                        classifier=clf_name, seed=seed,
                    ):
                        tracker.tick_skipped()
                        continue

                    t_run = time.perf_counter()
                    try:
                        set_global_seed(seed)

                        if method == "original":
                            ei_use = data.edge_index
                            X_use = X_raw
                            rewire_time = 0.0
                        else:  # eureka_rewired_learnedE
                            ei_use, E_star, rewire_time = get_eureka(seed)
                            X_use = E_star

                        # Reseed right before classifier init for reproducibility.
                        set_global_seed(seed)
                        clf = build_classifier(
                            clf_name, X_use.shape[1], cfg.clf_hid, C,
                            num_nodes=N,
                        )
                        val_acc, test_acc, test_f1, clf_time = train_and_eval(
                            clf, X_use, ei_use, data, cfg, device,
                        )

                        total_time = rewire_time + clf_time
                        rec = {
                            "dataset": dataset,
                            "method": method,
                            "classifier": clf_name,
                            "seed": seed,
                            "acc": test_acc,
                            "f1": test_f1,
                            "val_acc": val_acc,
                            "rewiring_time_s": rewire_time,
                            "train_time_s": clf_time,
                            "total_time_s": total_time,
                        }
                        store.append(rec)
                        note = (f"{dataset}/{method}/{clf_name}/s{seed} "
                                f"acc={test_acc:.4f} f1={test_f1:.4f}")
                        tracker.tick_done(time.perf_counter() - t_run, note=note)

                    except Exception as e:
                        print(f"    [SKIP run] {dataset}/{method}/{clf_name}"
                              f"/seed={seed}: {e}")
                        if cfg.verbose:
                            traceback.print_exc()
                        # Count it so the ETA does not stall; it will be retried
                        # on a future resume because nothing was written.
                        tracker.tick_done(time.perf_counter() - t_run,
                                          note="FAILED")

        # Refresh reports/plots after each dataset so partial progress is visible.
        _refresh_reports(store, cfg.datasets)

    tracker.finish()
    _refresh_reports(store, cfg.datasets)
    print(f"\n[Done] Results, tables and plots in {OUT_DIR.resolve()}")


# =============================================================================
#  SECTION 11 — CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="EUREKA (edge-utility rewiring) benchmark (Learned-E + original baseline)")

    p.add_argument("--datasets",   nargs="+", default=ALL_DATASETS)
    p.add_argument("--methods",    nargs="+", default=METHODS,
                   choices=METHODS,
                   help="Subset of: original eureka_rewired_learnedE")
    p.add_argument("--classifiers", nargs="+", default=CLASSIFIERS,
                   choices=CLASSIFIERS)
    p.add_argument("--seeds",      nargs="+", type=int, default=[0, 1, 2])

    # EUREKA representation / rewiring hyper-parameters
    p.add_argument("--emb_dim",            type=int,   default=128,
                   help="Total embedding width; split into d_z (FUSE) + d_x "
                        "(residual) by --struct_frac.")
    p.add_argument("--struct_frac",        type=float, default=0.5,
                   help="Fraction of emb_dim given to the structural branch S.")
    p.add_argument("--alpha",              type=float, default=1.0,
                   help="Weight on the residual features H in E_rep = [S | αH].")
    p.add_argument("--scorer_hidden",      type=int,   default=64,
                   help="Hidden width of the edge-utility scorer f_θ.")
    p.add_argument("--eureka_epochs",      type=int,   default=300,
                   help="Training epochs for the edge-utility scorer.")
    p.add_argument("--eureka_lr",          type=float, default=1e-2)
    p.add_argument("--eureka_dropout",     type=float, default=0.0)
    p.add_argument("--neg_samples",        type=int,   default=1,
                   help="Sampled non-edges per observed edge when training f_θ.")
    p.add_argument("--sparse_weight",      type=float, default=0.0,
                   help="L1-on-retention penalty; >0 encourages more pruning.")
    p.add_argument("--label_guide_weight", type=float, default=0.0,
                   help="Blend visible-label agreement into the utility target "
                        "for train-train pairs (0 = fully unsupervised).")
    p.add_argument("--prune_thresh",       type=float, default=0.5,
                   help="Retention weight below which an edge is dropped.")
    p.add_argument("--add_edges",          type=int,   default=0,
                   choices=[0, 1],
                   help="1 = also add high-utility kNN candidate edges before "
                        "density-matching; 0 = reweight/prune existing edges only.")
    p.add_argument("--knn_k",              type=int,   default=30)
    p.add_argument("--label_mask_ratio",   type=float, default=1.0, metavar="R",
                   help="Fraction of training labels visible to EUREKA (1.0 = full).")
    p.add_argument("--featureless", action="store_true",
                   help="Ignore node features X (H, D1, D2 become empty; "
                        "EUREKA falls back to structure-only S).")
    p.add_argument("--no_normalize_raw", action="store_true",
                   help="Do NOT z-score raw features before the residual projection.")
    p.add_argument("--no_center_residual", action="store_true",
                   help="Do NOT column-center the residual before SVD reduction.")

    # Downstream classifier hyper-parameters
    p.add_argument("--clf_epochs", type=int,   default=300)
    p.add_argument("--clf_hid",    type=int,   default=128)
    p.add_argument("--clf_lr",     type=float, default=5e-3)

    # Runtime
    p.add_argument("--data_root",  default="./data")
    p.add_argument("--device",     default="auto")
    p.add_argument("--resume",     action="store_true", default=True,
                   help="Skip runs already present in results.jsonl (default on).")
    p.add_argument("--no_resume",  dest="resume", action="store_false")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--verbose",    action="store_true")

    args = p.parse_args()

    if args.smoke_test:
        print("[Smoke-test mode]")
        args.seeds         = args.seeds[:1]
        args.eureka_epochs = min(args.eureka_epochs, 30)
        args.clf_epochs    = min(args.clf_epochs, 30)

    if not (0.0 <= args.label_mask_ratio <= 1.0):
        p.error("--label_mask_ratio must be in [0.0, 1.0]")

    # Derived flags consumed by run_eureka() / residual_augment().
    args.use_raw_feats   = not args.featureless
    args.normalize_raw   = not args.no_normalize_raw
    args.center_residual = not args.no_center_residual

    return args


if __name__ == "__main__":
    cfg = parse_args()
    run_benchmark(cfg)
