from __future__ import annotations
import os, json, glob
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_CSV     = "link_prediction_features.csv"
TARGET_COL   = "label"
SEED         = 0
BATCH_SIZE   = 128
TEST_SIZE    = 0.20
VAL_SIZE     = 0.20
OUT_BASEDIR  = "outputs_mlp"     
EXPLAIN_DIR  = "outputs_mlp_explain"

# Model 
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(64, 32), dropout: float = 0.10):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

#Utils
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def latest_run_dir(base: str) -> str:
    runs = sorted(glob.glob(os.path.join(base, "*")))
    if not runs:
        raise FileNotFoundError(f"No runs found under {base}. Train with mlp_baseline.py first.")
    return runs[-1]

def make_outdir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(EXPLAIN_DIR, ts)
    os.makedirs(d, exist_ok=True)
    return d

def sigmoid(x): return 1/(1+np.exp(-x))

#Data and loaders
def load_splits(csv_path: str, target: str, seed: int) -> Tuple[np.ndarray, ...]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    y = df[target].values.astype(np.int64)
    Xdf = df.drop(columns=[target]).select_dtypes(include=[np.number]).copy()
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    X_trn_all, X_te, y_trn_all, y_te = train_test_split(
        Xdf.values, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trn_all, y_trn_all, test_size=VAL_SIZE, random_state=seed, stratify=y_trn_all
    )
    feat_names = list(df.drop(columns=[target]).select_dtypes(include=[np.number]).columns)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names

def scale_all(X_tr, X_va, X_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    return X_tr_s, X_va_s, X_te_s, scaler

def to_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)

#Eval helpers
@torch.no_grad()
def eval_probs(model, loader, dev):
    model.eval()
    logits, ys = [], []
    for xb, yb in loader:
        xb = xb.to(dev)
        s = model(xb)
        logits.append(s.detach().cpu().numpy())
        ys.append(yb.numpy())
    s = np.concatenate(logits, 0)
    p = sigmoid(s)
    y = np.concatenate(ys, 0).astype(int)
    return y, p, s

#Permutation feature importance (validation set)
def permutation_importance(model, X_val, y_val, base_auc, dev, repeats=5, chunk=2048):
    rng = np.random.RandomState(SEED)
    imp = np.zeros(X_val.shape[1], dtype=float)
    for j in range(X_val.shape[1]):
        drop = []
        for _ in range(repeats):
            Xv = X_val.copy()
            rng.shuffle(Xv[:, j])  #permute feature j
            #chunked forward to save memory
            s_list = []
            for k in range(0, len(Xv), chunk):
                xb = torch.tensor(Xv[k:k+chunk], dtype=torch.float32, device=dev)
                s_list.append(model(xb).detach().cpu().numpy())
            p = sigmoid(np.concatenate(s_list, 0))
            drop.append(base_auc - roc_auc_score(y_val, p))
        imp[j] = np.mean(drop)
    return imp

#Gradient×Input and Integrated Gradients (validation set)
def gradient_x_input(model, X, dev):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=dev, requires_grad=True)
    s = model(X_t)  #logits
    s.sum().backward()  #gradient of sum(logits) wrt inputs
    gx = X_t.grad.detach().cpu().numpy() * X  #gradient times input
    return gx  #shape [N, d]

def integrated_gradients(model, X, dev, steps=64):
    #IG wrt baseline=0 (mean of z-scored features); returns [N, d] attributions for logit.
    model.eval()
    N, d = X.shape
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    baseline = torch.zeros_like(X_t)
    alphas = torch.linspace(0, 1, steps+1, device=dev).view(-1, 1, 1)  #[S+1,1,1]
    path = baseline.unsqueeze(0) + alphas * (X_t.unsqueeze(0) - baseline.unsqueeze(0))  #[S+1, N, d]
    path.requires_grad_(True)
    s = model(path.reshape(-1, d)).reshape(steps+1, N)  #logits along path
    #Trapezoidal approx of integral of gradients
    grads = torch.autograd.grad(s.sum(), path, retain_graph=False)[0]  #[S+1, N, d]
    avg_grads = (grads[:-1] + grads[1:]) / 2.0  # [S, N, d]
    integral = avg_grads.mean(dim=0)  #[N, d]
    ig = (X_t - baseline).detach().cpu().numpy() * integral.detach().cpu().numpy()
    return ig

#first layer weights + activations
def first_layer_weights(model: MLP) -> np.ndarray:
    W = model.net[0].weight.detach().cpu().numpy()  #[h0, d]
    return W 

@torch.no_grad()
def collect_activations(model: MLP, loader, dev):
    model.eval()
    acts = { "layer0": [], "layer1": [] }  #ReLU outputs after each hidden layer
    for xb, _ in loader:
        xb = xb.to(dev)
        z0 = model.net[0](xb); a0 = torch.relu(z0); a0 = model.net[2](a0) 
        z1 = model.net[3](a0); a1 = torch.relu(z1); a1 = model.net[5](a1)
        acts["layer0"].append(a0.detach().cpu().numpy())
        acts["layer1"].append(a1.detach().cpu().numpy())
    acts = {k: np.concatenate(v, 0).ravel() for k, v in acts.items()}
    return acts


def barplot(values, labels, title, path, top=None):
    order = np.argsort(values)[::-1]
    if top: order = order[:top]
    plt.figure(figsize=(8,4.5))
    plt.bar(np.arange(len(order)), values[order])
    plt.xticks(np.arange(len(order)), [labels[i] for i in order], rotation=60, ha="right")
    plt.ylabel("Importance"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=170); plt.close()

def heatmap(W, feat_names, path):
    plt.figure(figsize=(10,5))
    plt.imshow(W, aspect="auto")
    plt.colorbar(label="weight")
    plt.yticks(range(W.shape[0]), [f"h0_{i}" for i in range(W.shape[0])])
    plt.xticks(range(len(feat_names)), feat_names, rotation=60, ha="right")
    plt.title("First-layer weights (hidden x features)")
    plt.tight_layout(); plt.savefig(path, dpi=170); plt.close()

def hist_pair(a_tr, a_va, title, path):
    plt.figure(figsize=(7,4))
    plt.hist(a_tr, bins=60, alpha=0.6, label="train")
    plt.hist(a_va, bins=60, alpha=0.6, label="val")
    plt.title(title); plt.xlabel("activation"); plt.ylabel("count")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=170); plt.close()


set_seed(SEED)
dev = device()

print("[INFO] Loading data…")
X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names = load_splits(DATA_CSV, TARGET_COL, SEED)
X_tr_s, X_va_s, X_te_s, scaler = scale_all(X_tr, X_va, X_te)
train_loader = to_loader(X_tr_s, y_tr, BATCH_SIZE, shuffle=False)
val_loader   = to_loader(X_va_s, y_va, BATCH_SIZE, shuffle=False)

print("[INFO] Restoring latest trained MLP…")
run_dir = latest_run_dir(OUT_BASEDIR)
state_path = os.path.join(run_dir, "mlp_model.pt")
model = MLP(in_dim=X_tr_s.shape[1]).to(dev)
model.load_state_dict(torch.load(state_path, map_location=dev))

# Baseline AUC on validation
y_val, p_val, s_val = eval_probs(model, val_loader, dev)
base_auc = roc_auc_score(y_val, p_val)
print(f"[INFO] Val ROC-AUC (baseline): {base_auc:.4f}")

out_dir = make_outdir()

# 1) Permutation importance
print("[INFO] Permutation importance…")
imp = permutation_importance(model, X_va_s, y_val, base_auc, dev, repeats=5)
np.save(os.path.join(out_dir, "perm_importance.npy"), imp)
barplot(imp, feat_names, "Permutation Importance (validation)", os.path.join(out_dir, "MLP_PermutationImportance.png"), top=20)

# 2) Gradient×Input
print("[INFO] Gradient×Input…")
gxi = gradient_x_input(model, X_va_s, dev)        # [N, d]
gxi_mean = np.mean(np.abs(gxi), axis=0)
np.save(os.path.join(out_dir, "saliency_gradxinput.npy"), gxi_mean)
barplot(gxi_mean, feat_names, "Saliency (Gradient×Input, |mean|)", os.path.join(out_dir, "MLP_Saliency.png"), top=20)

# 3) Integrated Gradients
print("[INFO] Integrated Gradients…")
ig = integrated_gradients(model, X_va_s, dev, steps=64)   # [N, d]
ig_mean = np.mean(ig, axis=0)              # signed
ig_mean_abs = np.mean(np.abs(ig), axis=0)  # magnitude for ranking
np.save(os.path.join(out_dir, "integrated_gradients.npy"), ig)
top_idx = np.argsort(ig_mean_abs)[::-1][:20]
means = ig_mean[top_idx]
errs  = np.std(ig[:, top_idx], axis=0)
plt.figure(figsize=(9,5))
plt.bar(range(len(top_idx)), means, yerr=errs, capsize=3)
plt.xticks(range(len(top_idx)), [feat_names[i] for i in top_idx], rotation=60, ha="right")
plt.ylabel("Integrated Gradients (mean ± SD)"); plt.title("Integrated Gradients — validation")
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "MLP_IntegratedGradients.png"), dpi=170); plt.close()

# 4) First-layer weights heatmap
print("[INFO] First-layer weights…")
W = first_layer_weights(model)  #[h0, d]
heatmap(W, feat_names, os.path.join(out_dir, "MLP_FirstLayer_Weights.png"))

# 5) Activation histograms (saturation / dead ReLUs check)
print("[INFO] Activation histograms…")
acts_tr = collect_activations(model, train_loader, dev)
acts_va = collect_activations(model, val_loader, dev)
hist_pair(acts_tr["layer0"], acts_va["layer0"], "Hidden layer 0 activations", os.path.join(out_dir, "MLP_Activations_L0.png"))
hist_pair(acts_tr["layer1"], acts_va["layer1"], "Hidden layer 1 activations", os.path.join(out_dir, "MLP_Activations_L1.png"))

print(f"[INFO] Done. Outputs in: {out_dir}")
