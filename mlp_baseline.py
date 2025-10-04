# - Trains an MLP (seed=0, epochs=120, batch=128) on link_prediction_features.csv
# - Saves weights, scaler, metrics, predictions, history

from __future__ import annotations
import os, json
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


DATA_CSV     = "link_prediction_features.csv"
TARGET_COL   = "label"
SEED         = 0
EPOCHS       = 120
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 0.0
HIDDEN_DIMS  = (64, 32)
DROPOUT      = 0.10
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
PATIENCE     = 15
OUT_BASEDIR  = "outputs_mlp"

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_outdir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, ts)
    os.makedirs(path, exist_ok=True)
    return path

def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

#model
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]  #single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

#data
def load_splits(csv_path: str, target: str, seed: int):
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in CSV.")
    y = df[target].values.astype(np.int64)
    Xdf = df.drop(columns=[target]).select_dtypes(include=[np.number]).copy()
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        Xdf.values, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=seed, stratify=y_train
    )
    return X_tr, y_tr, X_val, y_val, X_test, y_test, list(Xdf.columns)

def scale_splits(X_tr, X_val, X_test):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_tr_s, X_val_s, X_test_s, scaler

def to_loader(X: np.ndarray, y: np.ndarray | None, batch: int, shuffle: bool) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(X_t)
    else:
        y_t = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def eval_full(model: nn.Module, loader: DataLoader, dev: torch.device, return_loss=False) -> Dict[str, Any]:
    model.eval()
    logits_all, y_all = [], []
    bce_loss, n_batches = 0.0, 0
    crit = nn.BCEWithLogitsLoss()
    for batch in loader:
        x = batch[0].to(dev)
        logits = model(x)
        logits_all.append(logits.detach().cpu().numpy())
        if len(batch) > 1:
            yb = batch[1].to(dev)
            y_all.append(yb.detach().cpu().numpy())
            if return_loss:
                bce_loss += crit(logits, yb).item()
                n_batches += 1

    logits_all = np.concatenate(logits_all, axis=0)
    probs = 1 / (1 + np.exp(-logits_all))
    out: Dict[str, Any] = {"probs": probs, "logits": logits_all}
    if y_all:
        y_true = np.concatenate(y_all, axis=0).astype(int)
        out["y_true"] = y_true
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, probs))
        except ValueError:
            out["roc_auc"] = float("nan")
        y_pred = (probs >= 0.5).astype(int)
        out["acc@0.5"] = float(accuracy_score(y_true, y_pred))
        out["f1@0.5"]  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if return_loss and n_batches > 0:
        out["bce"] = float(bce_loss / n_batches)
    return out

def sweep_threshold(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    ts = np.linspace(0.05, 0.95, 19)
    best = {"threshold": 0.5, "f1_macro": -1.0, "accuracy": 0.0}
    for t in ts:
        y_pred = (probs >= t).astype(int)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        if f1m > best["f1_macro"]:
            best = {"threshold": float(t), "f1_macro": float(f1m), "accuracy": float(acc)}
    return best


def plot_performance_story(out_dir: str,
                           train_eval: Dict[str, Any],
                           val_eval: Dict[str, Any],
                           test_eval: Dict[str, Any],
                           best_t: float) -> None:
    y_tr, p_tr = train_eval["y_true"], train_eval["probs"]
    y_va, p_va = val_eval["y_true"],   val_eval["probs"]
    y_te, p_te = test_eval["y_true"],  test_eval["probs"]

    fpr_tr, tpr_tr, _ = roc_curve(y_tr, p_tr); auc_tr = roc_auc_score(y_tr, p_tr)
    fpr_va, tpr_va, _ = roc_curve(y_va, p_va); auc_va = roc_auc_score(y_va, p_va)
    fpr_te, tpr_te, _ = roc_curve(y_te, p_te); auc_te = roc_auc_score(y_te, p_te)

    prec_tr, rec_tr, _ = precision_recall_curve(y_tr, p_tr); ap_tr = np.trapz(prec_tr[::-1], rec_tr[::-1])
    prec_va, rec_va, _ = precision_recall_curve(y_va, p_va); ap_va = np.trapz(prec_va[::-1], rec_va[::-1])
    prec_te, rec_te, _ = precision_recall_curve(y_te, p_te); ap_te = np.trapz(prec_te[::-1], rec_te[::-1])

    fig = plt.figure(figsize=(14,8))
    gs = GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.35)
    fig.suptitle("MLP Performance Story", fontsize=16, y=0.98)

    ax = fig.add_subplot(gs[0,0])
    ax.plot(fpr_tr, tpr_tr, label=f"Train AUC={auc_tr:.3f}")
    ax.plot(fpr_va, tpr_va, label=f"Val AUC={auc_va:.3f}")
    ax.plot(fpr_te, tpr_te, label=f"Test AUC={auc_te:.3f}")
    ax.plot([0,1],[0,1],"--", alpha=0.6)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves"); ax.legend()

    ax = fig.add_subplot(gs[0,1])
    ax.plot(rec_tr, prec_tr, label=f"Train AP={ap_tr:.3f}")
    ax.plot(rec_va, prec_va, label=f"Val AP={ap_va:.3f}")
    ax.plot(rec_te, prec_te, label=f"Test AP={ap_te:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–Recall Curves"); ax.legend()

    ax = fig.add_subplot(gs[0,2])
    ax.hist(p_te[y_te==0], bins=30, alpha=0.8, label="label=0")
    ax.hist(p_te[y_te==1], bins=30, alpha=0.6, label="label=1")
    ax.set_xlabel("P(edge)"); ax.set_ylabel("Count"); ax.set_title("Predicted P(edge) — Test"); ax.legend()

    ax = fig.add_subplot(gs[1,0])
    frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, "o-", label="MLP")
    ax.plot([0,1],[0,1],"--", label="Perfect")
    ax.set_xlabel("Predicted prob"); ax.set_ylabel("Empirical positive rate")
    ax.set_title("Calibration — Test"); ax.legend()

    ax = fig.add_subplot(gs[1,1])
    ts = np.linspace(0.05, 0.95, 19)
    f1s, accs = [], []
    for t in ts:
        y_pred = (p_va >= t).astype(int)
        f1s.append(f1_score(y_va, y_pred, average="macro", zero_division=0))
        accs.append(accuracy_score(y_va, y_pred))
    ax.plot(ts, f1s, label="F1")
    ax.plot(ts, accs, label="Accuracy")
    ax.axvline(best_t, color="r", linestyle="--", label=f"Best t={best_t:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep — Validation"); ax.legend()

    ax = fig.add_subplot(gs[1,2])
    y_pred_best = (p_te >= best_t).astype(int)
    cm = confusion_matrix(y_te, y_pred_best)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
    ax.set_title(f"Confusion — Test @ t={best_t:.2f}")
    ax.set_xlabel("Pred"); ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_aspect("equal")
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v>cm.max()/2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(os.path.join(out_dir, "MLP_Performance_Story.png"), dpi=180)
    plt.close(fig)

def plot_training_curves(out_dir: str, hist: Dict[str, List[float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12,7))
    e = hist["epoch"]
    axes[0,0].plot(e, hist["train_bce"], label="Train", lw=1.8)
    axes[0,0].plot(e, hist["val_bce"],   label="Val",   lw=1.8)
    axes[0,0].set_title("BCE vs Epoch"); axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("BCE"); axes[0,0].legend()
    axes[0,1].plot(e, hist["train_auc"], label="Train", lw=1.8)
    axes[0,1].plot(e, hist["val_auc"],   label="Val",   lw=1.8)
    axes[0,1].set_title("AUC vs Epoch"); axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("AUC"); axes[0,1].legend()
    axes[1,0].plot(e, hist["train_acc"], label="Train", lw=1.8)
    axes[1,0].plot(e, hist["val_acc"],   label="Val",   lw=1.8)
    axes[1,0].set_title("Accuracy vs Epoch"); axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("Accuracy"); axes[1,0].legend()
    axes[1,1].plot(e, hist["train_f1"], label="Train", lw=1.8)
    axes[1,1].plot(e, hist["val_f1"],   label="Val",   lw=1.8)
    axes[1,1].set_title("F1 vs Epoch"); axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("F1"); axes[1,1].legend()
    fig.suptitle("Training Curves", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(os.path.join(out_dir, "MLP_Training_Curves.png"), dpi=180)
    plt.close(fig)

@torch.no_grad()
def plot_mlp_interpretability(out_dir: str,
                              model: nn.Module,
                              feature_names: List[str],
                              dev: torch.device,
                              input_dim: int,
                              test_logits: np.ndarray) -> None:
    model.eval()
    grid = np.linspace(-3, 3, 201).astype(np.float32)
    base = torch.zeros((len(grid), input_dim), dtype=torch.float32, device=dev)

    pdp_curves, pdp_vars = [], []
    for j in range(input_dim):
        Xg = base.clone()
        Xg[:, j] = torch.from_numpy(grid).to(dev)
        logits = model(Xg).detach().cpu().numpy()
        pdp_curves.append(logits)
        pdp_vars.append(float(np.var(logits)))
    pdp_curves = np.array(pdp_curves)
    pdp_vars = np.array(pdp_vars)
    top_idx = np.argsort(-pdp_vars)[:3]

    fig = plt.figure(figsize=(16,3.6))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)
    fig.suptitle("How the MLP Learned — PDP (per-feature) and sigmoid (score shaping)", y=1.05)

    for k, j in enumerate(top_idx):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(grid, pdp_curves[j], color="black", lw=2.0)
        ax.set_xlabel(f"{feature_names[j]} (z-scored)")
        ax.set_ylabel("logit")
        ax.set_title(f"PDP for {feature_names[j]} (var={pdp_vars[j]:.2f})")
        ax.grid(alpha=0.2)

    ax = fig.add_subplot(gs[0, -1])
    ax.hist(test_logits, bins=50, alpha=0.6, label="s (test)")
    ax.set_xlabel("s (logit)"); ax.set_ylabel("count")
    ax2 = ax.twinx()
    sline = np.linspace(np.percentile(test_logits, 1)-2, np.percentile(test_logits, 99)+2, 400)
    ax2.plot(sline, 1/(1+np.exp(-sline)), color="red", lw=2.0, label="σ(s)")
    ax2.set_ylabel("σ(s)")
    ax.set_title("σ(s) and s (test)")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "MLP_Interpretability.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


set_seed(SEED)
dev = device()

print(f"[INFO] Using dataset: {DATA_CSV}")
X_tr, y_tr, X_val, y_val, X_te, y_te, feat_names = load_splits(DATA_CSV, TARGET_COL, SEED)
print(f"[INFO] Samples: {X_tr.shape[0] + X_val.shape[0] + X_te.shape[0]}  Features: {X_tr.shape[1]}  Labels: 2")
print(f"[INFO] Target column: {TARGET_COL}")

X_tr_s, X_val_s, X_te_s, scaler = scale_splits(X_tr, X_val, X_te)
train_loader = to_loader(X_tr_s, y_tr, BATCH_SIZE, shuffle=True)
val_loader   = to_loader(X_val_s, y_val, BATCH_SIZE, shuffle=False)
test_loader  = to_loader(X_te_s, y_te, BATCH_SIZE, shuffle=False)

model = MLP(in_dim=X_tr_s.shape[1], hidden=HIDDEN_DIMS, dropout=DROPOUT).to(dev)

#train with epoch logs + early stop on val AUC
crit = nn.BCEWithLogitsLoss()
opt  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
history = {"epoch": [], "train_bce": [], "val_bce": [], "train_auc": [], "val_auc": [],
           "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}
best_auc, best_state, no_improve = -np.inf, None, 0

print("\n[INFO] Training MLP baseline...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss, n = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        run_loss += loss.item(); n += 1
    train_bce = run_loss / max(n,1)

    train_eval = eval_full(model, train_loader, dev, return_loss=True)
    val_eval   = eval_full(model, val_loader,   dev, return_loss=True)

    history["epoch"].append(epoch)
    history["train_bce"].append(train_bce)
    history["val_bce"].append(val_eval["bce"])
    history["train_auc"].append(train_eval["roc_auc"])
    history["val_auc"].append(val_eval["roc_auc"])
    history["train_acc"].append(train_eval["acc@0.5"])
    history["val_acc"].append(val_eval["acc@0.5"])
    history["train_f1"].append(train_eval["f1@0.5"])
    history["val_f1"].append(val_eval["f1@0.5"])

    print(f"[Epoch {epoch:03d}] "
          f"train_bce={train_bce:.4f}  val_bce={val_eval['bce']:.4f}  "
          f"train_auc={train_eval['roc_auc']:.4f}  val_auc={val_eval['roc_auc']:.4f}")

    if np.isfinite(val_eval["roc_auc"]) and val_eval["roc_auc"] > best_auc:
        best_auc = val_eval["roc_auc"]
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping (no improvement {PATIENCE} epochs).")
            break

if best_state is not None:
    model.load_state_dict(best_state)

print("\n[INFO] Final evaluations...")
train_eval = eval_full(model, train_loader, dev)
val_eval   = eval_full(model, val_loader,   dev)
test_eval  = eval_full(model, test_loader,  dev)

best = sweep_threshold(val_eval["y_true"], val_eval["probs"])
t_star = best["threshold"]

y_true_te, p_te, s_te = test_eval["y_true"], test_eval["probs"], test_eval["logits"]
y_pred_05 = (p_te >= 0.5).astype(int)
y_pred_ts = (p_te >= t_star).astype(int)

acc_05  = accuracy_score(y_true_te, y_pred_05)
f1m_05  = f1_score(y_true_te, y_pred_05, average="macro", zero_division=0)
f1w_05  = f1_score(y_true_te, y_pred_05, average="weighted", zero_division=0)
auc_te  = roc_auc_score(y_true_te, p_te)

acc_ts  = accuracy_score(y_true_te, y_pred_ts)
f1m_ts  = f1_score(y_true_te, y_pred_ts, average="macro", zero_division=0)
f1w_ts  = f1_score(y_true_te, y_pred_ts, average="weighted", zero_division=0)

print("\n===== MLP (baseline) =====")
print(f"roc_auc (test)  : {auc_te:.4f}")
print(f"accuracy@0.50   : {acc_05:.4f}   f1_macro@0.50   : {f1m_05:.4f}   f1_weighted@0.50: {f1w_05:.4f}")
print(f"threshold* (val): {t_star:.2f}")
print(f"accuracy@best   : {acc_ts:.4f}   f1_macro@best   : {f1m_ts:.4f}   f1_weighted@best: {f1w_ts:.4f}")

print("\nClassification report (threshold=0.50):")
print(classification_report(y_true_te, y_pred_05, digits=4))

# --------------------------
# Save artifacts + figures
# --------------------------
out_dir = make_outdir(OUT_BASEDIR)
torch.save(model.state_dict(), os.path.join(out_dir, "mlp_model.pt"))
joblib.dump(scaler, os.path.join(out_dir, "mlp_scaler.pkl"))

save_json(os.path.join(out_dir, "mlp_metrics.json"), {
    "test_roc_auc": auc_te,
    "threshold_default": 0.5,
    "test_accuracy@0.5": acc_05,
    "test_f1_macro@0.5": f1m_05,
    "test_f1_weighted@0.5": f1w_05,
    "best_threshold_from_val": t_star,
    "test_accuracy@best": acc_ts,
    "test_f1_macro@best": f1m_ts,
    "test_f1_weighted@best": f1w_ts,
})
pd.DataFrame({
    "y_true": y_true_te,
    "prob": p_te,
    "logit": s_te,
    "y_pred@0.50": y_pred_05,
    f"y_pred@{t_star:.2f}": y_pred_ts,
}).to_csv(os.path.join(out_dir, "mlp_predictions.csv"), index=False)
save_json(os.path.join(out_dir, "mlp_training_history.json"), history)

plot_performance_story(out_dir, train_eval, val_eval, test_eval, best_t=t_star)
plot_training_curves(out_dir, history)
plot_mlp_interpretability(out_dir, model, feat_names, dev=dev, input_dim=X_tr_s.shape[1], test_logits=s_te)

np.savetxt(os.path.join(out_dir, "confusion_t05.txt"),
           confusion_matrix(y_true_te, y_pred_05).astype(int), fmt="%d")
np.savetxt(os.path.join(out_dir, f"confusion_t{t_star:.2f}.txt"),
           confusion_matrix(y_true_te, y_pred_ts).astype(int), fmt="%d")

print(f"\n[INFO] Saved artifacts to: {out_dir}")
print("[INFO] Figures: MLP_Performance_Story.png, MLP_Training_Curves.png, MLP_Interpretability.png")
