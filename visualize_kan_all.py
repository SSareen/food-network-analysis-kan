import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, accuracy_score, roc_auc_score
)

plt.style.use('default')
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
    "lines.linewidth": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
})

from train_kan import (
    KANBinaryClassifier, FEATURE_COLS,
    RANDOM_SEED, TRAIN_FRAC, VAL_FRAC, TEST_FRAC
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_CSV = "link_prediction_features.csv"
MODEL_PATH = "kan_model.pt"
SCALER_JSON = "kan_scaler.json"
HISTORY_JSON = "kan_training_history.json"

COL_ACCENT  = "#3b82f6"  #blue
COL_ACCENT2 = "#10b981"  #green
COL_POS     = "#ef4444"  #red
COL_NEG     = "#111827"  #near-black
COL_DIAG    = "#9ca3af"  #gray

def load_scaler(json_path=SCALER_JSON):
    with open(json_path, "r") as f:
        obj = json.load(f)
    mean = np.array(obj["mean"], dtype=np.float32)
    std  = np.array(obj["std"],  dtype=np.float32)
    std  = np.where(std == 0, 1e-8, std)
    return mean, std

def zscore(X, mean, std):
    return (X - mean) / std

def load_and_split(df, seed=RANDOM_SEED):
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    n = len(X)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    n_test  = n - n_train - n_val

    return {
        "train": (X[:n_train], y[:n_train]),
        "val":   (X[n_train:n_train+n_val], y[n_train:n_train+n_val]),
        "test":  (X[n_train+n_val:], y[n_train+n_val:])
    }

def load_model(path=MODEL_PATH):
    model = KANBinaryClassifier(d_features=len(FEATURE_COLS))
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

def predict_probs(model, Xs):
    with torch.no_grad():
        logits = model(torch.tensor(Xs, device=DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy()
    return probs

def pick_best_threshold(y_true, probs):
    ts = np.linspace(0, 1, 401)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def load_history(path=HISTORY_JSON):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

#KAN internals
def compute_sq(model, Xs):
    X = torch.tensor(Xs, device=DEVICE)
    B, d = X.shape
    s_list = []
    with torch.no_grad():
        for q in range(len(model.phi)):
            s_q = torch.zeros(B, device=DEVICE)
            for p in range(d):
                s_q = s_q + model.psi[q][p](X[:, p])
            s_list.append(s_q.cpu().numpy())
    return s_list

def psi_variance_ranking(model, Xs):
    X = torch.tensor(Xs, device=DEVICE)
    B, d = X.shape
    with torch.no_grad():
        agg = torch.zeros(B, d, device=DEVICE)
        for q in range(len(model.phi)):
            for p in range(d):
                agg[:, p] += model.psi[q][p](X[:, p])
        var = agg.var(dim=0).cpu().numpy()
    order = np.argsort(var)[::-1]
    return order, var

#training curves
def fig_training_curves(history, outpath):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), facecolor='white')
    fig.suptitle("Training Curves", fontsize=16)

    if history is None or not history.get("epoch"):
        for ax in axs.ravel():
            ax.axis("off")
        fig.text(0.5, 0.5, "No kan_training_history.json found.\nRun train_kan.py (updated) to log history.",
                 ha="center", va="center", fontsize=12)
    else:
        epochs = np.array(history["epoch"])
        tr = pd.DataFrame(history["train"])
        va = pd.DataFrame(history["val"])

        axs[0,0].plot(epochs, tr["bce"], label="Train", color=COL_ACCENT)
        axs[0,0].plot(epochs, va["bce"], label="Val", color=COL_POS)
        axs[0,0].set_title("BCE vs Epoch"); axs[0,0].set_xlabel("Epoch"); axs[0,0].set_ylabel("BCE"); axs[0,0].legend()

        axs[0,1].plot(epochs, tr["auc"], label="Train", color=COL_ACCENT)
        axs[0,1].plot(epochs, va["auc"], label="Val", color=COL_POS)
        axs[0,1].set_title("AUC vs Epoch"); axs[0,1].set_xlabel("Epoch"); axs[0,1].set_ylabel("AUC"); axs[0,1].legend()

        axs[1,0].plot(epochs, tr["acc"], label="Train", color=COL_ACCENT)
        axs[1,0].plot(epochs, va["acc"], label="Val", color=COL_POS)
        axs[1,0].set_title("Accuracy vs Epoch"); axs[1,0].set_xlabel("Epoch"); axs[1,0].set_ylabel("Accuracy"); axs[1,0].legend()

        axs[1,1].plot(epochs, tr["f1"], label="Train", color=COL_ACCENT)
        axs[1,1].plot(epochs, va["f1"], label="Val", color=COL_POS)
        axs[1,1].set_title("F1 vs Epoch"); axs[1,1].set_xlabel("Epoch"); axs[1,1].set_ylabel("F1"); axs[1,1].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

#performance story
def fig_performance_story(df, model, mean, std, outpath):
    splits = load_and_split(df)
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    Xte, yte = splits["test"]
    Xtr_s, Xva_s, Xte_s = zscore(Xtr, mean, std), zscore(Xva, mean, std), zscore(Xte, mean, std)
    ptr, pva, pte = predict_probs(model, Xtr_s), predict_probs(model, Xva_s), predict_probs(model, Xte_s)

    best_t, best_f1 = pick_best_threshold(yva, pva)

    fpr_tr, tpr_tr, _ = roc_curve(ytr, ptr); auc_tr = auc(fpr_tr, tpr_tr)
    fpr_va, tpr_va, _ = roc_curve(yva, pva); auc_va = auc(fpr_va, tpr_va)
    fpr_te, tpr_te, _ = roc_curve(yte, pte); auc_te = auc(fpr_te, tpr_te)

    pr_tr, rc_tr, _ = precision_recall_curve(ytr, ptr); ap_tr = average_precision_score(ytr, ptr)
    pr_va, rc_va, _ = precision_recall_curve(yva, pva); ap_va = average_precision_score(yva, pva)
    pr_te, rc_te, _ = precision_recall_curve(yte, pte); ap_te = average_precision_score(yte, pte)

    ts = np.linspace(0,1,401)
    f1s = [f1_score(yva, (pva>=t).astype(int)) for t in ts]
    accs= [accuracy_score(yva, (pva>=t).astype(int)) for t in ts]

    preds_te = (pte >= best_t).astype(int)
    cm = confusion_matrix(yte, preds_te)

    #calibration
    bins = np.linspace(0,1,11)
    inds = np.digitize(pte, bins) - 1
    avg_pred, frac_pos = [], []
    for b in range(len(bins)-1):
        mask = inds == b
        if mask.any():
            avg_pred.append(pte[mask].mean())
            frac_pos.append(yte[mask].mean())

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), facecolor='white')
    fig.suptitle("KAN Performance Story", fontsize=16)

    ax = axs[0,0]
    ax.plot(fpr_tr, tpr_tr, label=f"Train AUC={auc_tr:.3f}", color=COL_ACCENT)
    ax.plot(fpr_va, tpr_va, label=f"Val AUC={auc_va:.3f}", color=COL_POS)
    ax.plot(fpr_te, tpr_te, label=f"Test AUC={auc_te:.3f}", color=COL_ACCENT2)
    ax.plot([0,1],[0,1],'--', color=COL_DIAG, lw=1.2)
    ax.set_title("ROC Curves"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()

    ax = axs[0,1]
    ax.plot(rc_tr, pr_tr, label=f"Train AP={ap_tr:.3f}", color=COL_ACCENT)
    ax.plot(rc_va, pr_va, label=f"Val AP={ap_va:.3f}", color=COL_POS)
    ax.plot(rc_te, pr_te, label=f"Test AP={ap_te:.3f}", color=COL_ACCENT2)
    ax.set_title("Precision–Recall Curves"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()

    ax = axs[0,2]
    ax.hist(pte[yte==0], bins=40, alpha=0.65, color=COL_NEG, label="label=0")
    ax.hist(pte[yte==1], bins=40, alpha=0.65, color=COL_POS, label="label=1")
    ax.set_title("Predicted P(edge) — Test"); ax.set_xlabel("P(edge)"); ax.set_ylabel("Count"); ax.legend()

    ax = axs[1,0]
    ax.plot([0,1],[0,1],'--', color=COL_DIAG, lw=1.2, label="Perfect")
    if avg_pred:
        ax.plot(avg_pred, frac_pos, marker='o', color=COL_ACCENT)
    ax.set_title("Calibration — Test"); ax.set_xlabel("Predicted prob"); ax.set_ylabel("Empirical positive rate")

    ax = axs[1,1]
    ax.plot(ts, f1s, label="F1", color=COL_ACCENT)
    ax.plot(ts, accs, label="Accuracy", color=COL_ACCENT2)
    ax.axvline(best_t, color=COL_POS, linestyle="--", lw=2, label=f"Best t={best_t:.2f}")
    ax.set_title("Threshold Sweep — Validation"); ax.set_xlabel("Threshold"); ax.set_ylabel("Score"); ax.legend()

    ax = axs[1,2]
    im = ax.imshow(cm, cmap="Blues")
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=11)
    ax.set_xticks([0,1], labels=["Pred 0","Pred 1"])
    ax.set_yticks([0,1], labels=["True 0","True 1"])
    ax.set_title(f"Confusion — Test @ t={best_t:.2f}")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    return Xtr_s, ytr, Xva_s, yva, Xte_s, yte, best_t

#how the kan learned
def fig_how_kan_learned(model, Xtr_s, outpath, top_k_features=3, max_q_to_show=2):
    order, var = psi_variance_ranking(model, Xtr_s)
    top_feats = order[:min(top_k_features, len(order))]

    cols = len(top_feats) + max_q_to_show
    fig, axs = plt.subplots(1, cols, figsize=(5*cols, 4), facecolor='white')
    fig.suptitle("How the KAN Learned — ψ (per-feature) and φ (score shaping)", fontsize=16)

    for i, p in enumerate(top_feats):
        ax = axs[i] if cols > 1 else axs
        xs = torch.linspace(-3, 3, 400, device=DEVICE)
        ys_all = []
        with torch.no_grad():
            for q in range(len(model.phi)):
                ys = model.psi[q][p](xs).cpu().numpy()
                ys_all.append(ys)
                ax.plot(xs.cpu().numpy(), ys, alpha=0.5, lw=1.5)
            ys_sum = np.sum(np.stack(ys_all, axis=0), axis=0)
            ax.plot(xs.cpu().numpy(), ys_sum, lw=2.5, color="black", label="sum_q ψ")
        ax.set_title(f"ψ for {FEATURE_COLS[p]} (var={var[p]:.2f})")
        ax.set_xlabel(f"{FEATURE_COLS[p]} (z-scored)")
        ax.set_ylabel("ψ output")
        ax.legend(loc="best", fontsize=8)

    s_list = compute_sq(model, Xtr_s)
    for k in range(max_q_to_show):
        idx = len(top_feats) + k
        if idx >= cols: break
        ax = axs[idx] if cols > 1 else axs
        s_q = s_list[k]
        ax.hist(s_q, bins=40, alpha=0.35, color=COL_ACCENT, label="s_q")
        xs = torch.linspace(min(-10, float(np.min(s_q))), max(10, float(np.max(s_q))), 400, device=DEVICE)
        with torch.no_grad():
            ys = model.phi[k](xs).cpu().numpy()
        ax2 = ax.twinx()
        ax2.plot(xs.cpu().numpy(), ys, color=COL_POS, lw=2, label="φ_q(s)")
        ax.set_title(f"φ and s_q (q={k})")
        ax.set_xlabel("s_q"); ax.set_ylabel("count"); ax2.set_ylabel("φ_q(s)")
        # merge legends
        h1,l1 = ax.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, loc="best", fontsize=8)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

#main
def main(outdir="kan_viz"):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    mean, std = load_scaler()
    model = load_model()
    history = load_history()

    fig_training_curves(history, os.path.join(outdir, "fig1_training_curves.png"))
    Xtr_s, ytr, Xva_s, yva, Xte_s, yte, best_t = fig_performance_story(
        df, model, mean, std, os.path.join(outdir, "fig2_performance_story.png")
    )
    fig_how_kan_learned(model, Xtr_s, os.path.join(outdir, "fig3_how_kan_learned.png"))

    print("Saved:")
    print(" -", os.path.join(outdir, "fig1_training_curves.png"))
    print(" -", os.path.join(outdir, "fig2_performance_story.png"))
    print(" -", os.path.join(outdir, "fig3_how_kan_learned.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="kan_viz")
    args = ap.parse_args()
    main(outdir=args.outdir)
