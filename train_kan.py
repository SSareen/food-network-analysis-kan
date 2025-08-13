import json
import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils import shuffle as sk_shuffle

DATA_CSV = "link_prediction_features.csv" 
FEATURE_COLS = [
    "deg_u", "deg_v", "common_neighbors",
    "jaccard", "adamic_adar", "pref_attachment",
]
LABEL_COL = "label"

RANDOM_SEED = 0
#4k samples/600 samples/600 samples
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.15, 0.15
#number of samples model sees before updating the weights once
BATCH_SIZE = 128
EPOCHS = 120
#how big of a step the optimizer takes when updating weights
LR = 3e-3
#penalizes large weights to reduce overfitting
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#KAN parameter
Q_TERMS = 6       #number of outer sums (phi terms) (can take 2n+1 but first I am trying with n here)
M_CENTERS = 8      #RBF centers per 1D function (start with 5-10 for smooth problems)
INIT_GAMMA = 1.0   # RBF sharpness (trainable per (q,p))

#some utilities
def set_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class StandardScaler:
    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)

    def save(self, path: str):
        payload = {"mean": self.mean_.tolist(), "std": self.std_.tolist()}
        with open(path, "w") as f:
            json.dump(payload, f)

#load data
def load_data():
    df = pd.read_csv('link_prediction_features.csv')

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)

    #shuffling the data
    X, y = sk_shuffle(X, y, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #train/val/test split
    n = len(X)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    n_test = n - n_train - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],        y[n_train+n_val:]

    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler)

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#BUILDING KANs

#RBFIF: learnable 1-D dunction 
#Radial Basis Function
class RBF1D(nn.Module):
    '''1D function approximator: sum_j w_j * exp(-gamma * (x - c_j)^2)
    x: (...,) tensor
    returns: (...,) tensor'''

    #network learns any smooth 1-D curve
    def __init__ (self, num_centers=M_CENTERS):
        super().__init__()
        self.num_centers = num_centers 
        init_centers = torch.linspace(-3, 3, steps = num_centers)
        self.centers = nn.Parameter(init_centers)
        self.weights = nn.Parameter(torch.zeros(num_centers))
        self.log_gamma = nn.Parameter(torch.zeros(num_centers))

    def forward(self, x):
        x_exp = x.unsqueeze(-1)                      #(..., 1)
        diff2 = (x_exp - self.centers)**2            #(..., M)
        gamma = torch.exp(self.log_gamma)
        basis = torch.exp(-gamma * diff2)            #(..., M)
        out = (basis * self.weights).sum(dim=-1)     #(...,)
        return out
    
class KANBinaryClassifier(nn.Module):
    '''
    Implements: sum_q phi_q( sum_p psi_{q,p}(x_p) ) + b
    - psi_{q,p}: RBF1D for each term q and feature p
    - phi_q: a learnable 1D nonlinearity (another small RBF1D)
    '''
    def __init__(self, d_features, q_terms = Q_TERMS, m_centers = M_CENTERS):
        super().__init__()
        self.d = d_features
        self.q = q_terms

        #psi_{q, p}
        self.psi = nn.ModuleList([
            nn.ModuleList([RBF1D(num_centers=m_centers) for _ in range(d_features)])
            for _ in range(q_terms)
        ])

        #phi_q
        self.phi = nn.ModuleList([RBF1D(num_centers=m_centers) for _ in range(q_terms)])
        
        #final bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # x: (B, d)
        #For each q: s_q = sum_p psi_{q,p}(x_p)
        #then y = sum_q phi_q(s_q) + b
        B = x.shape[0]
        total = torch.zeros(B, device=x.device)
        for q in range(self.q):
            s_q = torch.zeros(B, device=x.device)
            for p in range(self.d):
                s_q = s_q + self.psi[q][p](x[:, p])
            total = total + self.phi[q](s_q)
        
        logits = total + self.bias
        return logits.squeeze(-1)

#training helpers
def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    # BCE
    bce = float(nn.BCEWithLogitsLoss()(torch.tensor(logits), torch.tensor(labels)))
    return {"acc": acc, "f1": f1, "auc": auc, "bce": bce}

def train():
    set_seeds()

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data()
    d = X_train.shape[1]

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  BATCH_SIZE, shuffle=False)

    model = KANBinaryClassifier(d_features=d).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    history = {"epoch": [], "train": [], "val": []}
    best_val = None
    patience, patience_limit = 0, 12

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        train_metrics = evaluate(model, train_loader, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch:03d} | "
            f"Train: acc {train_metrics['acc']:.3f} auc {train_metrics['auc']:.3f} f1 {train_metrics['f1']:.3f} bce {train_metrics['bce']:.3f} "
            f"| Val: acc {val_metrics['acc']:.3f} auc {val_metrics['auc']:.3f} f1 {val_metrics['f1']:.3f} bce {val_metrics['bce']:.3f}"
        )

        history["epoch"].append(epoch)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        score = val_metrics["auc"]
        if best_val is None or (not math.isnan(score) and score > best_val):
            best_val = score
            patience = 0
            torch.save(model.state_dict(), "kan_model.pt")
        else:
            patience += 1
            if patience >= patience_limit:
                print("Early stopping.")
                break

    #final test
    model.load_state_dict(torch.load("kan_model.pt", map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, DEVICE)
    print(f"\nTest: acc {test_metrics['acc']:.3f} auc {test_metrics['auc']:.3f} f1 {test_metrics['f1']:.3f}")

    history["test"] = test_metrics
    scaler.save("kan_scaler.json")
    with open("kan_training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Saved model to kan_model.pt, scaler to kan_scaler.json, history to kan_training_history.json")

if __name__ == "__main__":
    train()

