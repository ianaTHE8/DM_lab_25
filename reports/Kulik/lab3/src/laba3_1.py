import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)

CFG = {
    "seed": 42,
    "epochs_pre": 50,
    "epochs_head": 20,
    "epochs_full": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "hidden": (64, 32, 16),
}
CFG["epochs_base"] = CFG["epochs_head"] + CFG["epochs_full"]
DEVICE = "cpu"

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        h1, h2, h3 = CFG["hidden"]
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, output_dim),
        )

    def forward(self, x):
        return self.layers(x)

class AEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

def get_batches(X, y=None):
    order = torch.randperm(len(X))
    for start in range(0, len(X), CFG["batch_size"]):
        idx = order[start:start + CFG["batch_size"]]
        xb = torch.tensor(X[idx], dtype=torch.float32).to(DEVICE)
        if y is None:
            yield xb
        else:
            yield xb, torch.tensor(y[idx]).to(DEVICE)

def pretrain(X, layer_sizes):
    activations = X.copy()
    learned, ae_histories = [], []

    for i in range(len(layer_sizes) - 1):
        in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
        ae = AEModel(in_dim, out_dim).to(DEVICE)
        opt = torch.optim.Adam(ae.parameters(), lr=CFG["lr"])
        loss_fn = nn.MSELoss()

        hist = []
        for _ in range(CFG["epochs_pre"]):
            total, n = 0.0, 0
            for xb in get_batches(activations):
                opt.zero_grad()
                loss = loss_fn(ae(xb), xb)
                loss.backward()
                opt.step()
                total += loss.item()
                n += 1
            hist.append(total / n)
        ae_histories.append(hist)

        enc = ae.encoder[0]
        learned.append((enc.weight.detach().clone(), enc.bias.detach().clone()))

        with torch.no_grad():
            t = torch.tensor(activations, dtype=torch.float32).to(DEVICE)
            activations = ae.encoder(t).cpu().numpy()
    return learned, ae_histories

def make_net(input_dim, output_dim, weights=None):
    model = MLPModel(input_dim, output_dim).to(DEVICE)
    if weights is not None:
        linear = [m for m in model.layers if isinstance(m, nn.Linear)]
        for layer, (w, b) in zip(linear[:-1], weights):
            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)
    return model

def train(model, X, y, epochs):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=CFG["lr"])
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for _ in range(epochs):
        total, n = 0.0, 0
        for xb, yb in get_batches(X, y):
            loss = loss_fn(model(xb), yb.long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        history.append(total / n)
    return history

def set_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad = value

def train_pretrained(model, X, y):
    head = [m for m in model.layers if isinstance(m, nn.Linear)][-1]
    set_requires_grad(model, False)
    set_requires_grad(head, True)
    h1 = train(model, X, y, CFG["epochs_head"])
    set_requires_grad(model, True)
    h2 = train(model, X, y, CFG["epochs_full"])
    return h1 + h2

def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    model.train()
    return out.argmax(1).cpu().numpy()

def report_classification(yte, pred, title, labels, fname):
    print(f"{title}:")
    print(f"Accuracy: {accuracy_score(yte, pred):.4f}")
    print(f"F1 (macro): {f1_score(yte, pred, average='macro', zero_division=0):.4f}")
    print(f"F1 (weight): {f1_score(yte, pred, average='weighted', zero_division=0):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(yte, pred, labels=range(len(labels))))
    print()

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        yte, pred, labels=range(len(labels)), display_labels=labels,
        cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(f"Confusion matrix - {title}")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def plot_loss(h_base, h_pre, title, fname):
    plt.figure(figsize=(10, 5))
    plt.plot(h_base, label="Без предобучения", linewidth=2)
    plt.plot(h_pre, label="С предобучением", linewidth=2)
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def plot_ae_loss(ae_histories, title, fname):
    plt.figure(figsize=(10, 5))
    for i, h in enumerate(ae_histories, 1):
        plt.plot(h, label=f"Автоэнкодер слоя {i}", linewidth=2)
    plt.xlabel("Эпоха")
    plt.ylabel("MSE реконструкции")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def run_experiment(name, tag, X, y, class_labels):
    print(f"Датасет {name}")
    print("Задача: классификация")
    print(f"Объектов: {len(X)}, признаков: {X.shape[1]}, классов: {len(class_labels)}")
    counts = np.bincount(y, minlength=len(class_labels))
    print("Распределение классов:",
          {lab: int(c) for lab, c in zip(class_labels, counts)})

    n_classes = len(class_labels)
    seed = CFG["seed"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)
    d = Xtr.shape[1]

    torch.manual_seed(seed)
    m_base = make_net(d, n_classes)
    h_base = train(m_base, Xtr, ytr, CFG["epochs_base"])
    p_base = predict(m_base, Xte)

    torch.manual_seed(seed)
    weights, ae_hist = pretrain(Xtr, [d] + list(CFG["hidden"]))
    torch.manual_seed(seed)
    m_pre = make_net(d, n_classes, weights)
    h_pre = train_pretrained(m_pre, Xtr, ytr)
    p_pre = predict(m_pre, Xte)

    print("\nРезультаты:")
    report_classification(yte, p_base, "Без предобучения", class_labels,
                          f"{tag}_confusion_base.png")
    report_classification(yte, p_pre, "С предобучением", class_labels,
                          f"{tag}_confusion_pre.png")

    plot_loss(h_base, h_pre, f"{name} - Loss по эпохам", f"{tag}_loss.png")
    plot_ae_loss(ae_hist, f"{name} - обучение автоэнкодеров", f"{tag}_ae_loss.png")
    print()

def load_hcv():
    path = "HCV-Egy-Data.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Нет файла {path} рядом со скриптом.")
    df = pd.read_csv(path)

    target = [c for c in df.columns if "staging" in c.lower()]
    if not target:
        raise ValueError(f"Нет целевой колонки staging. Есть: {list(df.columns)}")
    target = target[0]

    df = df.fillna(df.median(numeric_only=True))
    y_raw = df[target].values
    X = df.drop(columns=[target]).values.astype(np.float32)

    classes = np.unique(y_raw)
    y = np.searchsorted(classes, y_raw).astype(np.int64)
    labels = [f"stage {c}" for c in classes]
    return X, y, labels

def load_mushroom():
    path = "agaricus-lepiota.data"
    cols = ["poisonous", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
            "gill-attachment", "gill-spacing", "gill-size", "gill-color",
            "stalk-shape", "stalk-root", "stalk-surface-above-ring",
            "stalk-surface-below-ring", "stalk-color-above-ring",
            "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
            "ring-type", "spore-print-color", "population", "habitat"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Нет файла {path} рядом со скриптом.")
    df = pd.read_csv(path, header=None, names=cols)

    df = df.fillna("?")
    y_raw = df["poisonous"].values
    X = pd.get_dummies(df.drop(columns=["poisonous"])).to_numpy(dtype=np.float32)

    classes = np.unique(y_raw)
    y = np.searchsorted(classes, y_raw).astype(np.int64)
    labels = [f"poisonous={c}" for c in classes]
    return X, y, labels

if __name__ == "__main__":
    X, y, labels = load_hcv()
    run_experiment("HCV Egyptian patients", "hcv", X, y, labels)

    X, y, labels = load_mushroom()
    run_experiment("Mushroom", "mushroom", X, y, labels)