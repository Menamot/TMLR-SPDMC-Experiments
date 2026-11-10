"""Dataset loading, fixed settings, and metrics shared by the experiments."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from spdmc.data import make_preprocessor
from spdmc.metrics import compute_all_metrics, compute_class_conditional_risk

ROOT = Path(__file__).resolve().parents[1]
SOFT = ("CMeans", "DiagGMM", "DiagGMM-ClassWise", "Tree-DiagGMM")


def settings(dataset):
    return json.loads((ROOT / "configs" / f"{dataset}.json").read_text())


def load_data(dataset, data_dir):
    """Retain the original scripts' preprocessing before cross-validation."""
    cfg = settings(dataset)
    path = Path(data_dir) / cfg["file"]
    if dataset == "cifar10":
        with np.load(path) as data:
            features, labels = data["X_train"], data["y_train"]
        X = PCA(n_components=50, random_state=0).fit_transform(features)
    else:
        if dataset == "scania":
            data = loadmat(path)
            features = pd.DataFrame(data["X"])
            labels = data["YR"].ravel()
        else:
            data = pd.read_csv(path)
            if dataset == "diabetes":
                features, labels = data.iloc[:, :29], data.iloc[:, 29]
            else:
                features, labels = data.drop(columns="target"), data["target"]
        X = make_preprocessor(features).fit_transform(features)
    encoder = LabelEncoder().fit(labels)
    return np.asarray(X), encoder.transform(labels), encoder.classes_


def metrics(y, prediction, scores, loss):
    k = len(loss)
    risks, _ = compute_class_conditional_risk(y, prediction, loss, np.arange(k))
    result = compute_all_metrics(risks, np.bincount(y, minlength=k) / len(y))
    aucs = [float(roc_auc_score(y == c, scores[:, c]))
            if len(np.unique(y == c)) == 2 else None for c in range(k)]
    result.update(
        class_risks=risks.tolist(),
        f1_macro=float(f1_score(y, prediction, average="macro", labels=np.arange(k), zero_division=0)),
        f1_positive=float(f1_score(y, prediction, pos_label=1, zero_division=0)) if k == 2 else None,
        auc=aucs[1] if k == 2 else (float(np.mean(aucs)) if None not in aucs else None),
        n_samples=len(y),
    )
    return result
