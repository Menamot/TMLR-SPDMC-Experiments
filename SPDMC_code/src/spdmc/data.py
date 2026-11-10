import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build numeric(impute+scale) + categorical(impute+onehot) pipeline."""
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )


def generate_shifted_priors(
    base_prior: np.ndarray,
    n_priors: int,
    method: str = "dirichlet",
    concentration: float = 1e3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate shifted class priors via Dirichlet sampling."""
    if rng is None:
        rng = np.random.default_rng()
    n_classes = len(base_prior)

    if method == "random":
        return rng.dirichlet(np.ones(n_classes), size=n_priors)
    if method == "dirichlet":
        return rng.dirichlet(concentration * np.asarray(base_prior), size=n_priors)
    raise ValueError(f"Unknown method: {method!r}. Use 'dirichlet' or 'random'.")


def generate_test_set(
    X: np.ndarray,
    y: np.ndarray,
    target_prior: np.ndarray,
    target_size: int = 2000,
    method: str = "exact_no_dup",
    noise_std: float = 0.0,
    rng: np.random.Generator = None,
    random_state_base: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a test set with a specified class prior."""
    if rng is None:
        rng = np.random.default_rng()

    classes = np.unique(y)
    m_dict = {k: int(np.sum(y == k)) for k in classes}
    X_sub, y_sub = [], []

    if method == "exact_no_dup":
        N_max_vals = []
        for k, pk in zip(classes, target_prior):
            if pk > 0:
                N_max_vals.append(m_dict[k] / pk)
        N_max = int(np.floor(min(N_max_vals)))

        for i, (k, pk) in enumerate(zip(classes, target_prior)):
            n_samples = max(1, int(np.floor(pk * N_max)))
            n_samples = min(n_samples, m_dict[k])
            Xk, yk = X[y == k], y[y == k]
            rs = int(random_state_base + 97 * int(k))
            X_s, y_s = resample(Xk, yk, n_samples=n_samples, replace=False, random_state=rs)
            if noise_std > 0:
                X_s = X_s + rng.normal(0.0, noise_std, size=X_s.shape)
            X_sub.append(X_s)
            y_sub.append(y_s)

    elif method == "exact_with_dup":
        for i, (k, pk) in enumerate(zip(classes, target_prior)):
            n_samples = max(1, int(round(pk * target_size)))
            Xk, yk = X[y == k], y[y == k]
            rs = int(random_state_base + 97 * int(k))
            X_s, y_s = resample(Xk, yk, n_samples=n_samples, replace=True, random_state=rs)
            if noise_std > 0:
                X_s = X_s + rng.normal(0.0, noise_std, size=X_s.shape)
            X_sub.append(X_s)
            y_sub.append(y_s)

    elif method == "limited_dup":
        for i, (k, pk) in enumerate(zip(classes, target_prior)):
            n_samples = max(1, int(round(pk * target_size)))
            Xk, yk = X[y == k], y[y == k]
            m_k = len(Xk)
            n_max = max(1, m_k)
            n_samples = min(n_samples, n_max)
            rs = int(random_state_base + 97 * int(k))
            X_s, y_s = resample(Xk, yk, n_samples=n_samples, replace=False, random_state=rs)
            if noise_std > 0:
                X_s = X_s + rng.normal(0.0, noise_std, size=X_s.shape)
            X_sub.append(X_s)
            y_sub.append(y_s)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return np.vstack(X_sub), np.concatenate(y_sub)


def prior_shift_strength(
    target_prior: np.ndarray, base_prior: np.ndarray, eps: float = 1e-12,
) -> tuple[float, float]:
    """Compute L1 distance and KL divergence between two priors."""
    p = np.asarray(target_prior)
    p0 = np.asarray(base_prior)
    l1 = float(np.sum(np.abs(p - p0)))
    kl = float(np.sum((p + eps) * np.log((p + eps) / (p0 + eps))))
    return l1, kl
