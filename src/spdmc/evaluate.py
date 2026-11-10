import numpy as np
from sklearn.model_selection import StratifiedKFold

from .data import (
    generate_shifted_priors,
    generate_test_set,
)
from .metrics import (
    compute_class_conditional_risk,
    compute_all_metrics,
)
from .models import build_model_from_params
from .storage import ExperimentDB


def run_stage2_evaluation(
    best_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    loss_function: np.ndarray,
    db: ExperimentDB,
    dataset: str,
    tag: str = None,
    description: str = None,
    n_folds: int = 5,
    n_repeats: int = 20,
    n_priors: int = 10,
    prior_sampling_method: str = "dirichlet",
    prior_concentration: float = 1e8,
    testset_target_size: int = 2000,
    testset_method: str = "exact_no_dup",
    feature_noise_std: float = 0.0,
    random_seed: int = 42,
    minmax_eps: float = 1e-3,
    batch_size: int = 200,
) -> int:
    """Run Stage 2 evaluation with prior-shifted test sets. Returns experiment_id."""
    X = np.asarray(X)
    y = np.asarray(y)

    exp_id = db.create_experiment(
        dataset=dataset,
        stage="evaluation",
        tag=tag,
        description=description,
        n_folds=n_folds,
        n_repeats=n_repeats,
        n_priors=n_priors,
        prior_sampling_method=prior_sampling_method,
        prior_concentration=prior_concentration,
        feature_noise_std=feature_noise_std,
        random_seed=random_seed,
    )
    db.save_loss_function(exp_id, loss_function)

    rows_buffer = []

    def flush():
        if rows_buffer:
            db.save_results_batch(rows_buffer)
            rows_buffer.clear()

    for repeat_idx in range(n_repeats):
        outer_seed = random_seed + 10_000 * repeat_idx + 123
        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=outer_seed
        )

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            classes_order, counts = np.unique(y_train, return_counts=True)
            prior_train = counts / counts.sum()

            rng_priors = np.random.default_rng(
                seed=200_000 + 1_000 * repeat_idx + fold_idx
            )
            priors_list = generate_shifted_priors(
                base_prior=prior_train,
                n_priors=n_priors,
                method=prior_sampling_method,
                concentration=prior_concentration,
                rng=rng_priors,
            )

            rng_tests = np.random.default_rng(
                seed=300_000 + 1_000 * repeat_idx + fold_idx
            )
            test_sets = []
            for pi_idx, target_prior in enumerate(priors_list):
                rs_base = (
                    400_000 + 10_000 * repeat_idx + 100 * fold_idx + pi_idx * 10_000
                )
                X_t, y_t = generate_test_set(
                    X=X_test,
                    y=y_test,
                    target_prior=target_prior,
                    target_size=testset_target_size,
                    method=testset_method,
                    noise_std=feature_noise_std,
                    rng=rng_tests,
                    random_state_base=rs_base,
                )
                test_sets.append((X_t, y_t))

            for model_name, params in best_params.items():
                model = build_model_from_params(
                    model_name=model_name,
                    params=params,
                    minmax_eps=minmax_eps,
                    random_seed=random_seed,
                    loss_function=loss_function,

                )
                model.fit(X_train, y_train)

                # Train row
                y_pred_train = model.predict(X_train)
                R_train, _ = compute_class_conditional_risk(
                    y_train, y_pred_train, loss_function, classes=classes_order
                )
                metrics_train = compute_all_metrics(R_train, prior_train)
                class_risks_train = [
                    {
                        "class_label": int(c),
                        "class_risk": float(r),
                        "target_prior": float(p),
                    }
                    for c, r, p in zip(classes_order, R_train, prior_train)
                ]
                rows_buffer.append(
                    {
                        "experiment_id": exp_id,
                        "model_name": model_name,
                        "repeat_idx": repeat_idx,
                        "fold_idx": fold_idx,
                        "data_split": "train",
                        "prior_idx": None,
                        **metrics_train,
                        "class_risks": class_risks_train,
                    }
                )

                # Test rows
                for pi_idx, ((X_t, y_t), target_prior) in enumerate(
                    zip(test_sets, priors_list)
                ):
                    y_pred_test = model.predict(X_t)
                    R_test, _ = compute_class_conditional_risk(
                        y_t, y_pred_test, loss_function, classes=classes_order
                    )
                    counts_test = np.array(
                        [(y_t == c).sum() for c in classes_order], dtype=float
                    )
                    actual_prior = counts_test / max(1.0, float(len(y_t)))
                    metrics_test = compute_all_metrics(R_test, actual_prior)
                    class_risks_test = [
                        {
                            "class_label": int(c),
                            "class_risk": float(r),
                            "target_prior": float(p),
                        }
                        for c, r, p in zip(classes_order, R_test, target_prior)
                    ]
                    rows_buffer.append(
                        {
                            "experiment_id": exp_id,
                            "model_name": model_name,
                            "repeat_idx": repeat_idx,
                            "fold_idx": fold_idx,
                            "data_split": "test",
                            "prior_idx": pi_idx,
                            **metrics_test,
                            "class_risks": class_risks_test,
                        }
                    )

                if len(rows_buffer) >= batch_size:
                    flush()

            flush()
            print(
                f"[Stage-2] Repeat {repeat_idx+1}/{n_repeats} | Fold {fold_idx+1}/{n_folds} done."
            )

    flush()
    return exp_id
