import tempfile

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from .metrics import compute_class_conditional_risk
from .models import build_model_from_params
from .storage import ExperimentDB


def _make_objective(
    X,
    y,
    loss_function,
    cv_splits,
    model_name,
    param_search_space,
    scorer_fn,
    random_seed,
    minmax_eps,
):
    def objective(trial):
        params = param_search_space(trial)
        fold_scores = []
        for step, (tr_idx, va_idx) in enumerate(cv_splits):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]
            model = build_model_from_params(
                model_name=model_name,
                params=params,
                minmax_eps=minmax_eps,
                random_seed=random_seed,
                loss_function=loss_function,
            )
            model.fit(X_tr, y_tr)
            score = -float(scorer_fn(model, X_va, y_va))
            fold_scores.append(score)
            trial.report(np.mean(fold_scores), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    return objective


def run_hyperparam_search(
    search_space: dict,
    X: np.ndarray,
    y: np.ndarray,
    loss_function: np.ndarray,
    db: ExperimentDB,
    dataset: str,
    tag: str = None,
    description: str = None,
    n_folds: int = 4,
    n_repeats: int = 1,
    n_trials: dict | int = 50,
    scorer: str = "max_class_risk",
    random_seed: int = 42,
    minmax_eps: float = 3e-2,
) -> dict:
    """Run Optuna hyperparameter search for all models in search_space.

    Uses a temporary Optuna SQLite database. After completion, extracts
    best params and writes them to the unified ExperimentDB.

    Returns {model_name: best_params_dict}.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    def _max_rk_scorer(estimator, X_eval, y_eval):
        y_pred = estimator.predict(X_eval)
        R, _ = compute_class_conditional_risk(y_eval, y_pred, loss_function)
        return -float(np.max(R))

    if scorer in (None, "max_class_risk"):
        scorer_fn = _max_rk_scorer
    elif callable(scorer):
        scorer_fn = scorer
    else:
        raise ValueError(
            "scorer must be None, 'max_class_risk', or a callable "
            "with signature scorer(estimator, X_eval, y_eval)"
        )

    cv_splits = []
    for repeat_idx in range(n_repeats):
        seed = random_seed + 1000 * repeat_idx + 11
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_splits.extend(list(splitter.split(X, y)))

    if isinstance(n_trials, int):
        n_trials_dict = {name: n_trials for name in search_space}
    else:
        n_trials_dict = n_trials

    exp_id = db.create_experiment(
        dataset=dataset,
        stage="hyperparam",
        tag=tag,
        description=description,
        n_folds=n_folds,
        n_repeats=n_repeats,
        random_seed=random_seed,
    )
    db.save_loss_function(exp_id, loss_function)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best_params_map = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        optuna_db = f"sqlite:///{tmpdir}/optuna_temp.db"
        for model_name, param_fn in search_space.items():
            study = optuna.create_study(
                study_name=f"tune_{model_name}_{dataset}",
                storage=optuna_db,
                load_if_exists=True,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=random_seed),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
            )
            obj = _make_objective(
                X=X,
                y=y,
                loss_function=loss_function,
                cv_splits=cv_splits,
                model_name=model_name,
                param_search_space=param_fn,
                scorer_fn=scorer_fn,
                random_seed=random_seed,
                minmax_eps=minmax_eps,
            )
            trials = n_trials_dict.get(model_name, 50)
            study.optimize(obj, n_trials=trials, n_jobs=1)
            best_params_map[model_name] = dict(study.best_params)
            db.save_hyperparams(
                experiment_id=exp_id,
                model_name=model_name,
                n_trials=trials,
                best_score=float(study.best_value),
                params=dict(study.best_params),
            )
            print(
                f"[Hyperparam] {model_name}: best={study.best_value:.4f}, params={study.best_params}"
            )

    return best_params_map
