"""Run the fixed-parameter classifiers and prior-shift comparisons."""
import argparse
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from dbc.discretizers import KMeansDiscretizer
from dbc.softmin import SoftminDiscreteMinimaxClassifier, sample_action_indices
from spdmc.data import generate_shifted_priors, generate_test_set
from spdmc.models import build_model_from_params
from .auc_scores import normalize_class_support
from .common import ROOT, load_data, metrics, settings
from .runtime import cached_training_risks, prediction_features


def fit_sdmc(X, y, cfg, repeat, fold):
    """Choose lambda by three-fold training-only validation, with paired draws."""
    def create(value):
        return SoftminDiscreteMinimaxClassifier(
            KMeansDiscretizer(**cfg["params"]["DMC-KMeans"], random_state=42),
            softmin_lambda=value, loss_matrix=np.asarray(cfg["loss"]),
            max_iter=cfg["sdmc_max_iter"], tol=cfg["sdmc_tol"], random_state=42,
        )
    candidates = {value: [] for value in cfg["sdmc_lambdas"]}
    cv = StratifiedKFold(3, shuffle=True, random_state=600000 + 1000 * repeat + fold)
    for inner, (train, validation) in enumerate(cv.split(X, y)):
        model = create(120)._fit_partition(X[train], y[train])
        profiles = model._membership(X[validation]).argmax(axis=0)
        uniforms = np.random.default_rng(np.random.SeedSequence(
            [42, 700000, repeat, fold, inner])).random(len(validation))
        for value in candidates:
            model.set_params(softmin_lambda=value)._fit_prior()
            scores = model.action_probabilities_[:, profiles].T
            prediction = sample_action_indices(scores, uniforms=uniforms)
            candidates[value].append(metrics(y[validation], prediction, scores,
                                             np.asarray(cfg["loss"]))["max_class_risk"])
    selected = min(candidates, key=lambda value: (np.mean(candidates[value]), value))
    return create(selected).fit(X, y), selected


def evaluate_model(name, X, y, train, sampled, cfg, repeat, fold):
    """Fit once and obtain predictions and AUC scores for all observations."""
    loss = np.asarray(cfg["loss"])
    if name == "SDMC-KMeans":
        model, selected = fit_sdmc(X[train], y[train], cfg, repeat, fold)
        scores = model.predict_proba(X)
        prediction = sample_action_indices(scores, random_state=np.random.SeedSequence(
            [42, 800000, repeat, fold]))
        return prediction, scores, {"selected_lambda": selected}
    base = name.removeprefix("Sampling-")
    fit_indices = sampled if name.startswith("Sampling-") else train
    model = build_model_from_params(base, cfg["params"][name], loss_function=loss,
                                    random_seed=cfg["model_seed"], minmax_eps=cfg["minimax_eps"])
    if base.startswith(("DMC-", "DBC-")):
        model.set_params(minmax_alpha=cfg["alpha"], minmax_n_iter=cfg["max_iter"])
    with cached_training_risks():
        model.fit(X[fit_indices], y[fit_indices])
    if base.startswith(("DMC-", "DBC-")):
        prior = np.asarray(model.prior_best_ if model.minmax else model.prior_).ravel()
        support = prediction_features(model, X) * prior
        prediction = model.label_encoder_.inverse_transform((support @ loss).argmin(axis=1))
        scores = normalize_class_support(support)
        extra = {"prediction_prior": prior.tolist()}
    else:
        prediction, scores, extra = model.predict(X), model.predict_proba(X), {}
    return prediction, scores, extra


def evaluation_sets(y, train, heldout, repeat, fold, scenario, cfg):
    k = len(cfg["loss"])
    prior = np.bincount(y[train], minlength=k) / len(train)
    yield "train", -1, train, prior
    shifted = scenario == "shift"
    targets = generate_shifted_priors(
        prior, cfg["shift_priors"] if shifted else 1,
        "random" if shifted else "dirichlet", 1e8,
        np.random.default_rng(200000 + 1000 * repeat + fold),
    )
    for index, target in enumerate(targets):
        selected, _ = generate_test_set(
            heldout[:, None], y[heldout], target, method="exact_no_dup",
            random_state_base=400000 + 10000 * repeat + 100 * fold + 10000 * index,
        )
        yield "test", index, selected.ravel(), target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["bank", "diabetes", "scania", "cifar10"], required=True)
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--output", type=str, required=True, help="New output directory")
    parser.add_argument("--scenario", choices=["original", "shift", "both"], default="both")
    parser.add_argument("--repeats", type=int, help="Override the configured repeat count")
    parser.add_argument("--fold", type=int, help="Run only this zero-based fold")
    parser.add_argument("--models", nargs="+", help="Subset of model names in the configuration")
    args = parser.parse_args()
    cfg = settings(args.dataset)
    names = args.models or list(cfg["params"]) + ["SDMC-KMeans"]
    unknown = set(names) - (set(cfg["params"]) | {"SDMC-KMeans"})
    if unknown:
        parser.error(f"Unknown models: {sorted(unknown)}")
    if args.repeats is not None and args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.fold is not None and not 0 <= args.fold < cfg["folds"]:
        parser.error("--fold is outside the configured folds")
    from pathlib import Path
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps({"config": cfg, "arguments": vars(args)}, indent=2))
    X, y, labels = load_data(args.dataset, args.data_dir)
    (output / "classes.json").write_text(json.dumps(labels.tolist()))
    scenarios = ["original", "shift"] if args.scenario == "both" else [args.scenario]
    repeats = {s: args.repeats or cfg[f"{s}_repeats"] for s in scenarios}
    with (output / "metrics.jsonl").open("w") as file, threadpool_limits(limits=1):
        for repeat in range(max(repeats.values())):
            cv = StratifiedKFold(cfg["folds"], shuffle=True, random_state=42 + 10000 * repeat + 123)
            for fold, (train, heldout) in enumerate(cv.split(X, y)):
                if args.fold is not None and fold != args.fold:
                    continue
                weights = 1 / np.bincount(y[train])[y[train]]
                sampled = np.random.default_rng(500000 + 1000 * repeat + fold).choice(
                    train, len(train), replace=True, p=weights / weights.sum())
                active = [s for s in scenarios if repeat < repeats[s]]
                tasks = [(s, *item) for s in active for item in
                         evaluation_sets(y, train, heldout, repeat, fold, s, cfg)]
                for name in names:
                    print(f"{args.dataset}: repeat={repeat}, fold={fold}, {name}", flush=True)
                    prediction, scores, extra = evaluate_model(name, X, y, train, sampled, cfg, repeat, fold)
                    for scenario, split, index, rows, target in tasks:
                        record = dict(dataset=args.dataset, model=name, repeat=repeat, fold=fold,
                                      scenario=scenario, split=split, prior=index,
                                      target_prior=target.tolist(), **extra,
                                      **metrics(y[rows], prediction[rows], scores[rows], np.asarray(cfg["loss"])))
                        file.write(json.dumps(record, allow_nan=False) + "\n")
                    file.flush()


if __name__ == "__main__":
    main()
