"""Initialization and step-size study using Algorithm 1's dispersion criterion."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import dbc.utils as dbc_utils
from spdmc.models import build_model_from_params
from .common import ROOT, SOFT, load_data, settings
from .runtime import cached_training_risks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["diabetes", "cifar10"], required=True)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1, 2, 4])
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])
    parser.add_argument("--starts", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--fold", type=int, help="Run only this zero-based fold")
    parser.add_argument("--representations", nargs="+", choices=SOFT, default=list(SOFT))
    args = parser.parse_args()
    if args.starts < 1 or args.max_iter < 1:
        parser.error("--starts and --max-iter must be positive")
    if any(not np.isfinite(a) or a <= 0 for a in args.alphas + args.tolerances):
        parser.error("Step sizes and tolerances must be finite and positive")
    if args.fold is not None and not 0 <= args.fold < 5:
        parser.error("--fold must be between 0 and 4")
    cfg = settings(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps(vars(args), indent=2))
    X, y, _ = load_data(args.dataset, args.data_dir)
    k = len(cfg["loss"])
    starts = np.random.default_rng(np.random.SeedSequence([20261012, k])).dirichlet(
        np.ones(k), size=args.starts)
    np.save(output / "initial_priors.npy", starts)
    cv = StratifiedKFold(5, shuffle=True, random_state=42 + 123)
    with (output / "attainment.jsonl").open("w") as file, threadpool_limits(limits=1):
        for fold, (train, _) in enumerate(cv.split(X, y)):
            if args.fold is not None and args.fold != fold:
                continue
            for representation in args.representations:
                params = cfg["params"]["DMC-" + representation]
                model = build_model_from_params("DBC-" + representation, params,
                                                loss_function=np.asarray(cfg["loss"]), random_seed=42)
                model.fit(X[train], y[train])
                def evaluate(prior):
                    return dbc_utils.compute_class_conditional_risk(
                        y[train], model.loss_function_, model.p_hat_, model.membership_train_,
                        model.prior_, prior,
                    )
                with cached_training_risks():
                    for alpha in args.alphas:
                        for initial, prior in enumerate(starts):
                            final, risks = dbc_utils.projected_prior_update(
                                evaluate, prior, alpha=alpha, n_iter=args.max_iter,
                                eps=min(args.tolerances),
                            )
                            dispersion = np.sum((risks - risks.mean(axis=1, keepdims=True)) ** 2, axis=1)
                            path = output / f"{representation}_fold{fold}_start{initial}_alpha{alpha:g}.npz"
                            np.savez_compressed(path, risks=risks, dispersion=dispersion,
                                                initial_prior=prior, final_prior=final)
                            for tolerance in args.tolerances:
                                hits = np.flatnonzero(dispersion[1:] < tolerance) + 1
                                step = int(hits[0]) if len(hits) else len(risks) - 1
                                row = dict(dataset=args.dataset, representation=representation,
                                           fold=fold, initial=initial, alpha=alpha, tolerance=tolerance,
                                           reached=bool(len(hits)), iteration=step,
                                           max_class_risk=float(risks[step].max()),
                                           dispersion=float(dispersion[step]))
                                file.write(json.dumps(row) + "\n")
                            file.flush()
                        print(f"{representation}: fold={fold}, alpha={alpha:g} complete", flush=True)


if __name__ == "__main__":
    main()
