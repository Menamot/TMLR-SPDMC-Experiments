"""Summarize classifier comparisons or plot a saved multiclass trajectory."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="metrics.jsonl or trajectory .npz")
    args = parser.parse_args()
    if args.input.suffix == ".npz":
        with np.load(args.input) as data:
            risks, dispersion = data["risks"], data["dispersion"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(risks)
        axes[0].set(xlabel="Update", ylabel="Class-conditional risk")
        axes[1].plot(dispersion)
        axes[1].set(xlabel="Update", ylabel="Squared risk dispersion")
        fig.tight_layout()
        fig.savefig(args.input.with_suffix(".pdf"))
        plt.close(fig)
        return
    frame = pd.DataFrame([json.loads(line) for line in args.input.read_text().splitlines() if line.strip()])
    columns = ["global_risk", "max_class_risk", "class_risk_range", "f1_macro", "auc"]
    summary = frame.groupby(["scenario", "split", "model"])[columns].agg(["mean", "std"])
    summary.to_csv(args.input.with_name("summary.csv"))
    for (scenario, split), subset in frame.groupby(["scenario", "split"]):
        grouped = subset.groupby("model")["class_risk_range"]
        means, stds = grouped.mean(), grouped.std().fillna(0)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(means.index, means.values, xerr=stds.values)
        ax.set(xlabel="Class-conditional risk range (mean ± SD)", title=f"{scenario}: {split}")
        fig.tight_layout()
        fig.savefig(args.input.with_name(f"risk_range_{scenario}_{split}.pdf"))
        plt.close(fig)


if __name__ == "__main__":
    main()
