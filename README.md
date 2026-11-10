# Soft Probabilistic Discrete Minimax Classifier

Code accompanying **Soft Probabilistic Discrete Minimax Classifier**, by Wenlong Chen, Cyprien Gilet, Benjamin Quost, and Sébastien Destercke.

SPDMC fits a probabilistic profile representation and adjusts the prediction prior to balance empirical class-conditional risks. Binary problems use bisection. Multiclass problems use the projected update

```text
g = class_risks - mean(class_risks)
prior = project_to_simplex(prior + alpha / (step + 1) * g)
```

## Installation

Use Python 3.11 or newer. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[experiments]"
python examples/quickstart.py
```

## Classifier

```python
from dbc import DiscreteBayesianClassifier
from dbc.discretizers import CMeansDiscretizer

model = DiscreteBayesianClassifier(
    CMeansDiscretizer(n_clusters=8, random_state=42, use_kmeans=True),
    minmax=True,
    minmax_alpha=2.0,
    minmax_eps=1e-3,
    minmax_n_iter=1000,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Set `minmax=False` for the fixed-prior SPDBC. Available representations include CMeans, diagonal GMM, classwise diagonal GMM, and tree-based diagonal GMM. Hard KMeans DMC and randomized SDMC are also included.

Custom losses use **rows for true classes and columns for predictions**, ordered by the fitted label encoder. For Scania the code uses `[[0, 10], [500, 0]]`; transpose a matrix written with predictions in rows before passing it to the code. Membership matrices have shape `(n_profiles, n_samples)`.

## Data and comparisons

Place the inputs below in `data/`, or pass their directory with `--data-dir`. Raw datasets and saved experiment results are separate from this code release.

| Dataset | Input file | Expected content |
| --- | --- | --- |
| Bank | `bank.csv` | Feature columns and the `target` column (`no` / `yes`) |
| Diabetes | `diabetes_dataset.csv` | First 29 columns are features; column 30 is `diabetes_stage` |
| Scania | `scaniaTrucks_data.mat` | Arrays `X` and `YR` |
| CIFAR-10 | `cifar10_vgg16_gap_features.npz` | VGG16 features `X_train` and labels `y_train` |

```bash
python -m experiments.run_comparison --dataset bank --output outputs/bank
python -m experiments.run_comparison --dataset diabetes --output outputs/diabetes
python -m experiments.run_comparison --dataset scania --output outputs/scania
python -m experiments.run_comparison --dataset cifar10 --output outputs/cifar10
python -m experiments.plot_results outputs/bank/metrics.jsonl
```

The defaults use five folds, 20 repetitions for the original-prior scenario, and five repetitions with four sampled priors per fold for prior shift. Shifted sets are sampled without replacement within each class, with the largest feasible total size followed by rounding and a minimum of one observation per class. Their size varies with the target prior.

To run a subset, use `--scenario original`, `--repeats 1`, `--fold 0`, or `--models DMC-DiagGMM`. Each command requires a new output directory and writes its settings and metrics there. 

## Multiclass update study

```bash
python -m experiments.run_multiclass --dataset diabetes --output outputs/diabetes_multiclass
python -m experiments.run_multiclass --dataset cifar10 --output outputs/cifar10_multiclass
```

Defaults follow the appendix: four representations, five folds, 30 shared Dirichlet initial priors, learning-rate scales 0.5, 1, 2, and 4, tolerances `1e-3`, `1e-4`, and `1e-5`, and at most 5000 updates. Saved trajectories include the initial risk and every evaluated update; attainment is counted from update 1. Plot a trajectory with `python -m experiments.plot_results path/to/trajectory.npz`.

The retained synthetic illustration code is available through `python -m experiments.plot_illustrations`; it writes figures to `outputs/illustrations/`.

## CIFAR-10 features

To produce the feature file from the official CIFAR-10 Python batches:

```bash
python -m pip install -e ".[cifar]"
python -m experiments.extract_cifar_features --data-dir data/cifar-10-batches-py --output data/cifar10_vgg16_gap_features.npz
```

The script uses ImageNet-pretrained VGG16 and global average pooling. Torchvision downloads the pretrained weights on first use. The comparison loader reduces the training feature vectors to 50 dimensions using PCA.

## Acknowledgements

This work was supported by research under grant ANR-24-CE23-0847-01 (project DMC).

## License

MIT; see [LICENSE](LICENSE).
