"""Small synthetic example; no paper dataset is required."""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from dbc import DiscreteBayesianClassifier, compute_class_conditional_risk_with_labels
from dbc.discretizers import CMeansDiscretizer


def main():
    X, y = make_classification(
        n_samples=180, n_features=6, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, weights=[0.6, 0.3, 0.1],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42,
    )
    model = DiscreteBayesianClassifier(
        CMeansDiscretizer(n_clusters=6, random_state=42, use_kmeans=True),
        minmax=True, minmax_n_iter=30,
    ).fit(X_train, y_train)
    prediction = model.predict(X_test)
    risks, classes = compute_class_conditional_risk_with_labels(y_test, prediction)
    print("Learned prior:", model.prior_best_)
    print("Test class risks:", dict(zip(classes.tolist(), risks.tolist())))


if __name__ == "__main__":
    main()
