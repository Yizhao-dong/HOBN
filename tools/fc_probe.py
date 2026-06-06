import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def main():
    parser = argparse.ArgumentParser(description="Run simple full-FC linear baselines.")
    parser.add_argument("--data-path", default=os.environ.get("HOBN_DATA_PATH", "data/abide.npy"))
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True).item()
    y = data["label"].astype(int)
    corr = data["corr"].astype("float32")
    iu = np.triu_indices(corr.shape[1], k=1)
    x = corr[:, iu[0], iu[1]]
    print("features", x.shape, "labels", np.unique(y, return_counts=True))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "ridge_a10": RidgeClassifier(alpha=10.0),
        "ridge_a100": RidgeClassifier(alpha=100.0),
        "svm_c001": LinearSVC(C=0.01, dual=False, max_iter=5000),
        "svm_c01": LinearSVC(C=0.1, dual=False, max_iter=5000),
        "log_c001": LogisticRegression(C=0.01, max_iter=5000, solver="liblinear"),
        "log_c01": LogisticRegression(C=0.1, max_iter=5000, solver="liblinear"),
    }
    for name, clf in models.items():
        acc = []
        for tr, te in skf.split(x, y):
            model = make_pipeline(StandardScaler(), clf)
            model.fit(x[tr], y[tr])
            pred = model.predict(x[te])
            acc.append(accuracy_score(y[te], pred))
        print(name, [round(a, 4) for a in acc], "max", round(max(acc), 4), "mean", round(float(np.mean(acc)), 4))


if __name__ == "__main__":
    main()
