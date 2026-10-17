import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml.url_features import JSON_PATH, load_split, verify

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "phishing_model.pkl"
OUTPUTS = ROOT / "outputs"
POSITIVE = "phishing"


def save_class_distribution(y_train, y_test):
    counts = pd.DataFrame({"train": y_train.value_counts(), "test": y_test.value_counts()})
    ax = counts.plot(kind="bar", figsize=(6, 4), color=["#4C72B0", "#55A868"])
    ax.set_title("Class distribution")
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "class_distribution.png", dpi=150)
    plt.close()


def save_confusion_matrix(cm, labels):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion matrix (test split)")
    plt.ylabel("actual")
    plt.xlabel("predicted")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=150)
    plt.close()


def save_feature_importance(clf, cols):
    imp = pd.Series(clf.feature_importances_, index=cols).sort_values()
    ax = imp.plot(kind="barh", figsize=(8, 10))
    ax.set_title("Random Forest feature importance")
    ax.set_xlabel("importance")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "feature_importance.png", dpi=150)
    plt.close()


def main():
    train = load_split("train")
    test = load_split("test")

    kept, dropped = verify(train, write_json=True)
    if dropped:
        raise SystemExit(f"aborting: features failed verification: {', '.join(dropped)}")

    cols = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    X_train = train[cols]
    y_train = train["status"]
    X_test = test[cols]
    y_test = test["status"]

    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = ["legitimate", POSITIVE]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=POSITIVE, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=POSITIVE, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=POSITIVE, zero_division=0)

    print("\n=== test split results (positive class = phishing)")
    print(f"accuracy : {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall   : {rec:.4f}")
    print(f"f1       : {f1:.4f}")
    print(f"confusion matrix rows=actual cols=predicted {labels}:\n{cm}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    OUTPUTS.mkdir(exist_ok=True)
    save_class_distribution(y_train, y_test)
    save_confusion_matrix(cm, labels)
    save_feature_importance(clf, cols)
    print(f"saved {MODEL_PATH.name} and 3 plots in {OUTPUTS.name}/")


if __name__ == "__main__":
    main()
