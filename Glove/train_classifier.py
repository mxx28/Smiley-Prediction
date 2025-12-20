#!/usr/bin/env python3
import numpy as np
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES_PATH = os.path.join(SCRIPT_DIR, "train_features.npy")
LABELS_PATH   = os.path.join(SCRIPT_DIR, "train_labels.npy")

def main():
    print("Loading features and labels...")
    X = np.load(FEATURES_PATH)   # (N, D)
    y = np.load(LABELS_PATH)     # (N,)

    print("Data shape:", X.shape, "Labels shape:", y.shape)

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y,
    )


    hidden_list = [(128,), (128, 64), (256, 128)]
    # hidden_list = [(256, 128)]
    best_acc = -1
    best_cfg = None

    for hidden in hidden_list:
        mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        max_iter=200,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        alpha=1e-4,
        random_state=42, 
        )

        clf = make_pipeline(
            StandardScaler(),
            mlp,
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"hidden={hidden} | val acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_cfg = hidden

    print(f"Best hidden={best_cfg} | val acc={best_acc:.4f}")

    # retrain the best model on the full data
    final_mlp = MLPClassifier(
        hidden_layer_sizes=best_cfg,
        activation="relu",
        max_iter=200,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        alpha=1e-4,
        random_state=42, 
    )

    final_clf = make_pipeline(
        StandardScaler(),
        final_mlp,
    )

    final_clf.fit(X, y)

    model_path = os.path.join(SCRIPT_DIR, "mlp_model.joblib")
    joblib.dump(final_clf, model_path)
    print("Saved final model to", model_path)

if __name__ == "__main__":
    main()

