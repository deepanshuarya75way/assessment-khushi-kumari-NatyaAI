"""
Scalable Training Pipeline for Hand Mudra/Gesture Recognition
=============================================================
Python 3.11 compatible | Windows path support
Features:
  - Reads master_dataset.csv  OR  per-class CSVs  OR  raw dataset folder directly
  - Wrist-relative normalisation + L2 normalisation + key-pair distances
  - Trains RandomForest, SVM, GradientBoosting; picks best via CV
  - RandomizedSearch for hyperparameter tuning
  - Detailed classification report + confusion matrix saved as PNG
  - Saves model.pkl + label_encoder.pkl
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import joblib
import warnings
from pathlib import Path

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ===================================================================
#  PATHS  —  edit only this block if your folder ever changes
# ===================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Output of create_dataset.py  (preferred — loaded if it exists)
MASTER_CSV  = BASE + r"\master_dataset.csv"

# Per-class CSVs folder  (second choice)
CSV_DIR     = BASE + r"\csv_output"

# Where to save outputs
MODEL_OUT   = BASE + r"\model.pkl"
ENCODER_OUT = BASE + r"\label_encoder.pkl"
CM_PNG      = BASE + r"\confusion_matrix.png"

# ===================================================================
#  TRAINING SETTINGS
# ===================================================================
TEST_SIZE              = 0.20
CV_FOLDS               = 5
RANDOM_STATE           = 42
TUNE_HYPERPARAMS       = True
N_ITER_SEARCH          = 30
MIN_SAMPLES_PER_CLASS  = 10   # classes with fewer samples are skipped
# ===================================================================


# ------------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------------
def load_data() -> tuple[np.ndarray, np.ndarray]:
    print("\n" + "=" * 62)
    print("  PATH CHECK")
    print(f"  Master CSV : {MASTER_CSV}")
    print(f"  CSV folder : {CSV_DIR}")
    print("=" * 62)

    # Option 1: master CSV
    if Path(MASTER_CSV).exists():
        print("\n[DATA] Found master_dataset.csv -> loading ...")
        df = pd.read_csv(MASTER_CSV)
        if "label" not in df.columns:
            sys.exit("[ERROR] master_dataset.csv has no 'label' column. "
                     "Re-run create_dataset.py to regenerate it.")
        labels   = df["label"].values
        features = df.drop(columns=["label"]).values.astype(np.float32)
        print(f"[DATA] {len(features)} rows, {features.shape[1]} features")

    # Option 2: per-class CSVs
    elif Path(CSV_DIR).exists():
        csv_files = list(Path(CSV_DIR).glob("*.csv"))
        if not csv_files:
            sys.exit(
                f"[ERROR] '{CSV_DIR}' exists but contains NO .csv files.\n"
                "        Please run  create_dataset.py  first."
            )
        print(f"\n[DATA] Found csv_output folder with {len(csv_files)} CSVs -> loading ...")
        all_X, all_y = [], []
        for csv_file in sorted(csv_files):
            label = csv_file.stem
            df    = pd.read_csv(csv_file)
            if "label" in df.columns:
                df = df.drop(columns=["label"])
            for row in df.values:
                all_X.append(row.astype(np.float32))
                all_y.append(label)
            print(f"       {label:30s}: {len(df)} rows")
        features = np.array(all_X)
        labels   = np.array(all_y)

    # Nothing found
    else:
        sys.exit(
            "\n[ERROR] No data found. Run create_dataset.py first!\n\n"
            "  Expected one of:\n"
            f"    (1) {MASTER_CSV}\n"
            f"    (2) {CSV_DIR}\\*.csv\n\n"
            "  After create_dataset.py finishes you will see those files\n"
            f"  inside:  {BASE}\n"
        )

    # Class distribution summary
    unique_cls, counts = np.unique(labels, return_counts=True)
    print(f"\n[DATA] {len(features)} total samples | {len(unique_cls)} classes")
    print("\n-- Class distribution ----------------------------------")
    for cls, cnt in zip(unique_cls, counts):
        bar = "#" * min(cnt // 5, 40)
        print(f"  {cls:30s}: {cnt:>5}  {bar}")
    print()

    # Drop classes with too few samples
    dropped = [cls for cls, cnt in zip(unique_cls, counts)
               if cnt < MIN_SAMPLES_PER_CLASS]
    if dropped:
        print(f"[WARN] Dropping {len(dropped)} class(es) with "
              f"< {MIN_SAMPLES_PER_CLASS} samples: {dropped}")
        keep     = np.isin(labels, dropped, invert=True)
        features = features[keep]
        labels   = labels[keep]
        print(f"       Remaining: {len(features)} samples | "
              f"{len(np.unique(labels))} classes\n")

    return features, labels


# ------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------------------------------
def engineer_features(X: np.ndarray) -> np.ndarray:
    n      = X.shape[0]
    normed = np.zeros_like(X)
    for i in range(n):
        row  = X[i]
        bx, by = row[0], row[1]
        for j in range(0, len(row), 2):
            normed[i, j]   = row[j]   - bx
            normed[i, j+1] = row[j+1] - by
    normed = normalize(normed, norm="l2")

    key_pairs = [
        (0,  4), (0,  8), (0, 12), (0, 16), (0, 20),
        (4,  8), (8, 12), (5,  9),
    ]
    dists = np.zeros((n, len(key_pairs)), dtype=np.float32)
    for k, (a, b) in enumerate(key_pairs):
        ax, ay = normed[:, a*2], normed[:, a*2+1]
        bx, by = normed[:, b*2], normed[:, b*2+1]
        dists[:, k] = np.sqrt((ax - bx)**2 + (ay - by)**2)

    return np.hstack([normed, dists])


# ------------------------------------------------------------------
# 3. MODEL CANDIDATES
# ------------------------------------------------------------------
def build_candidates() -> dict:
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=None,
                min_samples_split=2, class_weight="balanced",
                n_jobs=-1, random_state=RANDOM_STATE
            ))
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=10, gamma="scale",
                class_weight="balanced", probability=True,
                random_state=RANDOM_STATE
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.1,
                max_depth=5, subsample=0.8,
                random_state=RANDOM_STATE
            ))
        ]),
    }


SEARCH_SPACES = {
    "RandomForest": {
        "clf__n_estimators":      [200, 300, 400, 500],
        "clf__max_depth":         [None, 20, 30, 40],
        "clf__min_samples_split": [2, 4, 6],
        "clf__max_features":      ["sqrt", "log2", 0.3],
    },
    "SVM_RBF": {
        "clf__C":     [1, 5, 10, 50, 100],
        "clf__gamma": ["scale", "auto", 0.01, 0.001],
    },
    "GradientBoosting": {
        "clf__n_estimators":  [200, 300, 400],
        "clf__learning_rate": [0.05, 0.1, 0.15],
        "clf__max_depth":     [3, 5, 7],
        "clf__subsample":     [0.7, 0.8, 0.9],
    },
}


# ------------------------------------------------------------------
# 4. CONFUSION MATRIX
# ------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, class_names):
    cm    = confusion_matrix(y_true, y_pred, labels=class_names)
    fig_w = max(10, len(class_names))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Hand Mudra Recognition")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix -> {CM_PNG}")


# ------------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------------
def main():
    t0 = time.time()

    X_raw, y_str = load_data()

    le          = LabelEncoder()
    y           = le.fit_transform(y_str)
    class_names = list(le.classes_)
    print(f"[DATA] {len(class_names)} classes: {class_names}")

    print("\n[FEAT] Engineering features ...")
    X = engineer_features(X_raw)
    print(f"       {X_raw.shape[1]} raw coords -> {X.shape[1]} engineered features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

    # Auto-adjust CV folds so no fold is smaller than the smallest class
    min_class_train = int(np.bincount(y_train).min())
    cv_folds_used   = max(2, min(CV_FOLDS, min_class_train))
    if cv_folds_used < CV_FOLDS:
        print(f"[CV]  Reducing folds {CV_FOLDS} -> {cv_folds_used} "
              f"(smallest class has {min_class_train} train samples)")

    cv = StratifiedKFold(n_splits=cv_folds_used, shuffle=True,
                         random_state=RANDOM_STATE)

    print(f"\n[CV]  {cv_folds_used}-fold cross-validation on all 3 models ...")
    candidates = build_candidates()
    cv_scores  = {}
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="accuracy", n_jobs=-1)
        cv_scores[name] = scores
        print(f"  {name:20s}: {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(cv_scores, key=lambda k: cv_scores[k].mean())
    print(f"\n[CV]  Best model -> {best_name}")

    best_pipe = candidates[best_name]
    if TUNE_HYPERPARAMS and best_name in SEARCH_SPACES:
        print(f"[TUNE] RandomizedSearch ({N_ITER_SEARCH} combos) ...")
        search = RandomizedSearchCV(
            best_pipe,
            param_distributions=SEARCH_SPACES[best_name],
            n_iter=N_ITER_SEARCH,
            cv=cv, scoring="accuracy",
            n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        print(f"  Best params : {search.best_params_}")
        print(f"  CV accuracy : {search.best_score_:.4f}")
    else:
        print("[TRAIN] Fitting on full training set ...")
        best_pipe.fit(X_train, y_train)

    y_pred = best_pipe.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  Final Test Accuracy : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"{'='*55}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    save_confusion_matrix(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        class_names
    )

    Path(MODEL_OUT).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_OUT)
    joblib.dump(le,        ENCODER_OUT)
    print(f"\n[SAVE] model.pkl        -> {MODEL_OUT}")
    print(f"[SAVE] label_encoder.pkl-> {ENCODER_OUT}")
    print(f"[DONE] Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()