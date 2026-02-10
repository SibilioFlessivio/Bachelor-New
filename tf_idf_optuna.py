# -*- coding: utf-8 -*-
"""
Optuna hyperparameter search for TF-IDF + linear classifiers (LogReg / LinearSVC)
with spaCy-based German stopword handling.

"""

import os
import json
import argparse
from datetime import datetime
import numpy as np
import optuna
from datasets import load_from_disk
from joblib import Memory, dump
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
import spacy
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=UserWarning, module="optuna.distributions")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.linear_model")

os.environ.pop("OPTUNA_STORAGE", None)


# ---------------------------
# Load spaCy German stopwords
try:
    nlp = spacy.load("de_core_news_sm")
    SPACY_STOPWORDS = {w.lower() for w in nlp.Defaults.stop_words}
    print(f"? Loaded {len(SPACY_STOPWORDS)} spaCy German stopwords.")
except Exception as e:
    print(f"?? Could not load spaCy German model: {e}")
    SPACY_STOPWORDS = set()


# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", type=str, required=True,
                   help="Path to HF datasets dir (load_from_disk)")
    p.add_argument("--text-col", type=str, default="speech",
                   help="Column with raw text")
    p.add_argument("--label-col", type=str, default="party",
                   help="Column with labels (int encoded)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--n-jobs-inner", type=int, default=-1)
    p.add_argument("--n-jobs-trials", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--cache-dir", type=str, default="cache")
    return p.parse_args()


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def get_data(dataset_path, text_col, label_col, test_size=0.2, seed=42):
    ds = load_from_disk(dataset_path)
    print("label_col: ", label_col)
    ds = ds.class_encode_column("party")
    print(ds.features)
    if "train" in ds and "test" in ds:
        train, test = ds["train"], ds["test"]
    else:
        split = ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column=label_col)
        train, test = split["train"], split["test"]
    for col in (text_col, label_col):
        if col not in train.column_names:
            raise ValueError(f"Column '{col}' not found. Available: {train.column_names}")

    X_train = train[text_col]
    y_train = train[label_col]
    X_test = test[text_col]
    y_test = test[label_col]
    return X_train, y_train, X_test, y_test


def build_pipeline(memory):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ], memory=memory)
    return pipe


def objective_factory(X, y, cv, n_jobs_inner, cache_dir):
    memory = Memory(location=cache_dir, verbose=0)
    pipe = build_pipeline(memory)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial):
        # --- TF-IDF Parameters ---
        ngram_choices = [[1, 1], [1, 2], [1, 3]]
        ngram_range = tuple(trial.suggest_categorical("tfidf__ngram_range", ngram_choices))
        max_features = trial.suggest_int("tfidf__max_features", 20000, 80000, step=10000)
        min_df_mode = trial.suggest_categorical("tfidf__min_df_mode", ["int", "frac"])
        if min_df_mode == "int":
            min_df = trial.suggest_int("tfidf__min_df", 1, 5)
        else:
            min_df = trial.suggest_float("tfidf__min_df_frac", 0.0005, 0.005, step=0.0005)
        max_df = trial.suggest_float("tfidf__max_df", 0.8, 1.0)
        sublinear_tf = trial.suggest_categorical("tfidf__sublinear_tf", [True, False])
        smooth_idf = trial.suggest_categorical("tfidf__smooth_idf", [True, False])
        lowercase = trial.suggest_categorical("tfidf__lowercase", [True, False])
        strip_accents = trial.suggest_categorical("tfidf__strip_accents", [None, "unicode"])
        norm = trial.suggest_categorical("tfidf__norm", ["l2", "l1", None])
        stopword_source = trial.suggest_categorical("tfidf__stopword_source", ["none", "spacy"])

        stop_words = list(SPACY_STOPWORDS) if (stopword_source == "spacy") else None

        pipe.set_params(
            tfidf__analyzer="word",
            tfidf__ngram_range=ngram_range,
            tfidf__max_features=max_features,
            tfidf__min_df=min_df,
            tfidf__max_df=max_df,
            tfidf__sublinear_tf=sublinear_tf,
            tfidf__smooth_idf=smooth_idf,
            tfidf__lowercase=lowercase,
            tfidf__strip_accents=strip_accents,
            tfidf__norm=norm,
            tfidf__stop_words=stop_words,
            tfidf__use_idf=True
        )

        # --- Classifier choice ---
        clf_choice = trial.suggest_categorical("clf", ["logreg", "linearsvc"])
        if clf_choice == "logreg":
            C = trial.suggest_float("logreg__C", 1e-2, 1e+2, log=True)
            solver = trial.suggest_categorical("logreg__solver", ["lbfgs", "liblinear", "saga"])
            multi_class = "auto" if solver in ("lbfgs", "saga") else "ovr"
            class_weight = trial.suggest_categorical("logreg__class_weight", [None, "balanced"])
            pipe.set_params(
                clf=LogisticRegression(
                    C=C, solver=solver, penalty="l2",
                    class_weight=class_weight, max_iter=1000,
                    multi_class=multi_class
                )
            )
        else:
            C = trial.suggest_float("linearsvc__C", 1e-3, 1e+1, log=True)
            class_weight = trial.suggest_categorical("linearsvc__class_weight", [None, "balanced"])
            pipe.set_params(
                clf=LinearSVC(C=C, class_weight=class_weight, loss="squared_hinge", dual=True)
            )

        # --- CV evaluation ---
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=n_jobs_inner)
        return float(np.mean(scores))

    return objective


def main():
    args = parse_args()
    ensure_dirs(args.output_dir, args.cache_dir)

    X_train, y_train, X_test, y_test = get_data(
        args.dataset_path, args.text_col, args.label_col, args.test_size, args.seed
    )

    objective = objective_factory(X_train, y_train, args.cv, args.n_jobs_inner, args.cache_dir)

    study_name = f"tfidf_spacy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    storage_path = os.path.abspath(os.path.join(args.output_dir, f"{study_name}.db"))
    storage_url = f"sqlite:///{storage_path}"
    print("Using local Optuna storage at:", storage_url)
    print("Study name:", study_name)
    if os.path.exists(storage_path):
        print("?? Existing Optuna DB found:", storage_path)
        os.remove(storage_path)
        print("??? Deleted old Optuna DB before starting new study.")


    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10),
        load_if_exists=False 
    )
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs_trials)

    # --- Save results ---
    ensure_dirs(args.output_dir)
    best_params_path = os.path.join(args.output_dir, f"{study_name}_best_params.json")
    trials_csv_path = os.path.join(args.output_dir, f"{study_name}_trials.csv")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params},
                  f, indent=2, ensure_ascii=False)
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "datetime_start", "datetime_complete"))
    df.to_csv(trials_csv_path, index=False)

    # --- Train final model ---
    memory = Memory(location=args.cache_dir, verbose=0)
    pipe = build_pipeline(memory)
    params = study.best_params.copy()
    tfidf_kwargs = {
        "analyzer": "word",
        "ngram_range": params.get("tfidf__ngram_range", (1, 2)),
        "max_features": params.get("tfidf__max_features", 50000),
        "min_df": params.get("tfidf__min_df", params.get("tfidf__min_df_frac", 1)),
        "max_df": params.get("tfidf__max_df", 1.0),
        "sublinear_tf": params.get("tfidf__sublinear_tf", True),
        "smooth_idf": params.get("tfidf__smooth_idf", True),
        "lowercase": params.get("tfidf__lowercase", True),
        "strip_accents": params.get("tfidf__strip_accents", None),
        "norm": params.get("tfidf__norm", "l2"),
        "use_idf": True,
        "stop_words": SPACY_STOPWORDS if params.get("tfidf__stopword_source", "none") == "spacy" else None
    }

    # Convert Optuna list [1, 3] back into tuple (1, 3)
    if isinstance(tfidf_kwargs["ngram_range"], list):
        tfidf_kwargs["ngram_range"] = tuple(tfidf_kwargs["ngram_range"])
    
    pipe.set_params(tfidf=TfidfVectorizer(**tfidf_kwargs))

    if params.get("clf", "logreg") == "logreg":
        pipe.set_params(clf=LogisticRegression(
            C=params.get("logreg__C", 1.0),
            solver=params.get("logreg__solver", "lbfgs"),
            class_weight=params.get("logreg__class_weight", None),
            max_iter=1000
        ))
    else:
        pipe.set_params(clf=LinearSVC(
            C=params.get("linearsvc__C", 1.0),
            class_weight=params.get("linearsvc__class_weight", None),
            loss="squared_hinge",
            dual=True
        ))

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)

    print("\n=== Test Set Classification Report (macro-F1) ===")
    print(report)

    model_path = os.path.join(args.output_dir, f"{study_name}_best_model.pkl")
    report_path = os.path.join(args.output_dir, f"{study_name}_test_report.txt")
    dump(pipe, model_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved artifacts:\n  Params: {best_params_path}\n  Trials: {trials_csv_path}\n"
          f"  Model: {model_path}\n  Report: {report_path}")


if __name__ == "__main__":
    main()
