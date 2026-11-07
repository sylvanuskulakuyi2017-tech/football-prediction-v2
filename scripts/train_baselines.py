#!/usr/bin/env python3
"""Fit training-only football baselines and compare them on validation matches.

Run: python scripts/train_baselines.py
Requires numpy, pandas, scipy, scikit-learn, joblib and threadpoolctl.
Does not open test.csv or evaluate test predictions. Retrains on every run.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


CLASSES = ["home_win", "draw", "away_win"]
RATE_FLOOR = 1e-8
SEED = 42


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome_codes(frame):
    mapped = frame["outcome"].map({name: i for i, name in enumerate(CLASSES)})
    if mapped.isna().any():
        raise ValueError("Unknown outcome label.")
    return mapped.to_numpy(dtype=int)


def outcome_probabilities(home_rate, away_rate):
    """Independent-Poisson outcomes over the full score support, ordered H/D/A."""
    home, away = np.broadcast_arrays(np.asarray(home_rate, dtype=float), np.asarray(away_rate, dtype=float))
    if not np.isfinite(home).all() or not np.isfinite(away).all() or (home < 0).any() or (away < 0).any():
        raise ValueError("Goal rates must be finite and nonnegative.")
    home, away = np.maximum(home, RATE_FLOOR), np.maximum(away, RATE_FLOOR)
    probabilities = np.column_stack((skellam.sf(0, home, away), skellam.pmf(0, home, away), skellam.cdf(-1, home, away)))
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Poisson outcome probabilities failed numerical checks.")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def evaluate(frame, probabilities, rates=None):
    y = outcome_codes(frame)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y), 3) or not np.isfinite(probabilities).all():
        raise ValueError("Expected one finite H/D/A probability row per match.")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Invalid probability distribution.")
    labels = probabilities.argmax(axis=1)
    truth = np.eye(3)[y]
    metrics = {
        "accuracy": float(accuracy_score(y, labels)),
        "macro_f1": float(f1_score(y, labels, labels=[0, 1, 2], average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "brier_score": float(np.mean(np.sum((probabilities - truth) ** 2, axis=1))),
        "goal_mae": None, "scoreline_nll": None, "exact_score_accuracy": None,
        "confusion_matrix": confusion_matrix(y, labels, labels=[0, 1, 2]).tolist(),
    }
    if rates is not None:
        observed = frame[["home_score", "away_score"]].to_numpy(dtype=int)
        rates = np.maximum(np.asarray(rates, dtype=float), RATE_FLOOR)
        if rates.shape != observed.shape or not np.isfinite(rates).all():
            raise ValueError("Expected two finite goal-rate estimates per match.")
        metrics["goal_mae"] = float(np.mean(np.abs(observed - rates)))
        metrics["scoreline_nll"] = float(-poisson.logpmf(observed, rates).sum(axis=1).mean())
        # For integer lambda, floor(lambda) is one of two equal Poisson modes.
        metrics["exact_score_accuracy"] = float(np.all(np.floor(rates).astype(int) == observed, axis=1).mean())
    return metrics


def load_inputs(data_dir, report_dir):
    schema_path = data_dir / "feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report = json.loads((report_dir / "feature_engineering_report.json").read_text(encoding="utf-8"))
    features = schema["numeric_features"]
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature list is empty or duplicated.")
    if set(features) & set(schema["target_columns"] + schema["audit_columns_do_not_use_as_features"]):
        raise ValueError("Targets or audit columns were included as model features.")
    frames, digests = {}, {}
    for split in ("train", "validation"):
        path = data_dir / (split + ".csv")
        digest = sha256(path)
        if digest != report["output_sha256"][path.name]:
            raise ValueError(path.name + " differs from its feature report. Rerun build_features.py.")
        frame = pd.read_csv(path)
        required = set(features) | {"match_id", "date", "home_team", "away_team", "home_score", "away_score", "outcome", "split", "eligible_for_model"}
        if required - set(frame.columns) or frame.empty:
            raise ValueError("Missing columns or empty " + split + " split.")
        if len(frame) != report["splits"][split]["rows"] or frame["match_id"].duplicated().any():
            raise ValueError("Unexpected row count or duplicate IDs in " + split + ".")
        if not frame["split"].eq(split).all() or not frame["eligible_for_model"].astype(str).str.upper().eq("TRUE").all():
            raise ValueError("Wrong split label or ineligible matches in " + split + ".")
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
        frame[features] = frame[features].apply(pd.to_numeric, errors="raise")
        if np.isinf(frame[features].to_numpy(dtype=float)).any():
            raise ValueError("Infinite model inputs in " + split + ".")
        goals = frame[["home_score", "away_score"]].to_numpy(dtype=float)
        if not np.isfinite(goals).all() or (goals < 0).any() or not np.equal(goals, np.floor(goals)).all():
            raise ValueError("Invalid goal targets.")
        expected = np.where(goals[:, 0] > goals[:, 1], 0, np.where(goals[:, 0] == goals[:, 1], 1, 2))
        if not np.array_equal(outcome_codes(frame), expected):
            raise ValueError("Outcome labels disagree with goal targets.")
        if "latest_history_date" in frame:
            history = pd.to_datetime(frame["latest_history_date"], errors="raise")
            if history.ge(frame["date"]).any():
                raise ValueError("Current-date or future history found in " + split + ".")
        frames[split], digests[split] = frame, digest
    train, validation = frames["train"], frames["validation"]
    boundary = pd.Timestamp(schema["validation_start"])
    test_boundary = pd.Timestamp(schema["test_start"])
    if not (train["date"].max() < boundary <= validation["date"].min() and validation["date"].max() < test_boundary):
        raise ValueError("Chronological boundaries do not match the feature schema.")
    if set(train["match_id"]) & set(validation["match_id"]):
        raise ValueError("Training and validation match IDs overlap.")
    digests["feature_schema"] = sha256(schema_path)
    return train, validation, schema, digests


def make_regressor(name, jobs):
    if name == "RegularizedPoisson":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500, tol=1e-7))
    if name == "RandomForest":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=10, random_state=SEED, n_jobs=jobs))
    if name == "PoissonGradientBoosting":
        return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, random_state=SEED)
    raise ValueError("Unknown regressor: " + name)


def predict_item(item, X):
    if item["kind"] == "outcome_prior":
        return np.tile(item["probabilities"], (len(X), 1)), None
    if item["kind"] == "constant_goals":
        rates = np.tile(item["goal_means"], (len(X), 1))
    elif item["kind"] == "goal_models":
        rates = np.column_stack([item[side].predict(X) for side in ("home", "away")])
    else:
        raise ValueError("Unknown saved model kind.")
    return outcome_probabilities(rates[:, 0], rates[:, 1]), np.maximum(rates, RATE_FLOOR)


def train_and_validate(train, validation, schema, jobs=2):
    features = schema["numeric_features"]
    X_train, X_val = train[features], validation[features]
    counts = np.bincount(outcome_codes(train), minlength=3).astype(float)
    prior = (counts + 1.0) / (counts.sum() + 3.0)
    models = {
        "TrainingPrior": {"kind": "outcome_prior", "probabilities": prior.tolist()},
        "ConstantGoalsPoisson": {"kind": "constant_goals", "goal_means": train[["home_score", "away_score"]].mean().tolist()},
    }
    results, predictions, configs = [], [], {}
    for name in ("TrainingPrior", "ConstantGoalsPoisson", "RegularizedPoisson", "RandomForest", "PoissonGradientBoosting"):
        started = time.perf_counter()
        print("Evaluating " + name + "...", flush=True)
        if name not in models:
            item = {"kind": "goal_models"}
            for side, target in (("home", "home_score"), ("away", "away_score")):
                regressor = make_regressor(name, jobs)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    regressor.fit(X_train, train[target])
                item[side] = regressor
                if side == "home":
                    configs[name] = repr(regressor)
            models[name] = item
        probabilities, rates = predict_item(models[name], X_val)
        metrics = evaluate(validation, probabilities, rates)
        metrics.update(model=name, seconds=float(time.perf_counter() - started), has_goal_model=rates is not None)
        results.append(metrics)
        pred = validation[["match_id", "date", "home_team", "away_team", "home_score", "away_score", "outcome"]].copy()
        pred["model"] = name
        for i, outcome in enumerate(CLASSES):
            pred["prob_" + outcome] = probabilities[:, i]
        pred["predicted_outcome"] = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
        if rates is not None:
            pred["expected_home_goals"], pred["expected_away_goals"] = rates[:, 0], rates[:, 1]
            pred["mode_home_goals"], pred["mode_away_goals"] = np.floor(rates[:, 0]).astype(int), np.floor(rates[:, 1]).astype(int)
        predictions.append(pred)
        goal_text = "n/a" if metrics["goal_mae"] is None else f"{metrics['goal_mae']:.4f}"
        print(f"  Validation accuracy={metrics['accuracy']:.2%}, log loss={metrics['log_loss']:.4f}, Brier={metrics['brier_score']:.4f}, goal MAE={goal_text}", flush=True)
    return models, results, pd.concat(predictions, ignore_index=True), configs


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    parser.add_argument("--model-dir", type=Path, default=root / "models/baselines")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive.")
    try:
        train, validation, schema, hashes = load_inputs(args.data_dir.expanduser(), args.report_dir.expanduser())
        with threadpool_limits(limits=args.jobs):
            models, results, predictions, configs = train_and_validate(train, validation, schema, args.jobs)
        ranked = sorted(results, key=lambda row: row["log_loss"])
        best_goal = next(row for row in ranked if row["has_goal_model"])
        versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")}
        model_dir, report_dir = args.model_dir.expanduser(), args.report_dir.expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "builder_version": "0.1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0], "packages": versions, "seed": SEED,
            "training_rows": len(train), "validation_rows": len(validation),
            "input_sha256": hashes, "outcome_probability_order": CLASSES,
            "selection_metric": "validation log loss, lower is better",
            "best_outcome_predictor": ranked[0]["model"], "best_goal_model": best_goal["model"],
            "test_evaluated": False, "test_file_opened": False,
            "metrics": ranked, "model_configs": configs,
            "metric_definitions": {"accuracy": "fraction, not percent", "brier_score": "mean sum over three classes of squared errors; range 0 to 2", "goal_mae": "mean absolute goal error over both teams", "scoreline_nll": "mean negative log probability of the observed goal pair", "exact_score_accuracy": "fraction matching the independent-Poisson modal score pair"},
            "forecast_assumptions": "Independent Poisson goals; recorded final goals may include extra time; no penalty-shootout goals.",
            "rate_floor": RATE_FLOOR, "prior_additive_smoothing": 1.0,
            "probability_note": "Raw model-derived probabilities; no post-fit calibration has been applied.",
            "history_policy": schema["evaluation_mode"], "feature_schema": schema,
        }
        bundle_path = model_dir / "baseline_bundle.joblib"
        joblib.dump({"models": models, "best_goal_model": best_goal["model"], "best_outcome_predictor": ranked[0]["model"], "feature_schema": schema, "metadata": report}, bundle_path, compress=3)
        report["model_bundle_sha256"] = sha256(bundle_path)
        predictions.to_csv(report_dir / "validation_predictions.csv", index=False, date_format="%Y-%m-%d")
        scalar_columns = ["model", "accuracy", "macro_f1", "log_loss", "brier_score", "goal_mae", "scoreline_nll", "exact_score_accuracy", "seconds"]
        pd.DataFrame(ranked)[scalar_columns].to_csv(report_dir / "baseline_validation_metrics.csv", index=False)
        (report_dir / "baseline_validation_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (report_dir / "baseline_environment.txt").write_text("\n".join(name + "==" + version for name, version in versions.items()) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ConvergenceWarning) as exc:
        parser.exit(2, "ERROR: " + str(exc) + "\n")
    print("FOOTBALL BASELINE TRAINING COMPLETE")
    print("Training rows:", len(train), "| Validation rows:", len(validation))
    print("Best validation outcome predictor:", ranked[0]["model"])
    print("Best validation goal model:", best_goal["model"])
    print("Model bundle:", bundle_path.resolve())
    print("Metrics:", (report_dir / "baseline_validation_metrics.csv").resolve())
    print("Report:", (report_dir / "baseline_validation_report.json").resolve())
    print("Final test set not opened or evaluated. Selection is provisional on validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
