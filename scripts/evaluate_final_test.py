#!/usr/bin/env python3
"""Freeze the validation-selected goal model, refit on development data, and test.

Keep train_baselines.py in the same scripts/ directory. Run:
    python scripts/evaluate_final_test.py

The selected model and its hyperparameters are fixed before test.csv is read.
Development data means train + validation. Completed evaluations are cached;
rerunning prints the saved result rather than rescoring or selecting on test.
"""

import argparse
import importlib.metadata
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

try:
    from train_baselines import CLASSES, evaluate, load_inputs, outcome_codes, predict_item, sha256
except ImportError:
    sys.exit("ERROR: Keep train_baselines.py beside evaluate_final_test.py in scripts/.")


PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")


def write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_environment(expected):
    actual = {name: importlib.metadata.version(name) for name in PACKAGES}
    mismatched = [name for name in PACKAGES if actual[name] != expected[name]]
    if mismatched:
        raise ValueError("Package versions changed since model creation: " + ", ".join(mismatched) + ". Restore the recorded environment before loading that model.")
    return actual


def freeze_development_model(data_dir, report_dir, baseline_path, frozen_path, plan_path, jobs):
    """This stage opens train/validation and metadata only, never the test CSV."""
    previous = json.loads((report_dir / "baseline_validation_report.json").read_text(encoding="utf-8"))
    versions = require_environment(previous["packages"])
    baseline_hash = sha256(baseline_path)
    if baseline_hash != previous["model_bundle_sha256"]:
        raise ValueError("The baseline bundle does not match its validation report.")
    baseline = joblib.load(baseline_path)
    selected_name = previous["best_goal_model"]
    if selected_name != baseline["best_goal_model"]:
        raise ValueError("The saved validation model selections disagree.")
    train, validation, schema, input_hashes = load_inputs(data_dir, report_dir)
    for key in ("train", "validation", "feature_schema"):
        if input_hashes[key] != previous["input_sha256"][key]:
            raise ValueError("Development inputs changed after baseline training: " + key)
    development = pd.concat((train, validation), ignore_index=True)
    original = baseline["models"][selected_name]
    if original["kind"] == "goal_models":
        chosen = {"kind": "goal_models"}
        for side, target in (("home", "home_score"), ("away", "away_score")):
            chosen[side] = clone(original[side])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                chosen[side].fit(development[schema["numeric_features"]], development[target])
    elif original["kind"] == "constant_goals":
        chosen = {"kind": "constant_goals", "goal_means": development[["home_score", "away_score"]].mean().tolist()}
    else:
        raise ValueError("The selected entry is not a goal model.")
    counts = np.bincount(outcome_codes(development), minlength=3).astype(float)
    references = {
        "DevelopmentPrior": {"kind": "outcome_prior", "probabilities": ((counts + 1.0) / (counts.sum() + 3.0)).tolist()},
        "DevelopmentConstantGoalsPoisson": {"kind": "constant_goals", "goal_means": development[["home_score", "away_score"]].mean().tolist()},
    }
    feature_report = json.loads((report_dir / "feature_engineering_report.json").read_text(encoding="utf-8"))
    plan = {
        "version": "0.1.0", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_on_validation": selected_name,
        "selection_rule": "Lowest validation outcome log loss among entries with goal predictions; no test-based reselection.",
        "selected_original_validation_metrics": next(row for row in previous["metrics"] if row["model"] == selected_name),
        "baseline_bundle_sha256": baseline_hash, "development_input_sha256": input_hashes,
        "development_rows": len(development),
        "development_first_date": development["date"].min().date().isoformat(),
        "development_last_date": development["date"].max().date().isoformat(),
        "test_start": schema["test_start"],
        "expected_test_sha256": feature_report["output_sha256"]["test.csv"],
        "expected_test_rows": feature_report["splits"]["test"]["rows"],
        "test_rows_before_history_filter": feature_report["splits"]["test"]["before_history_filter"],
        "feature_schema_sha256": input_hashes["feature_schema"], "packages": versions,
        "outcome_probability_order": CLASSES, "feature_schema": schema,
        "model_parameters": {side: repr(chosen[side]) for side in ("home", "away")} if chosen["kind"] == "goal_models" else chosen,
        "refitting_rule": "Clone the selected training-only estimators with unchanged hyperparameters; fit on train plus validation.",
        "calibration": "No post-fit probability calibration applied.",
        "history_policy": schema["evaluation_mode"],
        "target_definition": schema["target_definition"],
    }
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"selected_model_name": selected_name, "selected_model": chosen, "references": references, "feature_schema": schema, "metadata": plan}, frozen_path, compress=3)
    plan["frozen_model_sha256"] = sha256(frozen_path)
    # Persist the complete model choice and its fingerprint before accessing test.csv.
    write_json(plan_path, plan)
    return plan, {"selected_model": chosen, "references": references, "feature_schema": schema}


def load_frozen(plan_path, frozen_path, data_dir):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    require_environment(plan["packages"])
    if sha256(frozen_path) != plan["frozen_model_sha256"]:
        raise ValueError("The frozen model differs from the saved test plan.")
    if sha256(data_dir / "feature_schema.json") != plan["feature_schema_sha256"]:
        raise ValueError("The feature schema changed after the model was frozen.")
    frozen = joblib.load(frozen_path)
    if frozen["selected_model_name"] != plan["selected_on_validation"]:
        raise ValueError("The selected model name differs from the frozen plan.")
    if any(plan.get(key) != value for key, value in frozen["metadata"].items()):
        raise ValueError("The test plan was changed after model creation.")
    return plan, frozen


def load_test(path, plan):
    if sha256(path) != plan["expected_test_sha256"]:
        raise ValueError("test.csv does not match the frozen test plan.")
    test = pd.read_csv(path)
    features = plan["feature_schema"]["numeric_features"]
    required = set(features) | {"match_id", "date", "home_team", "away_team", "home_score", "away_score", "outcome", "eligible_for_model", "split", "latest_history_date"}
    if required - set(test.columns) or len(test) != plan["expected_test_rows"] or test.empty:
        raise ValueError("Unexpected test columns or row count.")
    if test["match_id"].duplicated().any() or not test["split"].eq("test").all():
        raise ValueError("Duplicate match IDs or wrong split labels in test.csv.")
    if not test["eligible_for_model"].astype(str).str.upper().eq("TRUE").all():
        raise ValueError("Ineligible test matches were included.")
    test["date"] = pd.to_datetime(test["date"], format="%Y-%m-%d", errors="raise")
    if test["date"].min() < pd.Timestamp(plan["test_start"]) or test["date"].min() <= pd.Timestamp(plan["development_last_date"]):
        raise ValueError("The test period overlaps development data.")
    history = pd.to_datetime(test["latest_history_date"], errors="raise")
    if history.isna().any() or history.ge(test["date"]).any():
        raise ValueError("Missing or non-prior feature history in the eligible test rows.")
    test[features] = test[features].apply(pd.to_numeric, errors="raise")
    if np.isinf(test[features].to_numpy(dtype=float)).any():
        raise ValueError("Infinite test features.")
    goals = test[["home_score", "away_score"]].to_numpy(dtype=float)
    if not np.isfinite(goals).all() or (goals < 0).any() or not np.equal(goals, np.floor(goals)).all():
        raise ValueError("Invalid goal targets in test.csv.")
    expected = np.where(goals[:, 0] > goals[:, 1], 0, np.where(goals[:, 0] == goals[:, 1], 1, 2))
    if not np.array_equal(outcome_codes(test), expected):
        raise ValueError("Test outcome labels disagree with goal targets.")
    return test


def score_frozen(test, frozen, plan, report_dir):
    rows, forecasts = [], []
    entries = [(plan["selected_on_validation"], frozen["selected_model"]), *frozen["references"].items()]
    X = test[plan["feature_schema"]["numeric_features"]]
    for name, item in entries:
        probabilities, rates = predict_item(item, X)
        metrics = evaluate(test, probabilities, rates)
        metrics["model"] = name
        rows.append(metrics)
        forecast = test[["match_id", "date", "home_team", "away_team", "home_score", "away_score", "outcome"]].copy()
        forecast["model"] = name
        forecast["predicted_outcome"] = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
        for i, outcome in enumerate(CLASSES):
            forecast["prob_" + outcome] = probabilities[:, i]
        if rates is not None:
            forecast["expected_home_goals"], forecast["expected_away_goals"] = rates[:, 0], rates[:, 1]
            forecast["mode_home_goals"], forecast["mode_away_goals"] = np.floor(rates[:, 0]).astype(int), np.floor(rates[:, 1]).astype(int)
        forecasts.append(forecast)
    selected_probabilities, selected_rates = predict_item(frozen["selected_model"], X)
    by_year = {}
    for year in sorted(test["date"].dt.year.unique()):
        mask = test["date"].dt.year.eq(year).to_numpy()
        by_year[str(year)] = {"matches": int(mask.sum()), **evaluate(test.loc[mask], selected_probabilities[mask], selected_rates[mask])}
    report = {
        "version": "0.1.0", "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_on_validation": plan["selected_on_validation"], "test_based_reselection": False,
        "frozen_model_sha256": plan["frozen_model_sha256"], "test_sha256": plan["expected_test_sha256"],
        "development_rows": plan["development_rows"], "development_last_date": plan["development_last_date"],
        "test_rows": len(test), "test_first_date": test["date"].min().date().isoformat(), "test_last_date": test["date"].max().date().isoformat(),
        "test_rows_before_history_filter": plan["test_rows_before_history_filter"],
        "outcome_probability_order": CLASSES, "metrics": rows,
        "selected_model_by_year": by_year, "history_policy": plan["history_policy"],
        "target_definition": plan["target_definition"], "calibration": plan["calibration"],
        "metric_definitions": {"accuracy": "fraction, not percent", "brier_score": "multiclass sum convention, range 0 to 2", "goal_mae": "mean absolute goal error across both teams"},
    }
    prediction_path = report_dir / "final_test_predictions.csv"
    temporary = prediction_path.with_name(prediction_path.name + ".tmp")
    pd.concat(forecasts, ignore_index=True).to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(prediction_path)
    metric_path = report_dir / "final_test_metrics.csv"
    pd.DataFrame(rows)[["model", "accuracy", "macro_f1", "log_loss", "brier_score", "goal_mae", "scoreline_nll", "exact_score_accuracy"]].to_csv(metric_path, index=False)
    report["predictions_sha256"] = sha256(prediction_path)
    write_json(report_dir / "final_test_report.json", report)
    return report


def display(report, cached):
    print("FOOTBALL FINAL TEST EVALUATION")
    print("Status:", "SAVED EVALUATION; NOT REPEATED" if cached else "FROZEN MODEL EVALUATED")
    print("Selected on validation:", report["selected_on_validation"])
    print("Development rows:", report["development_rows"], "| Last development date:", report["development_last_date"])
    print("Test matches:", report["test_rows"], "|", report["test_first_date"], "to", report["test_last_date"])
    for item in report["metrics"]:
        mae = "n/a" if item["goal_mae"] is None else f"{item['goal_mae']:.4f}"
        print(f"{item['model']}: accuracy={item['accuracy']:.2%}, log loss={item['log_loss']:.4f}, Brier={item['brier_score']:.4f}, goal MAE={mae}")
    print("Model selection was fixed before test access. No test-based reselection.")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    parser.add_argument("--baseline-bundle", type=Path, default=root / "models/baselines/baseline_bundle.joblib")
    parser.add_argument("--frozen-model", type=Path, default=root / "models/frozen/frozen_model.joblib")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive.")
    data_dir, report_dir = args.data_dir.expanduser(), args.report_dir.expanduser()
    frozen_path = args.frozen_model.expanduser()
    plan_path, result_path = report_dir / "final_test_plan.json", report_dir / "final_test_report.json"
    try:
        if result_path.is_file():
            report = json.loads(result_path.read_text(encoding="utf-8"))
            if sha256(frozen_path) != report["frozen_model_sha256"]:
                raise ValueError("The model no longer matches the completed evaluation.")
            display(report, cached=True)
            return 0
        with threadpool_limits(limits=args.jobs):
            if plan_path.is_file():
                print("Resuming the existing frozen evaluation plan...", flush=True)
                plan, frozen = load_frozen(plan_path, frozen_path, data_dir)
            else:
                print("Freezing the validation-selected model and refitting on train + validation...", flush=True)
                plan, frozen = freeze_development_model(data_dir, report_dir, args.baseline_bundle.expanduser(), frozen_path, plan_path, args.jobs)
            print("Frozen plan saved. Evaluating the reserved test period...", flush=True)
            test = load_test(data_dir / "test.csv", plan)
            report = score_frozen(test, frozen, plan, report_dir)
        display(report, cached=False)
        print("Frozen model:", frozen_path.resolve())
        print("Report:", result_path.resolve())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ConvergenceWarning) as exc:
        parser.exit(2, "ERROR: " + str(exc) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
