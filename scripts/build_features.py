#!/usr/bin/env python3
"""Build pre-match features and chronological splits using only the standard library.

Run from the project: python scripts/build_features.py
Inputs: data/processed/matches.csv and reports/dataset_preparation.json.
Outputs: features.csv, train.csv, validation.csv, test.csv, feature_schema.json,
and reports/feature_engineering_report.json. No model is fitted.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from itertools import groupby
from pathlib import Path


ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0
FORM_WINDOWS = (5, 10)
TARGETS = ["home_score", "away_score", "outcome", "total_goals", "both_teams_scored"]
FEATURES = [
    "is_neutral", "is_friendly", "home_elo", "away_elo", "elo_diff",
    "elo_home_expected_score", "home_history_matches", "away_history_matches",
    "home_days_rest", "away_days_rest",
] + [
    f"{side}_{metric}_{window}"
    for window in FORM_WINDOWS for side in ("home", "away")
    for metric in ("goals_for_mean", "goals_against_mean", "points_mean", "win_rate")
] + ["h2h_matches_3", "h2h_home_goals_mean_3", "h2h_away_goals_mean_3", "h2h_home_points_mean_3"]
AUDIT_COLUMNS = ["latest_history_date", "eligible_for_model", "split"]


@dataclass
class TeamState:
    rating: float = ELO_INITIAL
    appearances: int = 0
    last_date: object = None
    results: deque = field(default_factory=lambda: deque(maxlen=10))


def points(goals_for, goals_against):
    return 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0


def prematch_features(row, states, meetings):
    """Reusable pre-match calculation. States/meetings must contain earlier dates only."""
    day = date.fromisoformat(row["date"])
    home = states[row["home_team"]]
    away = states[row["away_team"]]
    history = meetings.get(tuple(sorted((row["home_team"], row["away_team"]))), ())
    used_dates = [state.last_date for state in (home, away) if state.last_date is not None]
    used_dates += [meeting["date"] for meeting in history]
    if any(used >= day for used in used_dates):
        raise ValueError("Feature history contains the current date or a future date.")
    neutral = row["neutral"].upper() == "TRUE"
    advantage = 0.0 if neutral else ELO_HOME_ADVANTAGE
    expected = 1.0 / (1.0 + 10.0 ** ((away.rating - home.rating - advantage) / 400.0))
    values = {
        "is_neutral": int(neutral),
        "is_friendly": int(row["tournament"].casefold() == "friendly"),
        "home_elo": home.rating, "away_elo": away.rating,
        "elo_diff": home.rating - away.rating,
        "elo_home_expected_score": expected,
        "home_history_matches": home.appearances, "away_history_matches": away.appearances,
        "home_days_rest": (day - home.last_date).days if home.last_date else None,
        "away_days_rest": (day - away.last_date).days if away.last_date else None,
        "h2h_matches_3": len(history),
    }
    for window in FORM_WINDOWS:
        for side, state in (("home", home), ("away", away)):
            recent = list(state.results)[-window:]
            values[f"{side}_goals_for_mean_{window}"] = sum(x[0] for x in recent) / len(recent) if recent else None
            values[f"{side}_goals_against_mean_{window}"] = sum(x[1] for x in recent) / len(recent) if recent else None
            values[f"{side}_points_mean_{window}"] = sum(points(*x) for x in recent) / len(recent) if recent else None
            values[f"{side}_win_rate_{window}"] = sum(x[0] > x[1] for x in recent) / len(recent) if recent else None
    if history:
        h_goals = [item["goals"][row["home_team"]] for item in history]
        a_goals = [item["goals"][row["away_team"]] for item in history]
        values["h2h_home_goals_mean_3"] = sum(h_goals) / len(history)
        values["h2h_away_goals_mean_3"] = sum(a_goals) / len(history)
        values["h2h_home_points_mean_3"] = sum(points(h, a) for h, a in zip(h_goals, a_goals)) / len(history)
    else:
        for name in ("h2h_home_goals_mean_3", "h2h_away_goals_mean_3", "h2h_home_points_mean_3"):
            values[name] = None
    for key, value in values.items():
        if value is not None and not math.isfinite(value):
            raise ValueError("Non-finite feature: " + key)
    latest = max(used_dates).isoformat() if used_dates else ""
    return values, latest


def engineer(rows, validation_start, test_start, min_history=5):
    """All fixtures on a date use a common pre-date state; update after the batch."""
    states = defaultdict(TeamState)
    meetings = defaultdict(lambda: deque(maxlen=3))
    output = []
    ordered = sorted(rows, key=lambda item: (item["date"], item["match_id"]))
    for day_text, grouped in groupby(ordered, key=lambda item: item["date"]):
        batch = list(grouped)
        deltas = Counter()
        for row in batch:
            values, latest = prematch_features(row, states, meetings)
            eligible = min(values["home_history_matches"], values["away_history_matches"]) >= min_history
            day = date.fromisoformat(day_text)
            split = "train" if day < validation_start else "validation" if day < test_start else "test"
            item = dict(row)
            item.update({name: round(values[name], 8) if isinstance(values[name], float) else values[name] for name in FEATURES})
            item.update(latest_history_date=latest, eligible_for_model="TRUE" if eligible else "FALSE", split=split)
            output.append(item)
            actual = 1.0 if int(row["home_score"]) > int(row["away_score"]) else 0.5 if int(row["home_score"]) == int(row["away_score"]) else 0.0
            delta = ELO_K * (actual - values["elo_home_expected_score"])
            deltas[row["home_team"]] += delta
            deltas[row["away_team"]] -= delta
        for team, delta in deltas.items():
            states[team].rating += delta
        # Deterministic match-ID ordering resolves unavailable within-date order
        # for later rolling windows; it never feeds into same-date forecasts.
        for row in batch:
            home_goals, away_goals = int(row["home_score"]), int(row["away_score"])
            for team, gf, ga in ((row["home_team"], home_goals, away_goals), (row["away_team"], away_goals, home_goals)):
                state = states[team]
                state.appearances += 1
                state.last_date = date.fromisoformat(day_text)
                state.results.append((gf, ga))
            key = tuple(sorted((row["home_team"], row["away_team"])))
            meetings[key].append({"date": date.fromisoformat(day_text), "goals": {row["home_team"]: home_goals, row["away_team"]: away_goals}})
    return output


def load_prepared(path, preparation_report):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(preparation_report.read_text(encoding="utf-8"))
    if digest != manifest["output_sha256"]:
        raise ValueError("matches.csv does not match dataset_preparation.json. Rerun prepare_dataset.py first.")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        required = {"match_id", "date", "home_team", "away_team", "tournament", "neutral", *TARGETS}
        if required - set(columns):
            raise ValueError("Prepared dataset is missing columns: " + repr(sorted(required - set(columns))))
        rows = list(reader)
    if len(rows) != manifest["included_rows"] or not rows:
        raise ValueError("Prepared row count does not match its report, or the file is empty.")
    if len({row["match_id"] for row in rows}) != len(rows):
        raise ValueError("Prepared match IDs are not unique.")
    return rows, columns, digest, manifest


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_outputs(feature_rows, input_columns, input_hash, preparation, output_dir, report_dir, validation_start, test_start, min_history):
    columns = input_columns + FEATURES + AUDIT_COLUMNS
    selected = {name: [r for r in feature_rows if r["split"] == name and r["eligible_for_model"] == "TRUE"] for name in ("train", "validation", "test")}
    for name, rows in selected.items():
        if not rows:
            raise ValueError("No eligible matches in " + name + ". Check date boundaries and history requirements.")
    if not (max(r["date"] for r in selected["train"]) < min(r["date"] for r in selected["validation"]) and max(r["date"] for r in selected["validation"]) < min(r["date"] for r in selected["test"])):
        raise ValueError("Chronological split boundaries overlap.")
    if set(FEATURES) & set(TARGETS):
        raise ValueError("Target columns were included in the feature list.")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "version": "0.1.0", "numeric_features": FEATURES, "target_columns": TARGETS,
        "audit_columns_do_not_use_as_features": AUDIT_COLUMNS,
        "target_definition": preparation["target_definition"],
        "missing_values": "Blank means unavailable. Fit any imputer and scaler on training rows only.",
        "elo": {"kind": "custom engineering feature", "initial": ELO_INITIAL, "k": ELO_K, "scale": 400, "home_advantage": ELO_HOME_ADVANTAGE, "neutral_advantage": 0},
        "elo_expected_score_definition": "Expected win/draw/loss score with win=1, draw=0.5, loss=0; not a home-win probability.",
        "form_windows": list(FORM_WINDOWS), "head_to_head_window": 3,
        "window_definition": "Uses up to the specified number of prior appearances; an empty history is unavailable, not zero.",
        "form_points": {"win": 3, "draw": 1, "loss": 0},
        "history_initialization": "Begins at the first prepared match; no pre-1990 results are used by default.",
        "same_date_policy": "All features use strictly earlier calendar dates. All same-date ratings use the common pre-date state.",
        "within_date_history_order": "Deterministic match-ID order for subsequent rolling windows; kickoff order is unavailable.",
        "evaluation_mode": "Sequential forecasts: earlier validation/test results may update history before later fixtures. Model fitting remains restricted to the designated fitting period.",
        "minimum_prior_appearances_each_team": min_history,
        "validation_start": validation_start.isoformat(), "test_start": test_start.isoformat(),
        "defaults_note": "Engineering defaults fixed before model evaluation; tune using training/validation only.",
        "is_friendly_definition": "Tournament field equals Friendly, ignoring case.",
    }
    report = {
        "builder_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_hash, "total_feature_rows": len(feature_rows),
        "numeric_features": len(FEATURES), "eligible_rows": sum(len(r) for r in selected.values()),
        "excluded_for_insufficient_history": sum(r["eligible_for_model"] == "FALSE" for r in feature_rows),
        "splits": {}, "output_sha256": {},
    }
    report["output_sha256"]["features.csv"] = write_csv(output_dir / "features.csv", feature_rows, columns)
    for name, rows in selected.items():
        missing = {feature: sum(r[feature] is None for r in rows) for feature in FEATURES}
        report["splits"][name] = {
            "rows": len(rows), "first_match": rows[0]["date"], "last_match": rows[-1]["date"],
            "before_history_filter": sum(r["split"] == name for r in feature_rows),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "missing_feature_values": {key: value for key, value in missing.items() if value},
        }
        report["output_sha256"][name + ".csv"] = write_csv(output_dir / (name + ".csv"), rows, columns)
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    (report_dir / "feature_engineering_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=root / "data/processed/matches.csv")
    parser.add_argument("--preparation-report", type=Path, default=root / "reports/dataset_preparation.json")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2019, 1, 1))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--min-history", type=int, default=5)
    args = parser.parse_args()
    if args.validation_start >= args.test_start or args.min_history < 0:
        parser.error("Validation must start before test; minimum history must be nonnegative.")
    try:
        rows, columns, digest, preparation = load_prepared(args.matches.expanduser(), args.preparation_report.expanduser())
        engineered = engineer(rows, args.validation_start, args.test_start, args.min_history)
        report = save_outputs(engineered, columns, digest, preparation, args.output_dir.expanduser(), args.report_dir.expanduser(), args.validation_start, args.test_start, args.min_history)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        parser.exit(2, "ERROR: " + str(exc) + "\n")
    print("FOOTBALL FEATURE ENGINEERING")
    print("Status: READY FOR BASELINE TRAINING")
    print("Feature rows:", report["total_feature_rows"])
    print("Numeric features:", report["numeric_features"])
    print("Eligible matches:", report["eligible_rows"])
    print("Excluded for insufficient earlier history:", report["excluded_for_insufficient_history"])
    for name, split in report["splits"].items():
        print(name.upper() + ":", split["rows"], "matches |", split["first_match"], "to", split["last_match"])
    print("Data:", args.output_dir.expanduser().resolve())
    print("Report:", (args.report_dir.expanduser() / "feature_engineering_report.json").resolve())
    print("All histories use earlier calendar dates. Missing values remain blank.")
    print("No imputers, scalers or predictive models fitted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
