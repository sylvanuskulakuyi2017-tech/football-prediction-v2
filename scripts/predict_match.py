#!/usr/bin/env python3
"""Predict supported international fixtures or replay recorded matches.

Example (hypothetical, not a scheduled fixture):
  python scripts/predict_match.py --home Nigeria --away Brazil --date 2026-11-08 --neutral

Use --list-teams to inspect eligible team names. Add --replay for an existing
fixture; recorded venue and tournament are then used. Keep build_features.py,
train_baselines.py and evaluate_final_test.py beside this file.
"""

import argparse
import csv
import difflib
import hashlib
import json
import sys
from collections import Counter, defaultdict, deque
from datetime import date, datetime, timezone
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from threadpoolctl import threadpool_limits

try:
    from build_features import ELO_INITIAL, ELO_K, ELO_HOME_ADVANTAGE, FEATURES, FORM_WINDOWS, TeamState, prematch_features
    from evaluate_final_test import load_frozen
    from train_baselines import CLASSES, predict_item, sha256
except ImportError:
    sys.exit("ERROR: Keep the earlier project scripts beside predict_match.py in scripts/.")


def history_before(rows, cutoff):
    """Reproduce build_features' date-batched history, including warmup fixtures."""
    states = defaultdict(TeamState)
    meetings = defaultdict(lambda: deque(maxlen=3))
    available = sorted((r for r in rows if r["date"] < cutoff.isoformat()), key=lambda r: (r["date"], r["match_id"]))
    for day_text, grouped in groupby(available, key=lambda r: r["date"]):
        batch = list(grouped)
        deltas = Counter()
        for row in batch:
            values, _ = prematch_features(row, states, meetings)
            home, away = int(row["home_score"]), int(row["away_score"])
            actual = 1.0 if home > away else 0.5 if home == away else 0.0
            delta = ELO_K * (actual - values["elo_home_expected_score"])
            deltas[row["home_team"]] += delta
            deltas[row["away_team"]] -= delta
        for team, delta in deltas.items():
            states[team].rating += delta
        for row in batch:
            home, away = int(row["home_score"]), int(row["away_score"])
            for team, gf, ga in ((row["home_team"], home, away), (row["away_team"], away, home)):
                state = states[team]
                state.appearances += 1
                state.last_date = date.fromisoformat(day_text)
                state.results.append((gf, ga))
            key = tuple(sorted((row["home_team"], row["away_team"])))
            meetings[key].append({"date": date.fromisoformat(day_text), "goals": {row["home_team"]: home, row["away_team"]: away}})
    return states, meetings, available[-1]["date"] if available else None


def distribution_details(home_rate, away_rate, top_n=5):
    """Rank actual score probabilities; do not renormalize them to a small grid."""
    bounds = [int(poisson.ppf(1.0 - 1e-12, rate)) for rate in (home_rate, away_rate)]
    if (bounds[0] + 1) * (bounds[1] + 1) > 1_000_000:
        raise ValueError("Predicted goal rates require an unusually large score grid; inspect the model inputs.")
    grid = np.outer(poisson.pmf(np.arange(bounds[0] + 1), home_rate), poisson.pmf(np.arange(bounds[1] + 1), away_rate))
    positions = np.argsort(-grid, axis=None, kind="stable")[:top_n]
    scorelines = []
    for index in positions:
        h, a = np.unravel_index(index, grid.shape)
        scorelines.append({"home_goals": int(h), "away_goals": int(a), "probability": float(grid[h, a])})
    return {
        "top_scorelines": scorelines,
        "additional_probabilities": {
            "total_goals_at_least_3": float(poisson.sf(2, home_rate + away_rate)),
            "both_teams_score": float(poisson.sf(0, home_rate) * poisson.sf(0, away_rate)),
        },
    }


class MatchPredictor:
    """Reusable prediction component for the later interactive application."""

    def __init__(self, project_root):
        self.root = Path(project_root).expanduser().resolve()
        data_dir, report_dir = self.root / "data/processed", self.root / "reports"
        self.plan, self.frozen = load_frozen(report_dir / "final_test_plan.json", self.root / "models/frozen/frozen_model.joblib", data_dir)
        schema = self.plan["feature_schema"]
        if schema["numeric_features"] != FEATURES or schema["form_windows"] != list(FORM_WINDOWS):
            raise ValueError("Feature definitions differ from those used by the frozen model.")
        elo = schema["elo"]
        if (elo["initial"], elo["k"], elo["home_advantage"]) != (ELO_INITIAL, ELO_K, ELO_HOME_ADVANTAGE):
            raise ValueError("Elo settings differ from the frozen feature definitions.")
        source = data_dir / "matches.csv"
        preparation = json.loads((report_dir / "dataset_preparation.json").read_text(encoding="utf-8"))
        feature_report = json.loads((report_dir / "feature_engineering_report.json").read_text(encoding="utf-8"))
        self.history_sha256 = sha256(source)
        if self.history_sha256 != preparation["output_sha256"] or self.history_sha256 != feature_report["input_sha256"]:
            raise ValueError("The prepared history and its reports disagree. Restore a matching data snapshot.")
        for split in ("train", "validation"):
            if feature_report["output_sha256"][split + ".csv"] != self.plan["development_input_sha256"][split]:
                raise ValueError("Development history changed after the model was frozen.")
        with source.open(encoding="utf-8", newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        if len(self.rows) != preparation["included_rows"] or not self.rows:
            raise ValueError("Prepared history has an unexpected row count.")
        self.names = sorted({r[side] for r in self.rows for side in ("home_team", "away_team")})
        self.dataset_latest_date = max(r["date"] for r in self.rows)
        self.minimum_history = schema["minimum_prior_appearances_each_team"]

    def resolve_team(self, name):
        candidates = [team for team in self.names if team.casefold() == name.strip().casefold()]
        if len(candidates) == 1:
            return candidates[0]
        suggestions = difflib.get_close_matches(name, self.names, n=3, cutoff=0.4)
        raise ValueError("Unknown or ambiguous team: " + repr(name) + ". Suggestions: " + ", ".join(suggestions) + ". Use --list-teams for dataset names.")

    def check_date(self, match_date):
        earliest = date.fromisoformat(self.plan["test_start"])
        if match_date < earliest:
            raise ValueError("This model was fitted through " + self.plan["development_last_date"] + "; choose " + earliest.isoformat() + " or later for prediction/replay.")

    def available_teams(self, match_date):
        self.check_date(match_date)
        states, _, _ = history_before(self.rows, match_date)
        return [{"team": name, "prior_matches": state.appearances, "last_recorded_match": state.last_date.isoformat()}
                for name, state in sorted(states.items()) if state.appearances >= self.minimum_history]

    def predict(self, home, away, match_date, neutral=False, tournament=None, replay=False):
        self.check_date(match_date)
        home, away = self.resolve_team(home), self.resolve_team(away)
        if home == away:
            raise ValueError("Choose two different teams.")
        actual = None
        if replay:
            if neutral:
                raise ValueError("Replay uses the recorded venue. Remove --neutral.")
            matches = [r for r in self.rows if r["date"] == match_date.isoformat() and r["home_team"] == home and r["away_team"] == away]
            if tournament is not None:
                matches = [r for r in matches if r["tournament"].casefold() == tournament.casefold()]
            if not matches:
                raise ValueError("No recorded fixture with these teams, date and roles. Check the source record, including home/away order.")
            if len(matches) != 1:
                raise ValueError("Multiple recorded fixtures match. Specify the recorded --tournament.")
            context = dict(matches[0])
            neutral = context["neutral"] == "TRUE"
            tournament = context["tournament"]
            actual = {"home_goals": int(context["home_score"]), "away_goals": int(context["away_score"]), "outcome": context["outcome"]}
        else:
            tournament = tournament or "Friendly"
            if not tournament.strip():
                raise ValueError("Tournament name must not be empty.")
            known = {r["tournament"].casefold(): r["tournament"] for r in self.rows}
            if tournament.strip().casefold() not in known:
                raise ValueError("Use a tournament name found in the dataset, such as Friendly.")
            tournament = known[tournament.strip().casefold()]
            context = {"date": match_date.isoformat(), "home_team": home, "away_team": away, "neutral": "TRUE" if neutral else "FALSE", "tournament": tournament}
        states, meetings, history_end = history_before(self.rows, match_date)
        for name in (home, away):
            if states[name].appearances < self.minimum_history:
                raise ValueError(name + " has only " + str(states[name].appearances) + " prior matches; at least " + str(self.minimum_history) + " are required.")
        values, latest = prematch_features(context, states, meetings)
        values = {name: round(values[name], 8) if isinstance(values[name], float) else values[name] for name in FEATURES}
        X = pd.DataFrame([values], columns=FEATURES).astype(float)
        with threadpool_limits(limits=2):
            probabilities, rates = predict_item(self.frozen["selected_model"], X)
        home_rate, away_rate = map(float, rates[0])
        request = {"mode": "recorded_match_replay" if replay else "hypothetical_scenario", "date": match_date.isoformat(), "home_team": home, "away_team": away, "neutral": bool(neutral), "tournament": tournament}
        result = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "request": request,
            "model": {"name": self.plan["selected_on_validation"], "trained_through": self.plan["development_last_date"], "sha256": self.plan["frozen_model_sha256"], "target_definition": self.plan["target_definition"], "calibration": self.plan["calibration"]},
            "data": {"history_sha256": self.history_sha256, "dataset_latest_date": self.dataset_latest_date, "latest_included_history_date": history_end, "latest_relevant_history_date": latest,
                     "home_prior_matches": states[home].appearances, "away_prior_matches": states[away].appearances,
                     "home_last_recorded_match": states[home].last_date.isoformat(), "away_last_recorded_match": states[away].last_date.isoformat(),
                     "home_days_since_recorded_match": (match_date - states[home].last_date).days, "away_days_since_recorded_match": (match_date - states[away].last_date).days},
            "prediction": {"expected_home_goals": home_rate, "expected_away_goals": away_rate,
                           "outcome_probabilities": {name: float(probabilities[0, i]) for i, name in enumerate(CLASSES)},
                           "most_likely_outcome": CLASSES[int(probabilities[0].argmax())], **distribution_details(home_rate, away_rate)},
            "actual_recorded_result": actual, "feature_values": values,
        }
        return result


def show(result):
    request, prediction, data = result["request"], result["prediction"], result["data"]
    home, away = request["home_team"], request["away_team"]
    print("FOOTBALL MATCH PREDICTION")
    print("Mode:", "RECORDED MATCH REPLAY" if request["mode"] == "recorded_match_replay" else "HYPOTHETICAL SCENARIO")
    print("Fixture:", home, "vs", away, "|", request["date"])
    print("Venue:", "Neutral" if request["neutral"] else "Home advantage for " + home, "| Tournament:", request["tournament"])
    print("Model:", result["model"]["name"], "| Trained through:", result["model"]["trained_through"])
    print("History included through:", data["latest_included_history_date"])
    print(home + ":", data["home_prior_matches"], "prior matches | Last recorded:", data["home_last_recorded_match"])
    print(away + ":", data["away_prior_matches"], "prior matches | Last recorded:", data["away_last_recorded_match"])
    print(f"Expected goals: {home} {prediction['expected_home_goals']:.2f} - {away} {prediction['expected_away_goals']:.2f}")
    for outcome, label in (("home_win", home + " win"), ("draw", "Draw"), ("away_win", away + " win")):
        print(label + ":", f"{prediction['outcome_probabilities'][outcome]:.2%}")
    print("Most likely scorelines (" + home + " - " + away + "):")
    for item in prediction["top_scorelines"]:
        print(f"  {item['home_goals']}-{item['away_goals']}: {item['probability']:.2%}")
    if result["actual_recorded_result"] is not None:
        actual = result["actual_recorded_result"]
        print("Actual recorded result:", str(actual["home_goals"]) + "-" + str(actual["away_goals"]), "|", actual["outcome"])
    print("Target:", result["model"]["target_definition"])


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--home")
    parser.add_argument("--away")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), metavar="YYYY-MM-DD")
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument("--tournament")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--list-teams", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.list_teams and (not args.home or not args.away):
        parser.error("Specify --home and --away, or use --list-teams.")
    try:
        predictor = MatchPredictor(args.project_root)
        if args.list_teams:
            available = predictor.available_teams(args.date)
            print("Teams with sufficient prior history for", args.date.isoformat(), ":", len(available))
            for item in available:
                print(item["team"] + " | prior matches=" + str(item["prior_matches"]) + " | last recorded=" + item["last_recorded_match"])
            return 0
        result = predictor.predict(args.home, args.away, args.date, args.neutral, args.tournament, args.replay)
        show(result)
        identity = {"request": result["request"], "model": result["model"]["sha256"], "history": result["data"]["history_sha256"]}
        request_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        output = args.output.expanduser() if args.output else predictor.root / "reports/predictions" / ("prediction_" + args.date.isoformat() + "_" + request_hash + ".json")
        if output.resolve().is_relative_to((predictor.root / "data").resolve()) or output.resolve().is_relative_to((predictor.root / "models").resolve()):
            raise ValueError("Save prediction reports outside the data/ and models/ directories.")
        if output.suffix.lower() != ".json":
            raise ValueError("Prediction report filenames must end in .json.")
        if output.is_file():
            previous = json.loads(output.read_text(encoding="utf-8"))
            if not isinstance(previous, dict) or "request" not in previous or "prediction" not in previous:
                raise ValueError("The output path belongs to another project file; choose a prediction report filename.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print("Saved prediction:", output.resolve())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, csv.Error) as exc:
        parser.exit(2, "ERROR: " + str(exc) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
