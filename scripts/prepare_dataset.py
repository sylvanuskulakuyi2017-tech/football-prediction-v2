#!/usr/bin/env python3
"""Prepare a chronological modelling table without changing the raw dataset.

Keep validate_dataset.py beside this file in scripts/. Run from the project:
    python scripts/prepare_dataset.py

The default modelling window starts on 1990-01-01. Scores are recorded final
scores, including extra time and excluding penalty shoot-outs. Derived labels
are targets, not model features. No models or train/test splits are created.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

try:
    from validate_dataset import EXPECTED_COLUMNS, inspect_csv
except ImportError:
    sys.exit("ERROR: Put validate_dataset.py and prepare_dataset.py together in scripts/.")


TARGET_COLUMNS = (
    "home_score", "away_score", "outcome", "total_goals", "both_teams_scored",
)
PREMATCH_COLUMNS = (
    "date", "home_team", "away_team", "tournament", "city", "country", "neutral",
)
OUTPUT_COLUMNS = (
    "match_id", "source_csv_line", "date", "home_team", "away_team",
    "tournament", "city", "country", "neutral", *TARGET_COLUMNS,
)


def prepare_rows(path, start_date, as_of):
    """Validate all raw rows, then select the explicitly requested date window."""
    validation, _ = inspect_csv(path, as_of)
    if validation["errors"]:
        raise ValueError("Raw validation failed: " + json.dumps(validation["errors"]))
    rows = []
    excluded = 0
    fixture_lines = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            item = {column: raw[column].strip() for column in EXPECTED_COLUMNS}
            match_date = date.fromisoformat(item["date"])
            if match_date < start_date:
                excluded += 1
                continue
            fixture_key = tuple(item[column] for column in ("date", "home_team", "away_team", "tournament"))
            if fixture_key in fixture_lines:
                raise ValueError(
                    "Repeated fixture inside the modelling window at CSV lines "
                    + str(fixture_lines[fixture_key]) + " and " + str(reader.line_num)
                    + ": " + repr(fixture_key) + ". Review these records before proceeding."
                )
            fixture_lines[fixture_key] = reader.line_num
            home, away = int(item["home_score"]), int(item["away_score"])
            item["neutral"] = item["neutral"].upper()
            identity = [item[column] for column in PREMATCH_COLUMNS]
            item["match_id"] = hashlib.sha256(
                json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            item["source_csv_line"] = reader.line_num
            item["home_score"], item["away_score"] = home, away
            item["outcome"] = "home_win" if home > away else "away_win" if away > home else "draw"
            item["total_goals"] = home + away
            item["both_teams_scored"] = "TRUE" if home > 0 and away > 0 else "FALSE"
            rows.append(item)
    if hashlib.sha256(path.read_bytes()).hexdigest() != validation["input_sha256"]:
        raise ValueError("The raw CSV changed while it was being read. Run the script again.")
    if not rows:
        raise ValueError("The requested date window contains no matches.")
    rows.sort(key=lambda row: tuple(row[column] for column in ("date", "home_team", "away_team", "tournament")))
    summary = {
        "preparation_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(path.resolve()),
        "input_sha256": validation["input_sha256"],
        "as_of_date": as_of.isoformat(),
        "start_date": start_date.isoformat(),
        "raw_rows": validation["rows_read"],
        "excluded_before_start_date": excluded,
        "included_rows": len(rows),
        "first_match": rows[0]["date"],
        "last_match": rows[-1]["date"],
        "outcome_counts": dict(Counter(row["outcome"] for row in rows)),
        "raw_validation_status": validation["status"],
        "raw_validation_warnings": validation["warnings"],
        "repeated_fixture_keys_inside_window": 0,
        "raw_data_modified": False,
        "target_definition": validation["target_definition"],
        "target_columns_do_not_use_as_features": list(TARGET_COLUMNS),
        "prematch_context_columns": list(PREMATCH_COLUMNS),
        "context_note": "Source metadata is not an independently verified pre-kickoff availability archive.",
        "row_order": "date, home_team, away_team, tournament",
        "same_date_policy_for_future_features": "Only use results from strictly earlier dates; row order within a date does not establish kickoff order.",
        "team_note": "Names include historical and non-FIFA teams; presence does not establish current activity or sufficient recent history.",
        "window_rationale": "Initial engineering choice for a modern-era prototype, not a performance-selected cutoff.",
        "split_status": "Not yet split. Chronological train/validation/test boundaries must be defined before fitting models.",
    }
    return rows, summary


def write_outputs(rows, summary, output_dir, report_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    matches_path = output_dir / "matches.csv"
    with matches_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary["output_file"] = str(matches_path.resolve())
    summary["output_sha256"] = hashlib.sha256(matches_path.read_bytes()).hexdigest()
    teams = {}
    for row in rows:
        for column in ("home_team", "away_team"):
            name = row[column]
            if name not in teams:
                teams[name] = {"team": name, "matches": 0, "first_match": row["date"], "last_match": row["date"]}
            teams[name]["matches"] += 1
            teams[name]["last_match"] = row["date"]
    summary["team_names_in_window"] = len(teams)
    coverage_path = report_dir / "processed_team_coverage.csv"
    with coverage_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("team", "matches", "first_match", "last_match"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(teams.values(), key=lambda row: row["team"].casefold()))
    report_path = report_dir / "dataset_preparation.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return matches_path, coverage_path, report_path


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/raw/international_results")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(1990, 1, 1), metavar="YYYY-MM-DD")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today(), metavar="YYYY-MM-DD")
    args = parser.parse_args()
    if args.start_date > args.as_of:
        parser.error("--start-date must not be after --as-of")
    path = args.data_dir.expanduser() / "results.csv"
    try:
        rows, summary = prepare_rows(path, args.start_date, args.as_of)
        paths = write_outputs(rows, summary, args.output_dir.expanduser(), args.report_dir.expanduser())
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        parser.exit(2, "ERROR: " + str(exc) + "\n")
    print("FOOTBALL DATASET PREPARATION")
    print("Status: READY FOR FEATURE ENGINEERING")
    print("Raw rows:", summary["raw_rows"])
    print("Excluded before", summary["start_date"] + ":", summary["excluded_before_start_date"])
    print("Prepared matches:", summary["included_rows"])
    print("Team names in window:", summary["team_names_in_window"])
    print("Date range:", summary["first_match"], "to", summary["last_match"])
    print("Repeated fixture keys inside window:", summary["repeated_fixture_keys_inside_window"])
    print("Outcome counts:", summary["outcome_counts"])
    for label, result in zip(("Matches", "Team coverage", "Preparation report"), paths):
        print(label + ":", result.resolve())
    print("Raw data unchanged. No models trained. Train/validation/test split pending.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
