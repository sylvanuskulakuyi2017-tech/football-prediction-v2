#!/usr/bin/env python3
"""Validate international_results/results.csv using Python's standard library.

Place this file in the project's scripts/ directory and run:
    python scripts/validate_dataset.py

Reads the raw CSV without changing it. Writes dataset_validation.json and
team_coverage.csv under reports/. This checks data structure and consistency;
it does not certify historical accuracy or model readiness.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path


EXPECTED_COLUMNS = (
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
)
REQUIRED_VALUES = (
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "neutral",
)
SOURCE_URL = "https://github.com/martj42/international_results"


def inspect_csv(path, as_of):
    errors, warnings = Counter(), Counter()
    examples = defaultdict(list)
    teams = {}
    outcomes = Counter()
    tournaments = Counter()
    seen_rows, seen_fixtures = set(), {}
    dates = []
    neutral_count = 0
    total_rows = 0
    content = path.read_bytes()
    summary = {
        "validator_version": "0.1.0",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": as_of.isoformat(),
        "source_url": SOURCE_URL,
        "input_file": str(path.resolve()),
        "input_sha256": hashlib.sha256(content).hexdigest(),
        "input_size_bytes": len(content),
        "target_definition": "Recorded final score, including extra time, excluding penalty shoot-outs",
        "coverage_definition": "Rows with valid required fields; exact duplicate rows are counted once",
    }

    def issue(bucket, code, line_number):
        bucket[code] += 1
        if len(examples[code]) < 5:
            examples[code].append(line_number)

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        summary["columns"] = columns
        missing = sorted(set(EXPECTED_COLUMNS) - set(columns))
        summary["missing_columns"] = missing
        summary["extra_columns"] = sorted(set(columns) - set(EXPECTED_COLUMNS))
        if len(columns) != len(set(columns)):
            errors["duplicate_column_names"] = 1
        if missing:
            errors["missing_required_columns"] = len(missing)
        if not missing and not errors:
            for row in reader:
                total_rows += 1
                line_number = reader.line_num
                if None in row or any(row.get(c) is None for c in EXPECTED_COLUMNS):
                    issue(errors, "incorrect_number_of_fields", line_number)
                    continue
                values = {c: row[c].strip() for c in EXPECTED_COLUMNS}
                row_error = False
                if any(not values[c] for c in REQUIRED_VALUES):
                    issue(errors, "missing_required_values", line_number)
                    continue
                try:
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", values["date"]):
                        raise ValueError("Expected YYYY-MM-DD")
                    match_date = date.fromisoformat(values["date"])
                    if match_date > as_of:
                        issue(errors, "future_dated_result", line_number)
                        row_error = True
                except ValueError:
                    issue(errors, "invalid_date", line_number)
                    row_error = True
                score_values = (values["home_score"], values["away_score"])
                if any(not re.fullmatch(r"[0-9]+", s) for s in score_values):
                    issue(errors, "invalid_nonnegative_integer_score", line_number)
                    row_error = True
                if values["neutral"].lower() not in ("true", "false"):
                    issue(errors, "invalid_neutral_flag", line_number)
                    row_error = True
                if values["home_team"].casefold() == values["away_team"].casefold():
                    issue(errors, "same_team_on_both_sides", line_number)
                    row_error = True
                if row_error:
                    continue
                if any(row[c] != values[c] for c in EXPECTED_COLUMNS):
                    issue(warnings, "leading_or_trailing_whitespace", line_number)
                if not values["city"] or not values["country"]:
                    issue(warnings, "missing_venue_metadata", line_number)
                record_key = tuple(values[c] for c in EXPECTED_COLUMNS)
                if record_key in seen_rows:
                    issue(errors, "exact_duplicate_row", line_number)
                    continue
                seen_rows.add(record_key)
                home_goals, away_goals = map(int, score_values)
                fixture_key = tuple(values[c] for c in ("date", "home_team", "away_team", "tournament"))
                if fixture_key in seen_fixtures:
                    issue(warnings, "repeated_fixture_key_review_needed", line_number)
                    if seen_fixtures[fixture_key] != (home_goals, away_goals):
                        issue(warnings, "conflicting_scores_review_needed", line_number)
                else:
                    seen_fixtures[fixture_key] = (home_goals, away_goals)
                is_neutral = values["neutral"].lower() == "true"
                neutral_count += int(is_neutral)
                dates.append(match_date)
                tournaments[values["tournament"]] += 1
                result = "home_win" if home_goals > away_goals else "away_win" if away_goals > home_goals else "draw"
                outcomes[result] += 1
                for side in ("home_team", "away_team"):
                    team = values[side]
                    if team not in teams:
                        teams[team] = {
                            "team": team, "matches": 0,
                            "first_match": match_date.isoformat(),
                            "last_match": match_date.isoformat(),
                            "matches_since_2000": 0, "matches_since_2020": 0,
                        }
                    item = teams[team]
                    item["matches"] += 1
                    item["first_match"] = min(item["first_match"], match_date.isoformat())
                    item["last_match"] = max(item["last_match"], match_date.isoformat())
                    item["matches_since_2000"] += int(match_date >= date(2000, 1, 1))
                    item["matches_since_2020"] += int(match_date >= date(2020, 1, 1))
    if not total_rows and not errors:
        errors["empty_dataset"] = 1
    if dates and dates != sorted(dates):
        warnings["input_not_chronologically_sorted"] = 1
    summary.update({
        "status": "FAIL" if errors else "REVIEW" if warnings else "PASS",
        "rows_read": total_rows,
        "valid_unique_rows": len(dates),
        "team_names": len(teams),
        "earliest_match": min(dates).isoformat() if dates else None,
        "latest_match": max(dates).isoformat() if dates else None,
        "matches_since_1990": sum(d >= date(1990, 1, 1) for d in dates),
        "matches_since_2000": sum(d >= date(2000, 1, 1) for d in dates),
        "matches_since_2020": sum(d >= date(2020, 1, 1) for d in dates),
        "neutral_matches": neutral_count,
        "outcomes": dict(outcomes),
        "tournaments": dict(tournaments.most_common()),
        "errors": dict(errors),
        "warnings": dict(warnings),
        "example_csv_line_numbers": dict(examples),
    })
    return summary, sorted(teams.values(), key=lambda r: r["team"].casefold())


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/raw/international_results")
    parser.add_argument("--output-dir", type=Path, default=root / "reports")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today(), metavar="YYYY-MM-DD")
    args = parser.parse_args()
    path = args.data_dir.expanduser() / "results.csv"
    if not path.is_file():
        parser.exit(2, "ERROR: results.csv not found at " + str(path.resolve()) + "\n")
    try:
        summary, teams = inspect_csv(path, args.as_of)
    except (OSError, UnicodeError, csv.Error) as exc:
        parser.exit(2, "ERROR: Could not read the dataset: " + str(exc) + "\n")
    companions = ("README.md", "LICENSE", "shootouts.csv", "former_names.csv", "goalscorers.csv")
    summary["companion_files_present"] = {name: (args.data_dir / name).is_file() for name in companions}
    output = args.output_dir.expanduser()
    try:
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "dataset_validation.json"
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (output / "team_coverage.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("team", "matches", "first_match", "last_match", "matches_since_2000", "matches_since_2020"))
            writer.writeheader()
            writer.writerows(teams)
    except OSError as exc:
        parser.exit(2, "ERROR: Could not write reports: " + str(exc) + "\n")
    print("FOOTBALL DATASET VALIDATION")
    print("Status:", summary["status"])
    print("Rows read:", summary["rows_read"])
    print("Valid unique rows:", summary["valid_unique_rows"])
    print("Team names:", summary["team_names"])
    print("Date range:", summary["earliest_match"], "to", summary["latest_match"])
    print("Matches since 1990:", summary["matches_since_1990"])
    print("Errors:", summary["errors"] or "None")
    print("Warnings:", summary["warnings"] or "None")
    print("Report:", report_path.resolve())
    print("Team coverage:", (output / "team_coverage.csv").resolve())
    print("Raw data unchanged. No models trained.")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
