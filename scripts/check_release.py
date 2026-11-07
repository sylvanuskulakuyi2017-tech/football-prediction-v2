#!/usr/bin/env python3
"""Verify the frozen assets and replay one stored forecast; never fit a model."""
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import sys
from datetime import date
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_release(root, ui=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "release_manifest.json").read_text())
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current_python != manifest["python_major_minor"]:
        raise ValueError("Use Python " + manifest["python_major_minor"] + " for this saved release.")
    for name, expected in manifest["runtime_packages"].items():
        if metadata.version(name) != expected:
            raise ValueError("Restore the locked dependency version for " + name)
    for name, expected in manifest["asset_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError("A frozen release asset changed or is missing: " + name)
    sys.path.insert(0, str(root / "scripts"))
    from predict_match import MatchPredictor
    predictor = MatchPredictor(root)
    with (root / "reports/final_test_predictions.csv").open(encoding="utf-8", newline="") as stream:
        saved = next((row for row in csv.DictReader(stream)
                      if row["model"] == predictor.plan["selected_on_validation"]), None)
    if saved is None:
        raise ValueError("The selected model has no saved replay forecasts.")
    result = predictor.predict(saved["home_team"], saved["away_team"], date.fromisoformat(saved["date"]), replay=True)
    probabilities = result["prediction"]["outcome_probabilities"]
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Invalid forecast probabilities.")
    if abs(sum(probabilities.values()) - 1) > 1e-10:
        raise ValueError("Forecast probabilities do not sum to one.")
    for label, value in probabilities.items():
        if abs(value - float(saved["prob_" + label])) > 1e-8:
            raise ValueError("Replayed probabilities differ from the saved benchmark: " + label)
    for side in ("home", "away"):
        if abs(result["prediction"]["expected_" + side + "_goals"] - float(saved["expected_" + side + "_goals"])) > 1e-8:
            raise ValueError("Replayed expected goals differ from the saved benchmark.")
    if ui:
        from streamlit.testing.v1 import AppTest
        os.environ["FOOTBALL_PROJECT_ROOT"] = str(root)
        app = AppTest.from_file(str(root / "app.py"), default_timeout=90).run()
        for mode in (None, "Historical replay", "Model performance"):
            if mode:
                app.radio(key="workspace").set_value(mode).run()
            if app.exception or app.error:
                details = [str(item.value) for item in (*app.exception, *app.error)]
                raise ValueError("Streamlit check failed: " + "; ".join(details))
        print("All three Streamlit views loaded successfully.")
    print("RELEASE CHECK PASSED: assets, dependencies and one stored forecast verified.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui", action="store_true", help="Also load the three app views without a browser.")
    args = parser.parse_args()
    try:
        check_release(Path(__file__).resolve().parent.parent, ui=args.ui)
    except Exception as exc:
        parser.exit(2, "ERROR: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
