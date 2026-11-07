# International Match Lab

Football Prediction v2 is an interactive portfolio project for men's international
football: explore supported team pairings, replay historical forecasts, and inspect
the saved model benchmark.

## What you can explore

- **Match scenario:** choose two eligible teams, a date, competition and venue.
- **Historical replay:** reconstruct a held-out forecast before revealing the result.
- **Model performance:** inspect the frozen evaluation and year-by-year results.
- **One goal distribution, several views:** outcome probabilities, expected goals,
  likely scorelines, three-or-more goals and both teams scoring.
- **Evidence behind the forecast:** prior match counts, available history and data cutoff.

The project combines established methods into a reproducible application. It does
not claim a new prediction algorithm. Additional probabilities are derived from the
same two goal models; they are not independently trained or separately evaluated targets.

## Run locally

Use **Linux CPython 3.14** for this release. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements.lock.txt
python -m pip check
python scripts/check_release.py --ui
python scripts/start_server.py
```

Open http://localhost:8501. The saved app, model and history are bundled; no API key,
database or training job is required. Python/package details are recorded in
`runtime_environment.json`; the lock contains only the app's installed dependency closure.

## Saved benchmark

Validation selected **PoissonGradientBoosting**. Its settings were frozen,
then refitted on **27,552 train + validation matches** through
**2022-12-30**. The reserved period contains
**4,081 eligible matches**, 2023-01-02–2026-10-06.

| Model | Accuracy | Log loss ↓ | Brier score ↓ | Goal MAE ↓ |
| --- | ---: | ---: | ---: | ---: |
| PoissonGradientBoosting | 60.57% | 0.8691 | 0.5107 | 0.9362 |
| DevelopmentPrior | 47.17% | 1.0540 | 0.6361 | — |
| DevelopmentConstantGoalsPoisson | 47.17% | 1.0581 | 0.6385 | 1.0851 |

The selected model exceeds the outcome-prior reference by
**13.40 percentage points** in accuracy.
Metrics come directly from `reports/final_test_report.json`. Overall accuracy is
not the probability that an individual forecast will be correct.

## Data and method

The bundled snapshot contains **32,821 prepared matches** and
**327 team names**, 1990-01-12–2026-10-06.
Each team needs at least **5** earlier appearances.

| Split | Eligible matches | First date | Last date |
| --- | ---: | --- | --- |
| Train | 23,980 | 1990-03-08 | 2018-12-31 |
| Validation | 3,572 | 2019-01-02 | 2022-12-30 |
| Test | 4,081 | 2023-01-02 | 2026-10-06 |

The 30 numeric features describe venue, competition type,
custom Elo, prior form, rest and head-to-head history. Date-batched updates ensure
features use strictly earlier calendar dates. Missing history remains missing until
the selected training pipeline handles it.

Two goal regressors produce rates for independent Poisson distributions. Outcome
probabilities use the full score support; the small displayed score grid is not
renormalized into a complete distribution. No post-fit calibration was applied.

**Target:** recorded final score, including extra time and excluding penalty
shoot-outs. It is not a uniform 90-minute result or a prediction of who advances.
The fitted model ends in 2022; later recorded results update pre-match history
without refitting its parameters. This distinction is displayed in the app.

## Repository guide

| Location | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `scripts/` | Data pipeline, prediction and release checks |
| `data/raw/international_results/results.csv` | Exact source snapshot |
| `data/processed/matches.csv` | Prepared chronological history |
| `models/frozen/frozen_model.joblib` | Validation-selected, development-fitted model |
| `reports/` | Saved protocol, metrics and replay forecasts |
| `release_manifest.json` | SHA-256 fingerprints for bundled assets |
| `Dockerfile` | Railway container build |

[Model details and limitations](docs/MODEL_CARD.md) ·
[Reproduce the pipeline](docs/REPRODUCE.md) ·
[Publish on GitHub](docs/GITHUB.md) ·
[Deploy on Railway](docs/RAILWAY.md)

## Data credit and licence

Data: [Mart Jürisoo's international football results](https://github.com/martj42/international_results),
released under CC0-1.0. See [DATA_NOTICE.md](DATA_NOTICE.md).
Project code and original documentation use the MIT licence in [LICENSE](LICENSE).
Third-party dependencies retain their own licences.
# football-prediction-v2
