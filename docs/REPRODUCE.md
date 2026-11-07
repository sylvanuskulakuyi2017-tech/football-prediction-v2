# Reproduce the pipeline

Install the locked environment described in the README first. The bundled frozen
model can be used immediately; training is optional. Run the following commands
from the repository root to rebuild in experiments/rebuild, keeping the published
assets and benchmark intact. Use an empty rebuild folder for each independent run.

```bash
python scripts/validate_dataset.py --data-dir data/raw/international_results --output-dir experiments/rebuild/reports --as-of 2026-11-07
python scripts/prepare_dataset.py --data-dir data/raw/international_results --output-dir experiments/rebuild/data --report-dir experiments/rebuild/reports --start-date 1990-01-01 --as-of 2026-11-07
python scripts/build_features.py --matches experiments/rebuild/data/matches.csv --preparation-report experiments/rebuild/reports/dataset_preparation.json --output-dir experiments/rebuild/data --report-dir experiments/rebuild/reports --validation-start 2019-01-01 --test-start 2023-01-01 --min-history 5
python scripts/train_baselines.py --data-dir experiments/rebuild/data --report-dir experiments/rebuild/reports --model-dir experiments/rebuild/models/baselines
python scripts/evaluate_final_test.py --data-dir experiments/rebuild/data --report-dir experiments/rebuild/reports --baseline-bundle experiments/rebuild/models/baselines/baseline_bundle.joblib --frozen-model experiments/rebuild/models/frozen/frozen_model.joblib
```

Rebuilding reproduces the existing protocol. Do not use the reserved test period
to choose new hyperparameters and then describe it as untouched. The exact source
snapshot, package lock and feature definitions matter; downloading today's latest
results.csv may produce a different dataset. Hardware, Python patch versions and
numerical libraries can still cause small numerical differences.

Generated feature/split CSVs, training-only baseline bundles and local prediction
exports are excluded from the release because they can be regenerated. Final
benchmark reports and replay forecasts are included unchanged.
