# Model card

## Intended use

An educational portfolio application for exploring men's international football
forecasts and a saved historical evaluation. Hypothetical scenarios do not establish
that a fixture is scheduled. There are no live fixture, odds or lineup feeds.

## Model and protocol

- Selected model: PoissonGradientBoosting.
- Selection: lowest validation outcome log loss among goal-producing baselines.
- Refit period: 27,552 matches through 2022-12-30.
- Test: 4,081 matches, 2023-01-02–2026-10-06.
- Test-based reselection: False.
- Probability calibration: No post-fit probability calibration applied.
- Target: Recorded final score, including extra time, excluding penalty shoot-outs

Histories are updated between calendar dates. During test/replay, earlier test-period
results can inform later features; there is no within-period parameter refitting.
This is chronological evaluation with updated history, not a forecast of the entire
test period made on its first day. Same-day matches share the pre-date state because
kickoff times are unavailable. Training warmup and eligibility are documented in
the saved feature schema.

## Limitations

Independent Poisson scoring is an approximation. It does not represent lineup
changes, injuries, tactical interactions or every source of dependence in goals.
Exact scoreline probabilities are small and should be read alongside the full
outcome distribution. Extra time is mixed into the recorded-score target, so a
separate 90-minute interpretation is not justified.

Historical team names and non-FIFA teams may be present. Coverage and recency vary;
the app reports the available history. Tournament input is represented by the
existing friendly indicator, rather than a separately learned parameter for every
competition. The fixed model may age as team strength and football conditions change.

Outcome accuracy alone does not establish calibration or performance for every team,
competition or derived target. The headline benchmark belongs to this frozen model
and snapshot. A future refit needs its own version and future evaluation; the current
test period must not be recycled as an untouched test after tuning to its results.

Only load the bundled trusted joblib artifact. Hash checks detect accidental changes;
they are not a signature establishing the trustworthiness of an unknown model file.
