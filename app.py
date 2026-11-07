"""Local Streamlit interface for Football Prediction v2.

Save at the project root beside scripts/, data/, models/ and reports/.
Run: python -m streamlit run app.py --theme.base light --theme.primaryColor '#007F73'
The app reads the existing frozen model; it does not train or rescore it.
"""

import hashlib
import json
import os
import sys
import threading
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson


ROOT = Path(os.environ.get("FOOTBALL_PROJECT_ROOT", str(Path(__file__).resolve().parent))).expanduser().resolve()
sys.path.insert(0, str(ROOT / "scripts"))
st.set_page_config(page_title="International Match Lab", layout="wide", initial_sidebar_state="expanded")

try:
    from predict_match import MatchPredictor
    from train_baselines import sha256
except ImportError as exc:
    st.error("Save app.py in the project root and keep the earlier Python files in scripts/.")
    st.code(str(exc))
    st.stop()

HOME, DRAW, AWAY = "#1675B9", "#78889A", "#007F73"
st.markdown("""<style>
/* Leave room for Streamlit's fixed toolbar above the page header. */
.block-container {max-width:1220px; padding-top:5rem; padding-bottom:3rem;}
[data-testid="stSidebar"] {border-right:1px solid #DDE6ED;}
h1 {font-family:Georgia,serif; letter-spacing:-0.045em; font-size:2.8rem!important;}
h2,h3 {letter-spacing:-0.02em;}
[data-testid="stMetricValue"] {font-size:1.9rem;}
[data-testid="stMetricLabel"] {font-size:.88rem;}
.eyebrow {color:#007F73; font-size:.74rem; letter-spacing:.16em; font-weight:750; margin:0 0 .4rem;}
.rule {height:1px; background:#DDE6ED; margin:1.35rem 0 1.5rem;}
</style>""", unsafe_allow_html=True)


def file_signature():
    names = ["models/frozen/frozen_model.joblib", "reports/final_test_plan.json", "reports/final_test_report.json", "reports/dataset_preparation.json", "reports/feature_engineering_report.json", "data/processed/matches.csv", "data/processed/feature_schema.json", "reports/final_test_predictions.csv"]
    return tuple((name, (ROOT / name).stat().st_mtime_ns, (ROOT / name).stat().st_size) for name in names)


@st.cache_resource(show_spinner=False)
def resources(root, signature):
    return MatchPredictor(root), threading.RLock()


@st.cache_data(show_spinner=False, max_entries=16)
def eligible_teams(root, signature, day):
    predictor, lock = resources(root, signature)
    with lock:
        return predictor.available_teams(date.fromisoformat(day))


@st.cache_data(show_spinner=False, max_entries=64)
def forecast(root, signature, home, away, day, neutral, tournament, replay):
    predictor, lock = resources(root, signature)
    with lock:
        return predictor.predict(home, away, date.fromisoformat(day), neutral=neutral, tournament=tournament, replay=replay)


@st.cache_data(show_spinner=False)
def benchmark_report(root, signature):
    return json.loads((Path(root) / "reports/final_test_report.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def replay_index(root, signature):
    root = Path(root)
    report = benchmark_report(str(root), signature)
    path = root / "reports/final_test_predictions.csv"
    if sha256(path) != report["predictions_sha256"]:
        raise ValueError("Saved benchmark predictions do not match their report.")
    frame = pd.read_csv(path, dtype={"match_id": str})
    frame = frame.loc[frame["model"].eq(report["selected_on_validation"])].copy()
    if len(frame) != report["test_rows"] or frame["match_id"].duplicated().any():
        raise ValueError("Saved replay records have an unexpected size or duplicate IDs.")
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame.sort_values(["date", "home_team", "away_team"], ascending=[False, True, True])


def valid_select(label, options, key, preferred=None, **kwargs):
    if not options:
        st.info("No records are available for these settings.")
        st.stop()
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = preferred if preferred in options else options[0]
    return st.sidebar.selectbox(label, options, key=key, **kwargs)


def token(*values):
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def chart_style(chart):
    return chart.configure_view(stroke=None).configure_axis(labelFontSize=12, titleFontSize=12, gridColor="#E7EDF3").configure_legend(labelFontSize=11, titleFontSize=11)


def probability_chart(result):
    request, probabilities = result["request"], result["prediction"]["outcome_probabilities"]
    labels = [request["home_team"] + " win", "Draw", request["away_team"] + " win"]
    source = pd.DataFrame({"Outcome": labels, "Probability": [probabilities[name] for name in ("home_win", "draw", "away_win")]})
    base = alt.Chart(source).encode(
        y=alt.Y("Outcome:N", sort=labels, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1.08]), title=None, axis=alt.Axis(format=".0%", values=[0, .25, .5, .75, 1])),
        tooltip=["Outcome:N", alt.Tooltip("Probability:Q", format=".2%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, size=26).encode(color=alt.Color("Outcome:N", scale=alt.Scale(domain=labels, range=[HOME, DRAW, AWAY]), legend=None))
    labels_chart = base.mark_text(align="left", dx=7, color="#22384D", fontSize=13).encode(text=alt.Text("Probability:Q", format=".1%"))
    return chart_style((bars + labels_chart).properties(height=165))


def heatmap(result):
    p = result["prediction"]
    rows = [{"Home goals": h, "Away goals": a, "Probability": float(poisson.pmf(h, p["expected_home_goals"]) * poisson.pmf(a, p["expected_away_goals"]))} for h in range(7) for a in range(7)]
    base = alt.Chart(pd.DataFrame(rows)).encode(
        x=alt.X("Away goals:O", title=result["request"]["away_team"] + " goals", sort=list(range(7))),
        y=alt.Y("Home goals:O", title=result["request"]["home_team"] + " goals", sort=list(range(7))),
        tooltip=["Home goals:O", "Away goals:O", alt.Tooltip("Probability:Q", format=".2%")],
    )
    blocks = base.mark_rect(stroke="white", strokeWidth=2).encode(color=alt.Color("Probability:Q", scale=alt.Scale(range=["#EFF7F6", "#007F73"]), legend=None))
    maximum = max(row["Probability"] for row in rows)
    text = base.mark_text(fontSize=11).encode(text=alt.Text("Probability:Q", format=".1%"), color=alt.condition(alt.datum.Probability > maximum * .6, alt.value("white"), alt.value("#294A56")))
    return chart_style((blocks + text).properties(height=300))


def evidence_table(result):
    r, d, features = result["request"], result["data"], result["feature_values"]
    records = []
    definitions = [("Recorded prior matches", "count"), ("Last recorded match", "last"), ("Days since last recorded match", "days"), ("Goals scored per match, last 5", "goals_for_mean_5"), ("Goals conceded per match, last 5", "goals_against_mean_5"), ("Win rate, last 5", "win_rate_5"), ("Custom strength rating", "elo")]
    for label, key in definitions:
        item = {"Measure": label}
        for side in ("home", "away"):
            if key == "count": value = str(d[side + "_prior_matches"])
            elif key == "last": value = d[side + "_last_recorded_match"]
            elif key == "days": value = str(d[side + "_days_since_recorded_match"])
            else:
                raw = features[side + "_" + key]
                value = "Unavailable" if raw is None else f"{raw:.1%}" if key == "win_rate_5" else f"{raw:.1f}" if key == "elo" else f"{raw:.2f}"
            item[r[side + "_team"]] = value
        records.append(item)
    return pd.DataFrame(records)


def render_forecast(result, key, revealed=False):
    request, prediction = result["request"], result["prediction"]
    home, away = request["home_team"], request["away_team"]
    st.subheader(home + " vs " + away)
    venue = "Neutral venue" if request["neutral"] else home + " at home"
    st.caption(request["date"] + ". " + venue + ". " + request["tournament"] + ".")
    for column, outcome, label in zip(st.columns(3), ("home_win", "draw", "away_win"), (home + " win", "Draw", away + " win")):
        column.metric(label, f"{prediction['outcome_probabilities'][outcome]:.2%}")
    st.altair_chart(probability_chart(result), width="stretch", theme=None)
    st.caption("Model-estimated outcome probabilities. Scores may include extra time; penalty shoot-out goals are excluded.")
    goals, leading_score = st.columns([1.05, 1])
    with goals:
        st.markdown("#### Expected goals")
        st.markdown(f"**{home} {prediction['expected_home_goals']:.2f} — {away} {prediction['expected_away_goals']:.2f}**")
    with leading_score:
        best = prediction["top_scorelines"][0]
        st.markdown("#### Most likely single scoreline")
        st.markdown(f"**{best['home_goals']}–{best['away_goals']}** with **{best['probability']:.2%}** estimated probability")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown("#### Score probabilities")
        st.altair_chart(heatmap(result), width="stretch", theme=None)
        st.caption("Grid shows 0–6 goals per team. Higher scores remain possible.")
    with right:
        st.markdown("#### Five likely scorelines")
        scores = [{"Score": f"{row['home_goals']}–{row['away_goals']}", "Probability": f"{row['probability']:.2%}"} for row in prediction["top_scorelines"]]
        st.dataframe(pd.DataFrame(scores), hide_index=True, width="stretch")
        with st.expander("More goal probabilities"):
            extra = prediction["additional_probabilities"]
            st.metric("At least 3 total goals", f"{extra['total_goals_at_least_3']:.2%}")
            st.metric("Both teams score", f"{extra['both_teams_score']:.2%}")
            st.caption("Derived from the same goal distribution.")
    with st.expander("Data behind this forecast", expanded=False):
        st.dataframe(evidence_table(result), hide_index=True, width="stretch")
        st.caption("Recent form uses up to five earlier appearances. The custom rating updates from earlier results.")
        st.write("Model fitted through **" + result["model"]["trained_through"] + "**. History included through **" + str(result["data"]["latest_included_history_date"]) + "**.")
        st.caption(result["model"]["calibration"])
    for side in ("home", "away"):
        if result["data"][side + "_days_since_recorded_match"] > 365:
            st.info("The dataset contains no match for " + request[side + "_team"] + " in the 365 days before this forecast date.")
    exported = json.loads(json.dumps(result))
    if request["mode"] == "recorded_match_replay" and not revealed:
        exported["actual_recorded_result"] = None
    st.download_button("Download forecast JSON", json.dumps(exported, ensure_ascii=False, indent=2), file_name="forecast_" + request["date"] + ".json", mime="application/json", key=key + "_download")


def scenario_page(predictor, signature):
    st.sidebar.markdown("### Match setup")
    first_date = date.fromisoformat(predictor.plan["test_start"])
    day = st.sidebar.date_input("Match date", value=max(date.today() + timedelta(days=1), first_date), min_value=first_date, key="scenario_date")
    options = [row["team"] for row in eligible_teams(str(ROOT), signature, day.isoformat())]
    home = valid_select("Home team", options, "scenario_home", "Nigeria")
    away = valid_select("Away team", [name for name in options if name != home], "scenario_away", "Brazil")
    neutral = st.sidebar.checkbox("Neutral venue", value=True, key="scenario_neutral")
    competitions = sorted({row["tournament"] for row in predictor.rows})
    competition = valid_select("Competition", competitions, "scenario_competition", "Friendly", help="The model currently distinguishes friendlies from other competitions; competition names otherwise share that feature value.")
    run = st.sidebar.button("Generate forecast", type="primary", key="scenario_run", width="stretch")
    st.sidebar.caption("Team choices require at least " + str(predictor.minimum_history) + " recorded appearances before the chosen date. Historical team names are included.")
    st.markdown("### Match scenario")
    st.caption("Explore a hypothetical fixture using the selected date and venue.")
    request_key = token(signature, home, away, day, neutral, competition)
    first_run = not st.session_state.get("scenario_started", False)
    if run or first_run:
        with st.spinner("Computing match probabilities..."):
            try:
                result = forecast(str(ROOT), signature, home, away, day.isoformat(), neutral, competition, False)
                st.session_state["scenario_result"] = result
                st.session_state["scenario_request_key"] = request_key
                st.session_state["scenario_started"] = True
            except (ValueError, OSError, KeyError) as exc:
                st.error(str(exc))
                return
    if st.session_state.get("scenario_request_key") != request_key:
        st.info("Match settings changed. Select Generate forecast to update the result.")
        return
    render_forecast(st.session_state["scenario_result"], "scenario")


def replay_page(predictor, signature):
    frame = replay_index(str(ROOT), signature)
    st.sidebar.markdown("### Replay setup")
    teams = ["All teams"] + sorted(set(frame["home_team"]) | set(frame["away_team"]))
    team = valid_select("Team", teams, "replay_team", "Nigeria")
    if team != "All teams":
        frame = frame.loc[frame["home_team"].eq(team) | frame["away_team"].eq(team)]
    years = sorted({day.year for day in frame["date"]}, reverse=True)
    year = valid_select("Year", years, "replay_year")
    frame = frame.loc[frame["date"].map(lambda day: day.year == year)].set_index("match_id", drop=False)
    ids = frame.index.tolist()
    def fixture_label(match_id):
        row = frame.loc[match_id]
        return row["date"].isoformat() + "  " + row["home_team"] + " vs " + row["away_team"]
    match_id = valid_select("Recorded fixture", ids, "replay_fixture", format_func=fixture_label)
    run = st.sidebar.button("Replay forecast", type="primary", key="replay_run", width="stretch")
    selected = frame.loc[match_id]
    st.markdown("### Historical replay")
    st.caption("Reconstruct a held-out forecast from earlier results, then reveal the recorded score.")
    request_key = token(signature, match_id)
    first_run = not st.session_state.get("replay_started", False)
    if run or first_run:
        with st.spinner("Reconstructing the pre-match history..."):
            try:
                result = forecast(str(ROOT), signature, selected["home_team"], selected["away_team"], selected["date"].isoformat(), False, None, True)
                actual_probs = list(result["prediction"]["outcome_probabilities"].values())
                saved_probs = [float(selected["prob_" + name]) for name in ("home_win", "draw", "away_win")]
                if not np.allclose(actual_probs, saved_probs, atol=1e-8, rtol=1e-8):
                    raise ValueError("This reconstruction differs from the frozen benchmark. Check the recorded history snapshot before using benchmark replay.")
                st.session_state["replay_result"] = result
                st.session_state["replay_request_key"] = request_key
                st.session_state["replay_revealed"] = False
                st.session_state["replay_started"] = True
            except (ValueError, OSError, KeyError) as exc:
                st.error(str(exc))
                return
    if st.session_state.get("replay_request_key") != request_key:
        st.info("Fixture selection changed. Select Replay forecast to reconstruct it.")
        return
    if st.button("Reveal recorded result", key="replay_reveal", type="primary"):
        st.session_state["replay_revealed"] = True
    result = st.session_state["replay_result"]
    revealed = st.session_state.get("replay_revealed", False)
    if revealed:
        actual, request = result["actual_recorded_result"], result["request"]
        st.info("Recorded result: " + request["home_team"] + " " + str(actual["home_goals"]) + "–" + str(actual["away_goals"]) + " " + request["away_team"])
        assigned = result["prediction"]["outcome_probabilities"][actual["outcome"]]
        st.caption(f"The model assigned {assigned:.2%} probability to the recorded outcome.")
    render_forecast(result, "replay", revealed=revealed)


def performance_page(signature):
    report = benchmark_report(str(ROOT), signature)
    selected = next(row for row in report["metrics"] if row["model"] == report["selected_on_validation"])
    prior = next(row for row in report["metrics"] if row["model"] == "DevelopmentPrior")
    st.markdown("### Model performance")
    st.caption("Saved held-out evaluation. This view reads the recorded report without refitting or rescoring.")
    a, b, c = st.columns(3)
    a.metric("Held-out accuracy", f"{selected['accuracy']:.2%}")
    b.metric("Reference accuracy", f"{prior['accuracy']:.2%}")
    c.metric("Accuracy advantage", f"{100 * (selected['accuracy'] - prior['accuracy']):.2f} pp")
    st.write(f"**{report['test_rows']:,} test matches** from **{report['test_first_date']}** to **{report['test_last_date']}**. The model was fitted through **{report['development_last_date']}**.")
    labels = {"DevelopmentPrior": "Outcome prior", "DevelopmentConstantGoalsPoisson": "Constant goals + Poisson", "PoissonGradientBoosting": "Poisson gradient boosting", "RandomForest": "Random Forest", "RegularizedPoisson": "Regularized Poisson"}
    table = pd.DataFrame([{"Model": labels.get(row["model"], row["model"]), "Accuracy": f"{row['accuracy']:.2%}", "Log loss": f"{row['log_loss']:.4f}", "Brier score": f"{row['brier_score']:.4f}", "Goal MAE": "Not applicable" if row["goal_mae"] is None else f"{row['goal_mae']:.4f}"} for row in report["metrics"]])
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Lower log loss and Brier score indicate better probability forecasts. Goal MAE averages absolute goal error across both teams.")
    years = pd.DataFrame([{"Year": str(year), "Matches": row["matches"], "Accuracy": row["accuracy"]} for year, row in report["selected_model_by_year"].items()])
    st.markdown("#### Accuracy by year")
    chart = alt.Chart(years).mark_bar(color=AWAY, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(x=alt.X("Year:O", title=None), y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")), tooltip=["Year:O", "Matches:Q", alt.Tooltip("Accuracy:Q", format=".2%")]).properties(height=245)
    st.altair_chart(chart_style(chart), width="stretch", theme=None)
    years["Accuracy"] = years["Accuracy"].map(lambda value: f"{value:.2%}")
    st.dataframe(years, hide_index=True, width="stretch")
    st.caption("The final year ends at the displayed test cutoff. Matches with insufficient prior history are excluded from scoring.")


def main():
    try:
        signature = file_signature()
        with st.spinner("Loading the frozen model and recorded history..."):
            predictor, _ = resources(str(ROOT), signature)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        st.error("The model or data setup is incomplete, or its saved versions do not match.")
        with st.expander("Setup details", expanded=True):
            st.code(str(exc))
        st.stop()
    st.markdown('<p class="eyebrow">FOOTBALL PREDICTION V2</p>', unsafe_allow_html=True)
    st.title("International Match Lab")
    st.write("Men’s international football forecasts and historical match replay.")
    st.caption("Dataset through " + predictor.dataset_latest_date + ". Model fitted through " + predictor.plan["development_last_date"] + ".")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("## Match Lab")
    page = st.sidebar.radio("Workspace", ["Match scenario", "Historical replay", "Model performance"], key="workspace")
    if page == "Match scenario": scenario_page(predictor, signature)
    elif page == "Historical replay": replay_page(predictor, signature)
    else: performance_page(signature)
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    with st.expander("Methods and data"):
        st.write("Forecasts use two goal models and an independent Poisson distribution. Team history uses strictly earlier calendar dates. Model settings were selected on validation data before the held-out test.")
        st.write("The target is the recorded final score, including extra time and excluding penalty shoot-outs. These are model-derived, uncalibrated probabilities.")
        st.markdown("Data: [Mart Jürisoo’s international football results](https://github.com/martj42/international_results), released under CC0-1.0. The modelling window begins in 1990. Historical and non-FIFA team names may appear.")
        st.caption("The benchmark belongs to the saved frozen model. Historical replay is limited to its held-out period.")


if __name__ == "__main__":
    main()
