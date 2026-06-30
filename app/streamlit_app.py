"""Streamlit app for MLB batter breakout probability prediction.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logreg.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "breakout_features.parquet"

FEATURES = [
    "Age", "WAR", "PA", "BA", "OBP", "SLG", "OPS",
    "bb_rate", "k_rate", "hr_rate", "iso", "babip",
    "war_trend", "ops_trend",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_parquet(FEATURES_PATH)
    # Only rows with enough history to score reliably
    return df[df["has_prior_year"]].copy()


def add_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
            max-width: 95%;
        }
        h1 { font-size: 2.6rem !important; }
        div[data-testid="stMetricValue"] { font-size: 2.0rem; }

        .prediction-card {
            border: 1px solid rgba(250,250,250,0.15);
            border-radius: 18px;
            padding: 28px;
            background: rgba(255,255,255,0.035);
            text-align: center;
            margin-bottom: 16px;
        }
        .big-prob {
            font-size: 4.0rem;
            font-weight: 800;
            line-height: 1;
        }
        .subtle {
            color: rgba(250,250,250,0.65);
            font-size: 1.05rem;
        }
        .good { color: #21c55d; font-weight: 700; }
        .bad { color: #ef4444; font-weight: 700; }
        .info-card {
            border: 1px solid rgba(250,250,250,0.12);
            border-radius: 14px;
            padding: 18px;
            background: rgba(255,255,255,0.025);
            height: 100%;
            font-size: 1.0rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_explanation(row: pd.Series, probability: float, league_avg: float) -> list[tuple[str, str]]:
    explanations = []

    if row["war_trend"] < -1.0:
        explanations.append((
            "Down year vs. last season",
            f"WAR dropped {abs(row['war_trend']):.1f} from the prior year — "
            "history suggests some reversion back up is likely.",
        ))
    elif row["war_trend"] > 1.0:
        explanations.append((
            "Already trending up",
            f"WAR rose {row['war_trend']:.1f} from the prior year — "
            "momentum already building before this prediction.",
        ))

    if row["Age"] <= 25:
        explanations.append((
            "Young player",
            "Players at this age still typically have development runway left.",
        ))
    elif row["Age"] >= 33:
        explanations.append((
            "Older player",
            "Breakouts become rarer as players age past their physical peak.",
        ))

    if row["k_rate"] < 0.18:
        explanations.append((
            "Strong plate discipline",
            "Low strikeout rate is a stable underlying skill linked to sustained performance.",
        ))

    if row["babip"] < 0.280:
        explanations.append((
            "Low BABIP this season",
            "Below-average luck on balls in play — some bounce-back regression is plausible.",
        ))
    elif row["babip"] > 0.330:
        explanations.append((
            "High BABIP this season",
            "Above-average luck on balls in play — some of this season's results may not repeat.",
        ))

    if probability > league_avg:
        explanations.append((
            "Model vs. league baseline",
            "Predicted probability is above the league-wide breakout base rate.",
        ))
    else:
        explanations.append((
            "Model vs. league baseline",
            "Predicted probability is at or below the league-wide breakout base rate.",
        ))

    return explanations[:5]


def draw_trend_chart(name: str, war_prev: float, war: float):
    fig, ax = plt.subplots(figsize=(5, 4))
    years = ["Last season", "This season"]
    values = [war_prev, war]
    colors = ["#94a3b8", "#3b82f6"]
    ax.bar(years, values, color=colors, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("WAR", fontsize=11)
    ax.set_title(f"{name} — WAR trend", fontsize=13, weight="bold", pad=20)

    # Push the y-axis ceiling up so labels never collide with the title,
    # and place value labels with consistent spacing above/below bars.
    max_val = max(values + [0])
    min_val = min(values + [0])
    headroom = (max_val - min_val) * 0.15 if max_val != min_val else 1.0
    ax.set_ylim(min_val - headroom, max_val + headroom * 1.8)

    for i, v in enumerate(values):
        offset = headroom * 0.3 if v >= 0 else -headroom * 0.5
        ax.text(i, v + offset, f"{v:.1f}", ha="center", fontsize=11, weight="bold")

    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()
    return fig

def main():
    st.set_page_config(
        page_title="MLB Breakout Predictor",
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    add_css()

    st.title("⚾ MLB Breakout Predictor")
    st.caption(
        "Predicts a batter's probability of a 2.0+ WAR jump next season, "
        "using age, performance level, plate discipline, and year-over-year trend."
    )

    model = load_model()
    df = load_data()
    league_avg = df["breakout"].mean() if "breakout" in df.columns else 0.13

    players_years = (
        df[["name_common", "year_ID"]]
        .drop_duplicates()
        .sort_values(["name_common", "year_ID"])
    )
    names = sorted(players_years["name_common"].unique())

    with st.sidebar:
        st.header("Player & season")
        default_idx = names.index("Aaron Judge") if "Aaron Judge" in names else 0
        player = st.selectbox("Player", names, index=default_idx)

        player_years = sorted(
            players_years[players_years["name_common"] == player]["year_ID"].tolist()
        )
        year = st.selectbox(
            "Season to score from",
            player_years,
            index=len(player_years) - 1,
        )
        st.caption("Prediction is for the season immediately following the one selected.")

    row = df[(df["name_common"] == player) & (df["year_ID"] == year)].iloc[0]

    if row[FEATURES].isna().any():
        st.warning(
            f"{player} ({year}) is missing required prior-year history and "
            "cannot be scored reliably."
        )
        return

    X = pd.DataFrame([row[FEATURES]])
    probability = float(model.predict_proba(X)[0, 1])
    diff = probability - league_avg

    top_left, top_right = st.columns([1, 1])

    with top_left:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="subtle">Predicted breakout probability for {year + 1}</div>
                <div class="big-prob">{probability * 100:.1f}%</div>
                <div class="subtle">League base rate: {league_avg * 100:.1f}%</div>
                <div class="{'good' if diff >= 0 else 'bad'}">
                    {diff * 100:+.1f} points vs. league base rate
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Age", f"{row['Age']:.0f}")
        c2.metric("WAR", f"{row['WAR']:.1f}")
        c3.metric("OPS", f"{row['OPS']:.3f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("BB%", f"{row['bb_rate'] * 100:.1f}%")
        c5.metric("K%", f"{row['k_rate'] * 100:.1f}%")
        c6.metric("BABIP", f"{row['babip']:.3f}")

        if row.get("has_next_year") and not pd.isna(row.get("war_jump")):
            st.divider()
            actual = "Yes" if row["breakout"] == 1 else "No"
            st.write(
                f"**Actual outcome:** breakout = **{actual}** "
                f"(WAR went from {row['WAR']:.1f} to {row['WAR_next']:.1f}, "
                f"a {row['war_jump']:+.1f} change)"
            )

    with top_right:
        fig = draw_trend_chart(player, row["WAR_prev"], row["WAR"])
        st.pyplot(fig, width="stretch")

    st.divider()

    bottom_left, bottom_right = st.columns([1, 1])

    with bottom_left:
        st.subheader("Season stats")
        summary = pd.DataFrame({
            "Stat": ["Age", "PA", "AVG", "OBP", "SLG", "OPS", "HR rate", "BB rate", "K rate", "ISO", "BABIP", "WAR trend"],
            "Value": [
                f"{row['Age']:.0f}",
                f"{row['PA']:.0f}",
                f"{row['BA']:.3f}",
                f"{row['OBP']:.3f}",
                f"{row['SLG']:.3f}",
                f"{row['OPS']:.3f}",
                f"{row['hr_rate'] * 100:.1f}%",
                f"{row['bb_rate'] * 100:.1f}%",
                f"{row['k_rate'] * 100:.1f}%",
                f"{row['iso']:.3f}",
                f"{row['babip']:.3f}",
                f"{row['war_trend']:+.2f}",
            ],
        })
        st.dataframe(summary, width="stretch", hide_index=True)

    with bottom_right:
        st.subheader("Why this prediction?")
        for title, detail in build_explanation(row, probability, league_avg):
            st.markdown(
                f"""
                <div class="info-card">
                    <b>{title}</b><br>
                    <span class="subtle">{detail}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Model notes"):
        st.write(
            """
            This prediction uses a logistic regression model trained on 2015-2024
            MLB batter seasons (minimum 300 plate appearances). The breakout label
            is defined as a 2.0+ WAR increase in the following season, conditional
            on the player remaining a qualified batter (300+ PA) that next season.

            Logistic regression outperformed a tuned XGBoost model on this dataset,
            suggesting the underlying relationship between these features and
            breakout likelihood is close to linear rather than driven by complex
            feature interactions.
            """
        )


if __name__ == "__main__":
    main()
