import os
import re
import sqlite3
from datetime import datetime, timezone


import numpy as np
import pandas as pd

import requests
import streamlit as st


# ============================================================
# APPLICATION SETTINGS
# ============================================================

st.set_page_config(
    page_title="EUR/USD Research Engine",
    page_icon="💶",
    layout="wide",
)

APP_NAME = "EUR/USD Research Engine"
DATABASE = "eurusd_research.db"

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "EUR/USD": "DEXUSEU",
    "US 10Y Yield": "DGS10",
    "US 2Y Yield": "DGS2",
    "US CPI": "CPIAUCSL",
    "US Unemployment": "UNRATE",
    "Euro Area CPI": "CP0000EZ19M086NEST",
    "WTI Oil": "DCOILWTICO",
    "Gold": "GOLDAMGBD228NLBM",
}

NEWS_FEEDS = {
    "Federal Reserve": (
        "https://www.federalreserve.gov/feeds/press_all.xml"
    ),
    "ECB": (
        "https://www.ecb.europa.eu/rss/press.html"
    ),
    "Market News": (
        "https://feeds.finance.yahoo.com/rss/2.0/"
        "headline?s=EURUSD=X&region=US&lang=en-US"
    ),
}


# ============================================================
# SECRETS AND DATABASE
# ============================================================

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def database_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = database_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            decision TEXT NOT NULL,
            score REAL NOT NULL,
            confidence REAL NOT NULL,
            data_quality REAL NOT NULL,
            eurusd REAL,
            reasons TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            link TEXT,
            published TEXT
        )
        """
    )

    connection.commit()
    connection.close()


def save_decision(result, eurusd):
    connection = database_connection()

    connection.execute(
        """
        INSERT INTO decisions
        (
            created_at,
            decision,
            score,
            confidence,
            data_quality,
            eurusd,
            reasons
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            result["decision"],
            result["score"],
            result["confidence"],
            result["data_quality"],
            eurusd,
            " | ".join(result["reasons"]),
        ),
    )

    connection.commit()
    connection.close()


def get_decision_history(limit=100):
    connection = database_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM decisions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    connection.close()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(row) for row in rows])


# ============================================================
# DATA RETRIEVAL
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_fred_series(series_id, limit=500):
    api_key = get_secret("FRED_API_KEY")

    if not api_key:
        return pd.DataFrame(), "FRED_API_KEY is missing"

    parameters = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
        "limit": limit,
    }

    try:
        response = requests.get(
            FRED_URL,
            params=parameters,
            timeout=20,
        )
        response.raise_for_status()

        observations = response.json().get("observations", [])
        rows = []

        for observation in observations:
            try:
                rows.append(
                    {
                        "date": pd.to_datetime(
                            observation["date"]
                        ),
                        "value": float(observation["value"]),
                    }
                )
            except (ValueError, TypeError):
                continue

        if not rows:
            return pd.DataFrame(), "No valid data returned"

        return pd.DataFrame(rows).set_index("date"), None

    except Exception as error:
        return pd.DataFrame(), str(error)


@st.cache_data(ttl=900, show_spinner=False)
def get_ecb_eurusd(limit=500):
    url = (
        "https://data-api.ecb.europa.eu/service/data/"
        "EXR/D.USD.EUR.SP00.A"
        f"?format=jsondata&lastNObservations={limit}"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        payload = response.json()
        dataset = payload["dataSets"][0]
        series = next(iter(dataset["series"].values()))

        observations = series["observations"]
        values = [float(item[0]) for item in observations.values()]

        dates = payload["structure"]["dimensions"]["observation"][0][
            "values"
        ]

        dates = [
            pd.to_datetime(item["id"])
            for item in dates[-len(values):]
        ]

        frame = pd.DataFrame(
            {
                "date": dates,
                "value": values,
            }
        ).set_index("date")

        return frame, None

    except Exception as error:
        return pd.DataFrame(), str(error)


@st.cache_data(ttl=900, show_spinner=False)
def get_news():
    rows = []

    for source, url in NEWS_FEEDS.items():
        try:
            feed = feedparser.parse(url)

            for entry in feed.entries[:10]:
                rows.append(
                    {
                        "source": source,
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get(
                            "published",
                            entry.get("updated", ""),
                        ),
                    }
                )
        except Exception:
            continue

    return pd.DataFrame(rows)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(frame, name):
    if frame.empty:
        return frame

    data = frame.copy()
    data.columns = [name]

    data[f"{name}_return_1"] = data[name].pct_change(1)
    data[f"{name}_return_5"] = data[name].pct_change(5)
    data[f"{name}_return_20"] = data[name].pct_change(20)

    data[f"{name}_ma_20"] = (
        data[name].rolling(20, min_periods=5).mean()
    )

    data[f"{name}_volatility"] = (
        data[f"{name}_return_1"]
        .rolling(20, min_periods=5)
        .std()
        * np.sqrt(252)
    )

    return data


def latest_return(frame, column, periods=5):
    if frame.empty or column not in frame:
        return np.nan

    values = frame[column].dropna()

    if len(values) <= periods:
        return np.nan

    return values.pct_change(periods).iloc[-1]


def latest_value(frame, column):
    if frame.empty or column not in frame:
        return np.nan

    values = frame[column].dropna()

    if values.empty:
        return np.nan

    return values.iloc[-1]


def bounded_signal(value, multiplier=100, maximum=3):
    if pd.isna(value):
        return np.nan

    return float(np.tanh(value * multiplier) * maximum)


# ============================================================
# NEWS AND EVENT RISK
# ============================================================

def detect_event_risk(news):
    if news.empty:
        return 0, "No news feed data available"

    keywords = [
        "interest rate",
        "federal reserve",
        "fed",
        "ecb",
        "inflation",
        "cpi",
        "employment",
        "jobs",
        "nonfarm",
        "unemployment",
        "central bank",
        "monetary policy",
        "rate decision",
    ]

    text = " ".join(news["title"].astype(str)).lower()
    matches = sum(text.count(keyword) for keyword in keywords)

    if matches >= 5:
        return 80, "High-impact monetary or economic headlines detected"

    if matches >= 2:
        return 50, "Relevant economic headlines detected"

    return 20, "No major event concentration detected"


def headline_sentiment(news):
    if news.empty:
        return 0

    positive = [
        "growth",
        "strong",
        "improves",
        "recovery",
        "higher",
        "optimism",
        "support",
    ]

    negative = [
        "fall",
        "weak",
        "recession",
        "crisis",
        "lower",
        "risk",
        "uncertainty",
    ]

    score = 0

    for title in news["title"].astype(str).str.lower():
        score += sum(word in title for word in positive)
        score -= sum(word in title for word in negative)

    return max(-3, min(3, score / 3))


# ============================================================
# RESEARCH ENGINE
# ============================================================

def calculate_research(data, news):
    factors = []
    reasons = []

    def add_factor(name, signal, weight, explanation):
        if not pd.isna(signal):
            factors.append(
                {
                    "Factor": name,
                    "Signal": signal,
                    "Weight": weight,
                    "Contribution": signal * weight,
                }
            )
            reasons.append(explanation)

    eurusd = data.get("EUR/USD", pd.DataFrame())
    us10 = data.get("US 10Y Yield", pd.DataFrame())
    us2 = data.get("US 2Y Yield", pd.DataFrame())
    us_cpi = data.get("US CPI", pd.DataFrame())
    eu_cpi = data.get("Euro Area CPI", pd.DataFrame())
    oil = data.get("WTI Oil", pd.DataFrame())
    gold = data.get("Gold", pd.DataFrame())

    eurusd_change = latest_return(eurusd, "EUR/USD", 5)

    add_factor(
        "EUR/USD momentum",
        bounded_signal(eurusd_change, 100, 3),
        25,
        "Five-period EUR/USD movement was analyzed.",
    )

    us10_change = latest_return(us10, "US 10Y Yield", 5)

    add_factor(
        "U.S. 10Y yield",
        -bounded_signal(us10_change, 100, 3),
        15,
        "U.S. 10-year yield movement was analyzed.",
    )

    us2_change = latest_return(us2, "US 2Y Yield", 5)

    add_factor(
        "U.S. 2Y yield",
        -bounded_signal(us2_change, 100, 3),
        15,
        "U.S. 2-year yield movement was analyzed.",
    )

    us_inflation = latest_return(us_cpi, "US CPI", 3)
    eu_inflation = latest_return(eu_cpi, "Euro Area CPI", 3)

    if not pd.isna(us_inflation) and not pd.isna(eu_inflation):
        inflation_difference = eu_inflation - us_inflation

        add_factor(
            "Inflation differential",
            bounded_signal(inflation_difference, 100, 2),
            15,
            "Relative U.S. and euro-area inflation was analyzed.",
        )

    oil_change = latest_return(oil, "WTI Oil", 5)

    add_factor(
        "Oil pressure",
        -bounded_signal(oil_change, 100, 1),
        5,
        "Oil-price movement was analyzed.",
    )

    gold_change = latest_return(gold, "Gold", 5)

    add_factor(
        "Risk sentiment",
        -bounded_signal(gold_change, 100, 1),
        5,
        "Gold movement was analyzed as a risk indicator.",
    )

    news_score = headline_sentiment(news)

    add_factor(
        "News sentiment",
        news_score,
        5,
        "Recent headlines were analyzed for broad sentiment.",
    )

    event_risk, event_message = detect_event_risk(news)

    if not factors:
        return {
            "decision": "WAIT",
            "score": 0,
            "confidence": 0,
            "data_quality": 0,
            "event_risk": event_risk,
            "event_message": event_message,
            "reasons": ["No usable data was available."],
            "factors": pd.DataFrame(),
        }

    factor_frame = pd.DataFrame(factors)

    score = (
        factor_frame["Contribution"].sum()
        / factor_frame["Weight"].sum()
        * 100
    )

    data_quality = (
        factor_frame["Weight"].sum()
        / 90
        * 100
    )

    agreement = (
        abs(factor_frame["Signal"].mean())
        / 3
        * 100
    )

    confidence = (
        abs(score) * 0.55
        + data_quality * 0.25
        + agreement * 0.20
    )

    if event_risk >= 75:
        confidence *= 0.75
        reasons.append("High event risk reduced confidence.")

    confidence = max(0, min(95, confidence))

    if data_quality < 50 or confidence < 55:
        decision = "WAIT"
        reasons.append("Data coverage or confidence is too low.")
    elif score >= 60:
        decision = "BUY EUR/USD"
    elif score <= -60:
        decision = "SELL EUR/USD"
    else:
        decision = "WAIT"
        reasons.append("Signals are mixed.")

    return {
        "decision": decision,
        "score": score,
        "confidence": confidence,
        "data_quality": data_quality,
        "event_risk": event_risk,
        "event_message": event_message,
        "reasons": reasons,
        "factors": factor_frame,
    }


# ============================================================
# INITIALIZATION AND REFRESH
# ============================================================

initialize_database()

st.sidebar.header("Research controls")

if st.sidebar.button("🔄 Refresh research now"):
    st.cache_data.clear()
    st.rerun()

period = st.sidebar.selectbox(
    "Chart period",
    ["3 months", "6 months", "1 year"],
    index=1,
)

limits = {
    "3 months": 100,
    "6 months": 200,
    "1 year": 500,
}

limit = limits[period]

data = {}
errors = {}

for label, series_id in FRED_SERIES.items():
    frame, error = get_fred_series(series_id, limit)

    if error:
        data[label] = pd.DataFrame()
        errors[label] = error
    else:
        data[label] = add_features(frame, label)

if data["EUR/USD"].empty:
    ecb_frame, ecb_error = get_ecb_eurusd(limit)

    if not ecb_frame.empty:
        data["EUR/USD"] = add_features(
            ecb_frame,
            "EUR/USD",
        )
    else:
        errors["ECB EUR/USD"] = ecb_error

news = get_news()
research = calculate_research(data, news)

eurusd_price = latest_value(
    data.get("EUR/USD", pd.DataFrame()),
    "EUR/USD",
)

save_decision(research, eurusd_price)


# ============================================================
# DASHBOARD
# ============================================================

st.title(APP_NAME)
st.caption(
    "Automatic research dashboard — information only, "
    "no broker connection and no trade execution."
)

st.subheader("Final decision")

first, second, third, fourth = st.columns(4)

with first:
    st.metric("Decision", research["decision"])

with second:
    st.metric("Confidence", f"{research['confidence']:.0f}%")

with third:
    st.metric("Score", f"{research['score']:.1f}")

with fourth:
    if pd.isna(eurusd_price):
        st.metric("EUR/USD", "Unavailable")
    else:
        st.metric("EUR/USD", f"{eurusd_price:.5f}")

if research["decision"] == "BUY EUR/USD":
    st.success("Research currently favors euro strength.")

elif research["decision"] == "SELL EUR/USD":
    st.error("Research currently favors dollar strength.")

else:
    st.warning("Research is mixed or incomplete. No directional decision.")

st.subheader("Why this decision was produced")

for reason in research["reasons"]:
    st.write(f"- {reason}")

st.write(
    f"**Event risk:** {research['event_risk']}/100 — "
    f"{research['event_message']}"
)


# ============================================================
# EUR/USD GRAPH
# ============================================================

st.subheader("EUR/USD price graph")

eurusd = data.get("EUR/USD", pd.DataFrame())

if not eurusd.empty:
    chart = eurusd.reset_index()

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart["date"],
            y=chart["EUR/USD"],
            mode="lines",
            name="EUR/USD",
        )
    )

    if "EUR/USD_ma_20" in chart:
        figure.add_trace(
            go.Scatter(
                x=chart["date"],
                y=chart["EUR/USD_ma_20"],
                mode="lines",
                name="20-period average",
            )
        )

    figure.update_layout(
        height=450,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="USD per EUR",
    )

    st.plotly_chart(figure, use_container_width=True)

else:
    st.warning("EUR/USD graph data is unavailable.")


# ============================================================
# FACTOR GRAPH
# ============================================================

st.subheader("Research factor contributions")

factors = research["factors"]

if not factors.empty:
    figure = go.Figure(
        go.Bar(
            x=factors["Contribution"],
            y=factors["Factor"],
            orientation="h",
            marker_color=[
                "#16a34a" if value > 0 else "#dc2626"
                for value in factors["Contribution"]
            ],
        )
    )

    figure.update_layout(
        height=450,
        xaxis_title="Contribution toward EUR/USD",
        yaxis_title="Factor",
    )

    st.plotly_chart(figure, use_container_width=True)

    st.dataframe(
        factors.style.format(
            {
                "Signal": "{:.2f}",
                "Weight": "{:.1f}",
                "Contribution": "{:.2f}",
            }
        ),
        use_container_width=True,
    )


# ============================================================
# NEWS
# ============================================================

st.subheader("Recent research headlines")

if news.empty:
    st.info("No news feed data was available.")
else:
    for _, item in news.head(15).iterrows():
        title = str(item["title"])
        source = str(item["source"])
        link = str(item["link"])

        if link:
            st.markdown(
                f"- **{source}:** [{title}]({link})"
            )
        else:
            st.write(f"- **{source}:** {title}")


# ============================================================
# DATA QUALITY AND HISTORY
# ============================================================

st.subheader("Data quality")

available = len(data) - len(errors)
total = len(data)
quality = available / total * 100 if total else 0

st.progress(int(quality))
st.write(f"Available datasets: {available}/{total}")
st.write(f"Data coverage: {quality:.0f}%")

if errors:
    with st.expander("Unavailable datasets"):
        for name, error in errors.items():
            st.write(f"**{name}:** {error}")

st.subheader("Decision history")

history = get_decision_history(50)

if history.empty:
    st.info("No decision history available yet.")
else:
    history["created_at"] = pd.to_datetime(
        history["created_at"]
    )

    st.dataframe(
        history[
            [
                "created_at",
                "decision",
                "score",
                "confidence",
                "data_quality",
                "eurusd",
            ]
        ],
        use_container_width=True,
    )

st.caption(
    "Last checked: "
    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
)

st.caption(
    "This system is experimental. It does not guarantee direction "
    "and does not place trades."
)