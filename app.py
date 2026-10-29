import os
import sqlite3
from datetime import datetime, timezone

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="EUR/USD Research Dashboard",
    page_icon="💶",
    layout="wide",
)

st.title("EUR/USD Research Dashboard")
st.caption(
    "Information only — no automatic trading and no broker connection."
)


FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
DATABASE = "eurusd_history.db"

SERIES = {
    "EUR/USD": "DEXUSEU",
    "US 10Y": "DGS10",
    "US 2Y": "DGS2",
    "US CPI": "CPIAUCSL",
    "Euro CPI": "CP0000EZ19M086NEST",
    "Oil": "DCOILWTICO",
    "Gold": "GOLDAMGBD228NLBM",
}

FEEDS = {
    "Federal Reserve": (
        "https://www.federalreserve.gov/feeds/press_all.xml"
    ),
    "ECB": "https://www.ecb.europa.eu/rss/press.html",
    "Market News": (
        "https://feeds.finance.yahoo.com/rss/2.0/"
        "headline?s=EURUSD=X&region=US&lang=en-US"
    ),
}


def secret(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def database():
    return sqlite3.connect(DATABASE)


def setup_database():
    connection = database()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            decision TEXT,
            score REAL,
            confidence REAL,
            quality REAL,
            price REAL,
            reasons TEXT
        )
        """
    )

    connection.commit()
    connection.close()


def save_result(result, price):
    connection = database()

    connection.execute(
        """
        INSERT INTO decisions
        (created_at, decision, score, confidence, quality, price, reasons)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            result["decision"],
            result["score"],
            result["confidence"],
            result["quality"],
            price,
            " | ".join(result["reasons"]),
        ),
    )

    connection.commit()
    connection.close()


def history():
    connection = database()

    frame = pd.read_sql_query(
        """
        SELECT *
        FROM decisions
        ORDER BY id DESC
        LIMIT 50
        """,
        connection,
    )

    connection.close()
    return frame


def get_fred(series_id, limit=400):
    api_key = secret("FRED_API_KEY")

    if not api_key:
        return pd.DataFrame(), "Missing FRED_API_KEY"

    try:
        response = requests.get(
            FRED_URL,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "asc",
                "limit": limit,
            },
            timeout=20,
        )

        response.raise_for_status()

        rows = []

        for item in response.json().get("observations", []):
            try:
                rows.append(
                    {
                        "date": pd.to_datetime(item["date"]),
                        "value": float(item["value"]),
                    }
                )
            except (ValueError, TypeError):
                pass

        if not rows:
            return pd.DataFrame(), "No usable observations"

        return pd.DataFrame(rows).set_index("date"), None

    except Exception as error:
        return pd.DataFrame(), str(error)


def get_ecb():
    url = (
        "https://data-api.ecb.europa.eu/service/data/"
        "EXR/D.USD.EUR.SP00.A"
        "?format=jsondata&lastNObservations=400"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        payload = response.json()
        series = next(
            iter(payload["dataSets"][0]["series"].values())
        )

        values = [
            float(item[0])
            for item in series["observations"].values()
        ]

        dates = payload["structure"]["dimensions"]["observation"][0][
            "values"
        ]

        dates = [
            pd.to_datetime(item["id"])
            for item in dates[-len(values):]
        ]

        return pd.DataFrame(
            {"value": values},
            index=dates,
        ), None

    except Exception as error:
        return pd.DataFrame(), str(error)


def get_news():
    rows = []

    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)

            for item in feed.entries[:10]:
                rows.append(
                    {
                        "source": source,
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                    }
                )
        except Exception:
            pass

    return pd.DataFrame(rows)


def add_features(frame, name):
    if frame.empty:
        return frame

    data = frame.copy()
    data.columns = [name]

    data["return_5"] = data[name].pct_change(5)
    data["average_20"] = data[name].rolling(20).mean()
    data["volatility"] = data[name].pct_change().rolling(20).std()

    return data


def recent_return(frame, column, periods=5):
    if frame.empty or column not in frame:
        return np.nan

    values = frame[column].dropna()

    if len(values) <= periods:
        return np.nan

    return values.pct_change(periods).iloc[-1]


def signal(value, multiplier=100, maximum=3):
    if pd.isna(value):
        return np.nan

    return float(np.tanh(value * multiplier) * maximum)


def headline_score(news):
    if news.empty:
        return 0

    positive = [
        "growth",
        "strong",
        "improves",
        "recovery",
        "support",
    ]

    negative = [
        "weak",
        "fall",
        "recession",
        "crisis",
        "risk",
        "uncertainty",
    ]

    total = 0

    for title in news["title"].astype(str).str.lower():
        total += sum(word in title for word in positive)
        total -= sum(word in title for word in negative)

    return max(-3, min(3, total / 3))


def research(data, news):
    factors = []
    reasons = []

    def add(name, value, weight, explanation):
        if not pd.isna(value):
            factors.append(
                {
                    "Factor": name,
                    "Signal": value,
                    "Weight": weight,
                    "Contribution": value * weight,
                }
            )
            reasons.append(explanation)

    eurusd = data.get("EUR/USD", pd.DataFrame())
    us10 = data.get("US 10Y", pd.DataFrame())
    us2 = data.get("US 2Y", pd.DataFrame())
    us_cpi = data.get("US CPI", pd.DataFrame())
    euro_cpi = data.get("Euro CPI", pd.DataFrame())
    oil = data.get("Oil", pd.DataFrame())
    gold = data.get("Gold", pd.DataFrame())

    add(
        "EUR/USD movement",
        signal(recent_return(eurusd, "EUR/USD")),
        25,
        "Recent EUR/USD movement was analyzed.",
    )

    add(
        "U.S. 10-year yield",
        -signal(recent_return(us10, "US 10Y")),
        15,
        "U.S. 10-year yield movement was analyzed.",
    )

    add(
        "U.S. 2-year yield",
        -signal(recent_return(us2, "US 2Y")),
        15,
        "U.S. 2-year yield movement was analyzed.",
    )

    us_inflation = recent_return(us_cpi, "US CPI", 3)
    euro_inflation = recent_return(euro_cpi, "Euro CPI", 3)

    if not pd.isna(us_inflation) and not pd.isna(euro_inflation):
        add(
            "Inflation difference",
            signal(euro_inflation - us_inflation, maximum=2),
            15,
            "U.S. and euro-area inflation were compared.",
        )

    add(
        "Oil pressure",
        -signal(recent_return(oil, "Oil"), maximum=1),
        5,
        "Oil-price movement was analyzed.",
    )

    add(
        "Gold risk signal",
        -signal(recent_return(gold, "Gold"), maximum=1),
        5,
        "Gold movement was analyzed as a risk signal.",
    )

    add(
        "News sentiment",
        headline_score(news),
        5,
        "Recent headlines were analyzed.",
    )

    factor_frame = pd.DataFrame(factors)

    if factor_frame.empty:
        return {
            "decision": "WAIT",
            "score": 0,
            "confidence": 0,
            "quality": 0,
            "reasons": ["No usable research data was found."],
            "factors": factor_frame,
        }

    score = (
        factor_frame["Contribution"].sum()
        / factor_frame["Weight"].sum()
        * 100
    )

    quality = factor_frame["Weight"].sum() / 90 * 100
    agreement = abs(factor_frame["Signal"].mean()) / 3 * 100

    confidence = (
        abs(score) * 0.55
        + quality * 0.25
        + agreement * 0.20
    )

    confidence = max(0, min(95, confidence))

    if quality < 50 or confidence < 55:
        decision = "WAIT"
        reasons.append("Data quality or confidence is too low.")
    elif score >= 60:
        decision = "BUY EUR/USD"
    elif score <= -60:
        decision = "SELL EUR/USD"
    else:
        decision = "WAIT"
        reasons.append("The signals are mixed.")

    return {
        "decision": decision,
        "score": score,
        "confidence": confidence,
        "quality": quality,
        "reasons": reasons,
        "factors": factor_frame,
    }


# ============================================================
# RUN RESEARCH EVERY TIME THE APP RUNS
# ============================================================

setup_database()

if st.button("🔄 Research again now"):
    st.cache_data.clear()
    st.rerun()

data = {}
errors = {}

for name, series_id in SERIES.items():
    frame, error = get_fred(series_id)

    if error:
        data[name] = pd.DataFrame()
        errors[name] = error
    else:
        data[name] = add_features(frame, name)

if data["EUR/USD"].empty:
    ecb_frame, ecb_error = get_ecb()

    if not ecb_frame.empty:
        data["EUR/USD"] = add_features(ecb_frame, "EUR/USD")
    else:
        errors["ECB EUR/USD"] = ecb_error

news = get_news()
result = research(data, news)

eurusd = data["EUR/USD"]

if eurusd.empty:
    current_price = np.nan
else:
    current_price = eurusd["EUR/USD"].dropna().iloc[-1]

save_result(result, current_price)


# ============================================================
# DISPLAY RESULT
# ============================================================

first, second, third, fourth = st.columns(4)

with first:
    st.metric("Decision", result["decision"])

with second:
    st.metric("Confidence", f"{result['confidence']:.0f}%")

with third:
    st.metric("Score", f"{result['score']:.1f}")

with fourth:
    if pd.isna(current_price):
        st.metric("EUR/USD", "Unavailable")
    else:
        st.metric("EUR/USD", f"{current_price:.5f}")

for reason in result["reasons"]:
    st.write(f"- {reason}")


# ============================================================
# GRAPH
# ============================================================

st.subheader("EUR/USD graph")

if not eurusd.empty:
    chart_data = eurusd.reset_index()

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["EUR/USD"],
            mode="lines",
            name="EUR/USD",
        )
    )

    if "average_20" in chart_data:
        figure.add_trace(
            go.Scatter(
                x=chart_data["date"],
                y=chart_data["average_20"],
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


st.subheader("Factor contributions")

if not result["factors"].empty:
    factors = result["factors"]

    figure = go.Figure(
        go.Bar(
            x=factors["Contribution"],
            y=factors["Factor"],
            orientation="h",
            marker_color=[
                "green" if value > 0 else "red"
                for value in factors["Contribution"]
            ],
        )
    )

    figure.update_layout(height=450)

    st.plotly_chart(figure, use_container_width=True)


st.subheader("Recent headlines")

if news.empty:
    st.info("No headlines were found.")
else:
    for _, item in news.head(15).iterrows():
        st.markdown(
            f"- **{item['source']}:** "
            f"[{item['title']}]({item['link']})"
        )


st.subheader("Data quality")

available = len(data) - len(errors)
total = len(data)
quality = available / total * 100 if total else 0

st.progress(int(quality))
st.write(f"Available datasets: {available}/{total}")
st.write(f"Data coverage: {quality:.0f}%")

if errors:
    with st.expander("Unavailable data"):
        for name, error in errors.items():
            st.write(f"**{name}:** {error}")


st.subheader("Decision history")

history_data = history()

if history_data.empty:
    st.info("No stored decisions yet.")
else:
    st.dataframe(
        history_data[
            [
                "created_at",
                "decision",
                "score",
                "confidence",
                "quality",
                "price",
            ]
        ],
        use_container_width=True,
    )


st.caption(
    "Last research run: "
    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
)