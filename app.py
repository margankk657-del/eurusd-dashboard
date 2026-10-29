import os
from datetime import datetime, timezone

import requests
import streamlit as st


st.set_page_config(
    page_title="EUR/USD Decision Dashboard",
    page_icon="💶",
    layout="centered",
)

st.title("EUR/USD Decision Dashboard")
st.caption("Information only — this app does not place trades.")


def get_secret(name):
    try:
        return st.secrets.get(name, os.getenv(name, ""))
    except Exception:
        return os.getenv(name, "")


@st.cache_data(ttl=900)
def get_fred_data(series_id):
    api_key = get_secret("FRED_API_KEY")

    if not api_key:
        return [], "FRED_API_KEY is missing"

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 30,
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()

        observations = response.json().get("observations", [])
        values = [
            float(item["value"])
            for item in observations
            if item["value"] not in ("", ".")
        ]

        return values, None

    except Exception as error:
        return [], str(error)


@st.cache_data(ttl=900)
def get_ecb_data():
    url = (
        "https://data-api.ecb.europa.eu/service/data/"
        "EXR/D.USD.EUR.SP00.A"
        "?format=jsondata&lastNObservations=30"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        data = response.json()
        series = next(iter(data["dataSets"][0]["series"].values()))

        values = [
            float(item[0])
            for item in series["observations"].values()
        ]

        return values, None

    except Exception as error:
        return [], str(error)


def recent_direction(values):
    if len(values) < 5:
        return 0

    oldest = values[-1]
    newest = values[0]
    percentage_change = ((newest - oldest) / oldest) * 100

    if percentage_change > 0.15:
        return 1

    if percentage_change < -0.15:
        return -1

    return 0


eur_usd, eur_usd_error = get_fred_data("DEXUSEU")
us_yields, us_yields_error = get_fred_data("DGS10")
ecb_rates, ecb_error = get_ecb_data()

score = 0
reasons = []

if eur_usd:
    direction = recent_direction(eur_usd)

    if direction > 0:
        score += 2
        reasons.append("Recent EUR/USD movement favors the euro.")
    elif direction < 0:
        score -= 2
        reasons.append("Recent EUR/USD movement favors the dollar.")
    else:
        reasons.append("Recent EUR/USD movement is neutral.")
else:
    reasons.append("EUR/USD data is unavailable.")

if us_yields:
    direction = recent_direction(us_yields)

    if direction > 0:
        score -= 2
        reasons.append("Rising U.S. yields favor the dollar.")
    elif direction < 0:
        score += 2
        reasons.append("Falling U.S. yields reduce dollar support.")
    else:
        reasons.append("U.S. yield movement is neutral.")
else:
    reasons.append("U.S. yield data is unavailable.")

data_quality = 100

if not eur_usd:
    data_quality -= 40

if not us_yields:
    data_quality -= 30

if not ecb_rates:
    data_quality -= 20

if data_quality < 50:
    decision = "WAIT"
    explanation = "Important data is missing."
elif abs(score) < 2:
    decision = "WAIT"
    explanation = "The signals are too weak or mixed."
elif score > 0:
    decision = "BUY EUR/USD"
    explanation = "The combined signals favor euro strength."
else:
    decision = "SELL EUR/USD"
    explanation = "The combined signals favor dollar strength."

confidence = min(
    95,
    max(35, 50 + abs(score) * 10 + (data_quality - 70) * 0.2),
)

st.subheader("Final decision")

left, right = st.columns(2)

with left:
    st.metric("Decision", decision)

with right:
    st.metric("Confidence", f"{confidence:.0f}%")

st.info(explanation)

st.subheader("Main reason")

for reason in reasons:
    st.write(f"- {reason}")

st.warning(
    "This is an experimental information tool. "
    "It does not guarantee the market direction and does not place trades."
)

st.caption(
    "Last checked: "
    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
)

with st.expander("Data status"):
    st.write("FRED EUR/USD:", "OK" if eur_usd else eur_usd_error)
    st.write("FRED U.S. 10-year yield:", "OK" if us_yields else us_yields_error)
    st.write("ECB data:", "OK" if ecb_rates else ecb_error)
