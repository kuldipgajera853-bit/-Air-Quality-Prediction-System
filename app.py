import pickle
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .dashboard-card {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .dashboard-subtle {
        color: #cbd5e1;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = pickle.load(open("models/model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    return model, scaler


model, scaler = load_artifacts()
feature_names = ["PM2.5", "PM10", "CO", "O3", "NH3"]


def classify_risk(aqi_value):
    if aqi_value <= 50:
        return "Good", "#16a34a"
    if aqi_value <= 100:
        return "Moderate", "#ca8a04"
    if aqi_value <= 200:
        return "Poor", "#ea580c"
    if aqi_value <= 300:
        return "Very Poor", "#dc2626"
    return "Severe", "#7f1d1d"


def predict_aqi_value(pm25, pm10, co, o3, nh3):
    values = np.array([[pm25, pm10, co, o3, nh3]], dtype=float)
    scaled_values = scaler.transform(values)
    return float(model.predict(scaled_values)[0])


def build_aqi_forecast(feature_values, hourly_change_percent):
    """Create short-term AQI scenarios from the current pollutant readings."""
    hourly_multiplier = 1 + (hourly_change_percent / 100)
    forecast_rows = []

    for hours_ahead in range(1, 5):
        future_values = np.asarray(feature_values, dtype=float) * (hourly_multiplier ** hours_ahead)
        future_aqi = predict_aqi_value(*future_values)
        risk_label, _ = classify_risk(future_aqi)
        forecast_rows.append(
            {
                "Time": (datetime.now() + timedelta(hours=hours_ahead)).strftime("%I:%M %p"),
                "Hours ahead": hours_ahead,
                "Forecast AQI": round(future_aqi, 2),
                "Risk": risk_label,
            }
        )

    return pd.DataFrame(forecast_rows)


st.title("Air Quality Prediction Dashboard")
st.markdown("Enter values manually on your desktop, then see the AQI output on the same dashboard page.")

st.subheader("Manual Input and AQI Output")
left_panel, right_panel = st.columns([1, 1])

with left_panel:
    st.markdown("### Manual Input")
    st.write("Type the readings yourself instead of using sensors.")

    with st.form("manual_input_form"):
        input_top_left, input_top_right = st.columns(2)

        with input_top_left:
            pm25 = st.number_input("PM2.5", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
            co = st.number_input("CO", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.2f")
            nh3 = st.number_input("NH3", min_value=0.0, max_value=200.0, value=20.0, step=1.0)

        with input_top_right:
            pm10 = st.number_input("PM10", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
            o3 = st.number_input("O3", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
            st.write("")

        hourly_change_percent = st.number_input(
            "Expected hourly change in pollutant readings (%)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=0.5,
            help="Use 0% when you expect conditions to stay stable. For example, 2% increases each reading by 2% every hour.",
        )

        submitted = st.form_submit_button("Run AQI Prediction")

    st.markdown(
        """
        <div class="dashboard-card">
            <div class="dashboard-subtle">Input mode</div>
            <h2 style="margin:0.35rem 0 0 0;">Manual desktop entry</h2>
            <div class="dashboard-subtle">All values are typed directly from your keyboard.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_panel:
    st.markdown("### AQI Output")

    if submitted:
        prediction = predict_aqi_value(pm25, pm10, co, o3, nh3)
        risk_label, risk_color = classify_risk(prediction)

        output_col1, output_col2, output_col3 = st.columns(3)
        with output_col1:
            st.metric("Predicted AQI", f"{prediction:.2f}")
        with output_col2:
            st.metric("Health Risk", risk_label)
        with output_col3:
            st.metric("Inputs", len(feature_names))

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="dashboard-subtle">Current risk level</div>
                <h1 style="margin:0.35rem 0 0 0; color:{risk_color};">{risk_label}</h1>
                <div class="dashboard-subtle">Predicted AQI: {prediction:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        feature_values = [pm25, pm10, co, o3, nh3]
        summary_df = pd.DataFrame({"Feature": feature_names, "Value": feature_values})
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("### Future AQI Forecast")
        forecast_df = build_aqi_forecast(feature_values, hourly_change_percent)
        three_hour_aqi = forecast_df.loc[forecast_df["Hours ahead"] == 3, "Forecast AQI"].iloc[0]
        four_hour_aqi = forecast_df.loc[forecast_df["Hours ahead"] == 4, "Forecast AQI"].iloc[0]

        forecast_col1, forecast_col2 = st.columns(2)
        with forecast_col1:
            st.metric("AQI in 3 hours", f"{three_hour_aqi:.2f}")
        with forecast_col2:
            st.metric("AQI in 4 hours", f"{four_hour_aqi:.2f}")

        st.line_chart(forecast_df, x="Time", y="Forecast AQI", use_container_width=True)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        st.caption(
            "This is a short-term scenario forecast based on the current readings and the expected hourly change entered above. "
            "Use 0% for stable conditions."
        )
    else:
        st.info("Fill the manual inputs and click Run AQI Prediction to show the AQI output here.")

st.subheader("Feature Dashboard")
feature_values = [pm25, pm10, co, o3, nh3]
feature_col1, feature_col2 = st.columns([1, 1])

with feature_col1:
    st.metric("Features", len(feature_names))
    st.metric("PM2.5", f"{pm25:.1f}")
    st.metric("PM10", f"{pm10:.1f}")

with feature_col2:
    st.metric("CO", f"{co:.2f}")
    st.metric("O3", f"{o3:.1f}")
    st.metric("NH3", f"{nh3:.1f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(feature_names, feature_values, color=["#0ea5e9", "#38bdf8", "#22c55e", "#f59e0b", "#a855f7"])
ax.set_ylabel("Value")
ax.set_title("Manually Entered Feature Values")
st.pyplot(fig)


st.markdown("---")
st.markdown("Built using Machine Learning + manual desktop input.")
