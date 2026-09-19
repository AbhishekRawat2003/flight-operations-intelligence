import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PROJECT PATH SETUP
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.append(str(SRC_DIR))

from data_cleaning import prepare_data
from analysis import (
    calculate_overall_kpis,
    analyze_airlines,
    analyze_routes,
    analyze_delay_causes,
    analyze_airports
)

DATA_PATH = ROOT_DIR / "data" / "sample_flight_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "delay_model.joblib"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Flight Operations Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .insight-box {
        padding: 16px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATA / MODEL LOADING
# =========================================================

@st.cache_data
def load_dashboard_data():
    return prepare_data(DATA_PATH)


@st.cache_resource
def load_delay_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return None


df = load_dashboard_data()
delay_model = load_delay_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("✈️ Flight Intelligence")

    st.caption(
        "Filter flight operations data and explore "
        "performance, delays and operational risk."
    )

    st.divider()

    airline_options = ["All"] + sorted(
        df["AIRLINE"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_airline = st.selectbox(
        "Airline",
        airline_options
    )

    airport_options = ["All"] + sorted(
        df["ORIGIN"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_airport = st.selectbox(
        "Origin Airport",
        airport_options
    )

    year_options = ["All"] + sorted(
        df["YEAR"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    selected_year = st.selectbox(
        "Year",
        year_options
    )

    month_options = ["All"] + [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    selected_month = st.selectbox(
        "Month",
        month_options
    )

    st.divider()

    st.caption(
        "Delay threshold: ≥ 15 minutes"
    )

    st.caption(
        "Severe delay: ≥ 60 minutes"
    )


# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df.copy()

if selected_airline != "All":
    filtered_df = filtered_df[
        filtered_df["AIRLINE"] == selected_airline
    ]

if selected_airport != "All":
    filtered_df = filtered_df[
        filtered_df["ORIGIN"] == selected_airport
    ]

if selected_year != "All":
    filtered_df = filtered_df[
        filtered_df["YEAR"] == selected_year
    ]

if selected_month != "All":
    filtered_df = filtered_df[
        filtered_df["MONTH_NAME"] == selected_month
    ]


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    'Flight Operations Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Aviation analytics platform for flight delays,
    airline performance, route intelligence and
    airport operational risk.
    </div>
    """,
    unsafe_allow_html=True
)

if filtered_df.empty:
    st.warning(
        "No flight records match the selected filters."
    )

    st.stop()


# =========================================================
# KPI CALCULATIONS
# =========================================================

kpis = calculate_overall_kpis(filtered_df)


# =========================================================
# KPI CARDS
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Operational Overview'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Flights",
    f"{kpis['TOTAL_FLIGHTS']:,}",
    border=True
)

col2.metric(
    "Delayed Flights",
    f"{kpis['DELAYED_FLIGHTS']:,}",
    border=True
)

col3.metric(
    "Delay Rate",
    f"{kpis['DELAY_RATE']:.2f}%",
    border=True
)

col4.metric(
    "Severe Delay Rate",
    f"{kpis['SEVERE_DELAY_RATE']:.2f}%",
    border=True
)

col5, col6, col7 = st.columns(3)

col5.metric(
    "Average Arrival Delay",
    f"{kpis['AVG_ARRIVAL_DELAY']:.2f} min",
    border=True
)

col6.metric(
    "Median Arrival Delay",
    f"{kpis['MEDIAN_ARRIVAL_DELAY']:.2f} min",
    border=True
)

col7.metric(
    "Maximum Delay",
    f"{kpis['MAX_ARRIVAL_DELAY']:.0f} min",
    border=True
)

st.divider()


# =========================================================
# ANALYSIS DATA
# =========================================================

airline_performance = analyze_airlines(filtered_df)

route_performance = analyze_routes(filtered_df)

delay_causes = analyze_delay_causes(filtered_df)

airport_performance = analyze_airports(filtered_df)


# =========================================================
# AUTOMATIC INSIGHTS
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Key Operational Insights'
    '</div>',
    unsafe_allow_html=True
)

insight1, insight2, insight3 = st.columns(3)


with insight1:

    if not airline_performance.empty:

        worst_airline = airline_performance.iloc[0]

        worst_airline_name = (
            airline_performance.index[0]
        )

        st.markdown(
            f"""
            <div class="insight-box">

            <b>Highest Airline Delay Rate</b>

            <br><br>

            {worst_airline_name}

            <br>

            {worst_airline['DELAY_RATE']:.2f}% delayed

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "Not enough airline data."
        )


with insight2:

    if delay_causes:

        dominant_cause = max(
            delay_causes,
            key=lambda x:
            delay_causes[x]["minutes"]
        )

        cause_data = delay_causes[
            dominant_cause
        ]

        st.markdown(
            f"""
            <div class="insight-box">

            <b>Primary Delay Driver</b>

            <br><br>

            {dominant_cause}

            <br>

            {cause_data['percentage']:.2f}% of
            recorded delay minutes

            </div>
            """,
            unsafe_allow_html=True
        )


with insight3:

    if not airport_performance.empty:

        highest_risk_airport = (
            airport_performance.index[0]
        )

        airport_risk = (
            airport_performance.iloc[0]
        )

        st.markdown(
            f"""
            <div class="insight-box">

            <b>Highest Operational Risk</b>

            <br><br>

            {highest_risk_airport}

            <br>

            Risk Score:
            {airport_risk['RISK_SCORE']:.2f}

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "Not enough airport data."
        )


st.divider()


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "✈️ Airlines",
        "🛫 Airports",
        "🗺 Routes",
        "⚠️ Delay Causes",
        "📈 Trends",
        "🤖 Delay Predictor"
    ]
)


# =========================================================
# AIRLINE TAB
# =========================================================

with tab1:

    st.subheader(
        "Airline Performance"
    )

    st.caption(
        "Comparison of airline reliability "
        "using delay rates and severe delays."
    )

    if not airline_performance.empty:

        airline_chart = (
            airline_performance[
                ["DELAY_RATE"]
            ]
            .sort_values(
                "DELAY_RATE",
                ascending=False
            )
        )

        st.bar_chart(
            airline_chart,
            height=450
        )

        st.subheader(
            "Airline Performance Table"
        )

        airline_columns = [
            "TOTAL_FLIGHTS",
            "DELAYED_FLIGHTS",
            "SEVERE_DELAYS",
            "DELAY_RATE",
            "SEVERE_DELAY_RATE",
            "AVG_ARRIVAL_DELAY",
            "MEDIAN_ARRIVAL_DELAY",
            "MAX_DELAY"
        ]

        st.dataframe(
            airline_performance[
                airline_columns
            ].round(2),
            use_container_width=True
        )

    else:

        st.info(
            "Not enough airline records "
            "for analysis."
        )


# =========================================================
# AIRPORT TAB
# =========================================================

with tab2:

    st.subheader(
        "Airport Operational Risk"
    )

    st.caption(
        "Relative operational risk based on "
        "delay frequency, severe delays and "
        "average departure delay."
    )

    if not airport_performance.empty:

        top_airports = (
            airport_performance
            .head(10)
        )

        st.bar_chart(
            top_airports[
                ["RISK_SCORE"]
            ],
            height=420
        )

        airport_columns = [
            "TOTAL_FLIGHTS",
            "DELAY_RATE",
            "SEVERE_DELAY_RATE",
            "AVG_DEP_DELAY",
            "RISK_SCORE",
            "RISK_LEVEL"
        ]

        st.dataframe(
            top_airports[
                airport_columns
            ].round(2),
            use_container_width=True
        )

        st.info(
            "Risk Score represents relative "
            "operational delay risk within this "
            "dataset. It is not an aviation "
            "safety rating."
        )

    else:

        st.info(
            "Not enough airport records "
            "for risk analysis."
        )


# =========================================================
# ROUTES TAB
# =========================================================

with tab3:

    st.subheader(
        "Route Intelligence"
    )

    st.caption(
        "Routes ranked according to delay "
        "frequency and average arrival delay."
    )

    if not route_performance.empty:

        top_routes = (
            route_performance
            .head(10)
        )

        st.bar_chart(
            top_routes[
                ["DELAY_RATE"]
            ],
            height=420
        )

        st.dataframe(
            top_routes.round(2),
            use_container_width=True
        )

        st.warning(
            "The current sample dataset contains "
            "only 1,000 flights. Routes with "
            "small sample sizes should not be "
            "treated as statistically definitive."
        )

    else:

        st.info(
            "No routes meet the minimum "
            "flight threshold."
        )


# =========================================================
# DELAY CAUSES TAB
# =========================================================

with tab4:

    st.subheader(
        "Delay Cause Analysis"
    )

    delay_cause_df = pd.DataFrame(
        [
            {
                "Cause": cause,
                "Delay Minutes": values["minutes"],
                "Contribution (%)": values["percentage"]
            }

            for cause, values
            in delay_causes.items()
        ]
    )

    delay_cause_df = (
        delay_cause_df
        .sort_values(
            "Delay Minutes",
            ascending=False
        )
    )

    chart_col, table_col = st.columns(
        [2, 1]
    )

    with chart_col:

        st.bar_chart(
            delay_cause_df
            .set_index("Cause")[
                "Delay Minutes"
            ],
            height=400
        )

    with table_col:

        st.dataframe(
            delay_cause_df.round(2),
            hide_index=True,
            use_container_width=True
        )


# =========================================================
# TRENDS TAB
# =========================================================

with tab5:

    st.subheader(
        "Monthly Delay Trend"
    )

    monthly_performance = (
        filtered_df
        .groupby("MONTH_NAME")
        .agg(
            TOTAL_FLIGHTS=(
                "FL_NUMBER",
                "count"
            ),
            DELAYED_FLIGHTS=(
                "IS_DELAYED",
                "sum"
            )
        )
    )

    monthly_performance[
        "DELAY_RATE"
    ] = (
        monthly_performance[
            "DELAYED_FLIGHTS"
        ]
        /
        monthly_performance[
            "TOTAL_FLIGHTS"
        ]
        * 100
    )

    MONTH_ORDER = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    monthly_performance = (
        monthly_performance
        .reindex(MONTH_ORDER)
        .dropna()
    )

    st.line_chart(
        monthly_performance[
            ["DELAY_RATE"]
        ],
        height=420
    )

    with st.expander(
        "View Monthly Data"
    ):

        st.dataframe(
            monthly_performance.round(2),
            use_container_width=True
        )

# =========================================================
# ML DELAY PREDICTOR TAB
# =========================================================

with tab6:

    st.subheader(
        "Flight Delay Predictor"
    )

    st.caption(
        "Estimate the probability of a flight "
        "arriving at least 15 minutes late."
    )

    if delay_model is None:

        st.warning(
            "ML model not found. Run "
            "`python src/train_model.py` first."
        )

    else:

        left, right = st.columns(2)

        with left:

            prediction_airline = st.selectbox(
                "Airline",
                sorted(
                    df["AIRLINE"]
                    .dropna()
                    .unique()
                    .tolist()
                ),
                key="prediction_airline"
            )

            prediction_origin = st.selectbox(
                "Origin Airport",
                sorted(
                    df["ORIGIN"]
                    .dropna()
                    .unique()
                    .tolist()
                ),
                key="prediction_origin"
            )

            prediction_destination = st.selectbox(
                "Destination Airport",
                sorted(
                    df["DEST"]
                    .dropna()
                    .unique()
                    .tolist()
                ),
                key="prediction_destination"
            )

            prediction_month = st.selectbox(
                "Month",
                [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December"
                ],
                key="prediction_month"
            )

        with right:

            prediction_day = st.selectbox(
                "Day of Week",
                [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday"
                ],
                key="prediction_day"
            )

            departure_time = st.number_input(
                "Scheduled Departure Time (HHMM)",
                min_value=0,
                max_value=2359,
                value=1200,
                step=5
            )

            distance = st.number_input(
                "Flight Distance (miles)",
                min_value=1,
                max_value=6000,
                value=800,
                step=10
            )

        predict_button = st.button(
            "Predict Delay Risk",
            type="primary"
        )

        if predict_button:

            # Convert HHMM into hour and minute
            dep_hour = departure_time // 100
            dep_minute = departure_time % 100

            if dep_hour >= 24:
                dep_hour = 0

            # Time-of-day feature
            if 5 <= dep_hour < 12:
                time_of_day = "Morning"

            elif 12 <= dep_hour < 17:
                time_of_day = "Afternoon"

            elif 17 <= dep_hour < 21:
                time_of_day = "Evening"

            else:
                time_of_day = "Night"

            # Weekend feature
            is_weekend = int(
                prediction_day in [
                    "Saturday",
                    "Sunday"
                ]
            )

            # Prediction input
            prediction_data = pd.DataFrame(
                [
                    {
                        "AIRLINE": prediction_airline,
                        "ORIGIN": prediction_origin,
                        "DEST": prediction_destination,
                        "MONTH_NAME": prediction_month,
                        "DAY_OF_WEEK": prediction_day,
                        "TIME_OF_DAY": time_of_day,
                        "DEP_HOUR": dep_hour,
                        "DEP_MINUTE": dep_minute,
                        "DISTANCE": distance,
                        "IS_WEEKEND": is_weekend
                    }
                ]
            )

            probability = (
                delay_model
                .predict_proba(
                    prediction_data
                )[0][1]
            )

            probability_percent = (
                probability * 100
            )

            if probability_percent >= 70:
                risk_level = "HIGH"

            elif probability_percent >= 40:
                risk_level = "MEDIUM"

            else:
                risk_level = "LOW"

            st.divider()

            result1, result2 = st.columns(2)

            result1.metric(
                "Predicted Delay Probability",
                f"{probability_percent:.1f}%"
            )

            result2.metric(
                "Predicted Risk Level",
                risk_level
            )

            st.progress(
                int(
                    min(
                        probability_percent,
                        100
                    )
                )
            )

            st.caption(
                "Prediction means estimated probability "
                "of arrival being delayed by at least "
                "15 minutes."
            )


with st.expander(
    "🔎 Explore Filtered Flight Records"
):

    st.write(
        f"Showing {len(filtered_df):,} flight records"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Flight Operations Analytics & "
    "Delay Risk Intelligence System | "
    "Python • Pandas • Streamlit • Scikit-learn"
)