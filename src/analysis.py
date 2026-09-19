from data_cleaning import prepare_data


DATA_PATH = "data/sample_flight_data.csv"

MIN_ROUTE_FLIGHTS = 2
MIN_AIRPORT_FLIGHTS = 5
MIN_AIRLINE_FLIGHTS = 10


def calculate_overall_kpis(df):
    """
    Calculate overall flight operation KPIs.
    """

    total_flights = len(df)

    delayed_flights = df["IS_DELAYED"].sum()

    severe_delays = df["IS_SEVERELY_DELAYED"].sum()

    delay_rate = (
        delayed_flights / total_flights * 100
        if total_flights
        else 0
    )

    severe_delay_rate = (
        severe_delays / total_flights * 100
        if total_flights
        else 0
    )

    return {
        "TOTAL_FLIGHTS": total_flights,
        "DELAYED_FLIGHTS": delayed_flights,
        "SEVERE_DELAYS": severe_delays,
        "DELAY_RATE": delay_rate,
        "SEVERE_DELAY_RATE": severe_delay_rate,
        "AVG_ARRIVAL_DELAY": df["ARR_DELAY"].mean(),
        "MEDIAN_ARRIVAL_DELAY": df["ARR_DELAY"].median(),
        "MAX_ARRIVAL_DELAY": df["ARR_DELAY"].max()
    }


def analyze_airlines(df):
    """
    Calculate airline-level performance metrics.
    """

    airline_performance = (
        df.groupby("AIRLINE")
        .agg(
            TOTAL_FLIGHTS=("FL_NUMBER", "count"),
            DELAYED_FLIGHTS=("IS_DELAYED", "sum"),
            SEVERE_DELAYS=("IS_SEVERELY_DELAYED", "sum"),
            AVG_ARRIVAL_DELAY=("ARR_DELAY", "mean"),
            MEDIAN_ARRIVAL_DELAY=("ARR_DELAY", "median"),
            MAX_DELAY=("ARR_DELAY", "max")
        )
    )

    airline_performance["DELAY_RATE"] = (
        airline_performance["DELAYED_FLIGHTS"]
        / airline_performance["TOTAL_FLIGHTS"]
        * 100
    )

    airline_performance["SEVERE_DELAY_RATE"] = (
        airline_performance["SEVERE_DELAYS"]
        / airline_performance["TOTAL_FLIGHTS"]
        * 100
    )

    airline_performance = airline_performance[
        airline_performance["TOTAL_FLIGHTS"]
        >= MIN_AIRLINE_FLIGHTS
    ]

    return airline_performance.sort_values(
        "DELAY_RATE",
        ascending=False
    )


def analyze_routes(df):
    """
    Analyze route-level delay performance.
    """

    route_performance = (
        df.groupby("ROUTE")
        .agg(
            TOTAL_FLIGHTS=("FL_NUMBER", "count"),
            DELAYED_FLIGHTS=("IS_DELAYED", "sum"),
            AVG_DELAY=("ARR_DELAY", "mean"),
            MAX_DELAY=("ARR_DELAY", "max")
        )
    )

    route_performance["DELAY_RATE"] = (
        route_performance["DELAYED_FLIGHTS"]
        / route_performance["TOTAL_FLIGHTS"]
        * 100
    )

    route_performance = route_performance[
        route_performance["TOTAL_FLIGHTS"]
        >= MIN_ROUTE_FLIGHTS
    ]

    return route_performance.sort_values(
        ["DELAY_RATE", "AVG_DELAY"],
        ascending=[False, False]
    )


def analyze_delay_causes(df):
    """
    Calculate delay minutes and percentage contribution by cause.
    """

    delay_causes = {
        "Carrier": df["DELAY_DUE_CARRIER"].sum(),
        "Weather": df["DELAY_DUE_WEATHER"].sum(),
        "NAS": df["DELAY_DUE_NAS"].sum(),
        "Security": df["DELAY_DUE_SECURITY"].sum(),
        "Late Aircraft": df["DELAY_DUE_LATE_AIRCRAFT"].sum()
    }

    total_delay_minutes = sum(delay_causes.values())

    results = {}

    for cause, minutes in delay_causes.items():

        percentage = (
            minutes / total_delay_minutes * 100
            if total_delay_minutes
            else 0
        )

        results[cause] = {
            "minutes": minutes,
            "percentage": percentage
        }

    return results


def min_max_normalize(series):
    """
    Normalize a pandas Series between 0 and 100.
    """

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return series * 0

    return (
        (series - minimum)
        / (maximum - minimum)
        * 100
    )


def assign_risk_level(score):
    """
    Convert numerical risk score into category.
    """

    if score >= 75:
        return "CRITICAL"

    if score >= 50:
        return "HIGH"

    if score >= 25:
        return "MEDIUM"

    return "LOW"


def analyze_airports(df):
    """
    Calculate normalized airport operational risk.
    """

    airport_performance = (
        df.groupby("ORIGIN")
        .agg(
            TOTAL_FLIGHTS=("FL_NUMBER", "count"),
            DELAYED_FLIGHTS=("IS_DELAYED", "sum"),
            SEVERE_DELAYS=("IS_SEVERELY_DELAYED", "sum"),
            AVG_DEP_DELAY=("DEP_DELAY", "mean"),
            MEDIAN_DEP_DELAY=("DEP_DELAY", "median"),
            MAX_DEP_DELAY=("DEP_DELAY", "max")
        )
    )

    airport_performance["DELAY_RATE"] = (
        airport_performance["DELAYED_FLIGHTS"]
        / airport_performance["TOTAL_FLIGHTS"]
        * 100
    )

    airport_performance["SEVERE_DELAY_RATE"] = (
        airport_performance["SEVERE_DELAYS"]
        / airport_performance["TOTAL_FLIGHTS"]
        * 100
    )

    airport_performance = airport_performance[
        airport_performance["TOTAL_FLIGHTS"]
        >= MIN_AIRPORT_FLIGHTS
    ].copy()

    # Normalize different metrics
    airport_performance["DELAY_RATE_NORM"] = (
        min_max_normalize(
            airport_performance["DELAY_RATE"]
        )
    )

    airport_performance["SEVERE_DELAY_RATE_NORM"] = (
        min_max_normalize(
            airport_performance["SEVERE_DELAY_RATE"]
        )
    )

    airport_performance["AVG_DEP_DELAY_NORM"] = (
        min_max_normalize(
            airport_performance[
                "AVG_DEP_DELAY"
            ].clip(lower=0)
        )
    )

    # Weighted operational risk
    airport_performance["RISK_SCORE"] = (
        airport_performance["DELAY_RATE_NORM"]
        * 0.50
        +
        airport_performance[
            "SEVERE_DELAY_RATE_NORM"
        ]
        * 0.30
        +
        airport_performance[
            "AVG_DEP_DELAY_NORM"
        ]
        * 0.20
    )

    airport_performance["RISK_LEVEL"] = (
        airport_performance["RISK_SCORE"]
        .apply(assign_risk_level)
    )

    return airport_performance.sort_values(
        "RISK_SCORE",
        ascending=False
    )


def print_overall_summary(kpis):

    print("\nFLIGHT OPERATIONS SUMMARY")

    print(
        f"Total Flights: "
        f"{kpis['TOTAL_FLIGHTS']}"
    )

    print(
        f"Delayed Flights: "
        f"{kpis['DELAYED_FLIGHTS']}"
    )

    print(
        f"Delay Rate: "
        f"{kpis['DELAY_RATE']:.2f}%"
    )

    print(
        f"Severely Delayed Flights: "
        f"{kpis['SEVERE_DELAYS']}"
    )

    print(
        f"Severe Delay Rate: "
        f"{kpis['SEVERE_DELAY_RATE']:.2f}%"
    )

    print(
        f"Average Arrival Delay: "
        f"{kpis['AVG_ARRIVAL_DELAY']:.2f} minutes"
    )

    print(
        f"Median Arrival Delay: "
        f"{kpis['MEDIAN_ARRIVAL_DELAY']:.2f} minutes"
    )

    print(
        f"Maximum Arrival Delay: "
        f"{kpis['MAX_ARRIVAL_DELAY']:.0f} minutes"
    )


def main():

    df = prepare_data(DATA_PATH)

    # Overall metrics
    kpis = calculate_overall_kpis(df)

    print_overall_summary(kpis)

    # Airlines
    airline_performance = analyze_airlines(df)

    print("\nAIRLINE PERFORMANCE")

    print(
        airline_performance
        .head(10)
        .round(2)
    )

    # Routes
    route_performance = analyze_routes(df)

    print("\nTOP DELAYED ROUTES")

    print(
        route_performance
        .head(10)
        .round(2)
    )

    # Delay causes
    delay_causes = analyze_delay_causes(df)

    print("\nDELAY CAUSE ANALYSIS")

    for cause, values in delay_causes.items():

        print(
            f"{cause}: "
            f"{values['minutes']} minutes "
            f"({values['percentage']:.2f}%)"
        )

    # Airports
    airport_performance = analyze_airports(df)

    print("\nAIRPORT OPERATIONAL RISK")

    columns = [
        "TOTAL_FLIGHTS",
        "DELAY_RATE",
        "SEVERE_DELAY_RATE",
        "AVG_DEP_DELAY",
        "RISK_SCORE",
        "RISK_LEVEL"
    ]

    print(
        airport_performance[
            columns
        ]
        .head(10)
        .round(2)
    )


if __name__ == "__main__":
    main()