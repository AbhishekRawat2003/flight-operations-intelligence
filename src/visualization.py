import matplotlib.pyplot as plt

from data_cleaning import prepare_data
from analysis import (
    analyze_airlines,
    analyze_delay_causes,
    analyze_routes,
    analyze_airports
)


DATA_PATH = "data/sample_flight_data.csv"


def plot_airline_delay_rate(df):
    airline_performance = analyze_airlines(df)

    airline_performance = airline_performance.sort_values(
        "DELAY_RATE",
        ascending=True
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        airline_performance.index,
        airline_performance["DELAY_RATE"]
    )

    plt.xlabel("Delay Rate (%)")
    plt.ylabel("Airline")
    plt.title("Airline Delay Rate")

    plt.tight_layout()
    plt.show()


def plot_delay_causes(df):
    delay_causes = analyze_delay_causes(df)

    causes = list(delay_causes.keys())

    minutes = [
        values["minutes"]
        for values in delay_causes.values()
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        causes,
        minutes
    )

    plt.xlabel("Delay Cause")
    plt.ylabel("Total Delay Minutes")
    plt.title("Total Delay Minutes by Cause")

    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.show()


def plot_top_delayed_routes(df):
    route_performance = analyze_routes(df)

    top_routes = (
        route_performance
        .head(10)
        .sort_values(
            "DELAY_RATE",
            ascending=True
        )
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_routes.index,
        top_routes["DELAY_RATE"]
    )

    plt.xlabel("Delay Rate (%)")
    plt.ylabel("Route")
    plt.title("Top Delayed Routes")

    plt.tight_layout()
    plt.show()


def plot_airport_risk(df):
    airport_performance = analyze_airports(df)

    top_airports = (
        airport_performance
        .head(10)
        .sort_values(
            "RISK_SCORE",
            ascending=True
        )
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_airports.index,
        top_airports["RISK_SCORE"]
    )

    plt.xlabel("Operational Risk Score")
    plt.ylabel("Airport")
    plt.title("Top Airport Operational Risk")

    plt.tight_layout()
    plt.show()


def main():
    df = prepare_data(DATA_PATH)

    plot_airline_delay_rate(df)

    plot_delay_causes(df)

    plot_top_delayed_routes(df)

    plot_airport_risk(df)


if __name__ == "__main__":
    main()