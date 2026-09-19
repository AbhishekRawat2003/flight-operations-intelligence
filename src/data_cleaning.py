import pandas as pd


DELAY_COLUMNS = [
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT"
]


def load_data(file_path):
    """
    Load flight dataset from CSV.
    """

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def inspect_data(df):
    """
    Display important information about the dataset.
    """

    print("\nDATASET SUMMARY")

    print(f"Shape: {df.shape}")

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nCancelled Flights:")
    print(df["CANCELLED"].value_counts())

    print("\nDiverted Flights:")
    print(df["DIVERTED"].value_counts())

    print("\nArrival Delay Statistics:")
    print(df["ARR_DELAY"].describe())


def clean_data(df):
    """
    Clean dates, duplicates and delay columns.
    """

    df = df.copy()

    # Convert date column
    df["FL_DATE"] = pd.to_datetime(
        df["FL_DATE"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    invalid_dates = df["FL_DATE"].isna().sum()

    print(f"\nInvalid dates after conversion: {invalid_dates}")

    # Remove duplicates
    df = df.drop_duplicates()

    # Missing delay values mean no recorded delay
    df[DELAY_COLUMNS] = df[DELAY_COLUMNS].fillna(0)

    print("Cleaning completed.")

    return df


def create_features(df):

    df = df.copy()

    # Date features
    df["YEAR"] = df["FL_DATE"].dt.year
    df["MONTH"] = df["FL_DATE"].dt.month
    df["MONTH_NAME"] = df["FL_DATE"].dt.month_name()
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.day_name()

    # Weekend flag
    df["IS_WEEKEND"] = (
        df["DAY_OF_WEEK"]
        .isin(["Saturday", "Sunday"])
    ).astype(int)

    # Route
    df["ROUTE"] = (
        df["ORIGIN"]
        + " → "
        + df["DEST"]
    )

    # Scheduled departure time
    df["DEP_HOUR"] = (
        df["CRS_DEP_TIME"] // 100
    )

    df["DEP_MINUTE"] = (
        df["CRS_DEP_TIME"] % 100
    )

    # Fix special 2400 style values if present
    df.loc[
        df["DEP_HOUR"] >= 24,
        "DEP_HOUR"
    ] = 0

    # Time-of-day bucket
    def get_time_of_day(hour):

        if 5 <= hour < 12:
            return "Morning"

        elif 12 <= hour < 17:
            return "Afternoon"

        elif 17 <= hour < 21:
            return "Evening"

        else:
            return "Night"

    df["TIME_OF_DAY"] = (
        df["DEP_HOUR"]
        .apply(get_time_of_day)
    )

    # Delay labels
    df["IS_DELAYED"] = (
        df["ARR_DELAY"] >= 15
    ).astype(int)

    df["IS_SEVERELY_DELAYED"] = (
        df["ARR_DELAY"] >= 60
    ).astype(int)

    return df


def prepare_data(file_path):
    """
    Full data preparation pipeline.
    """

    df = load_data(file_path)

    df = clean_data(df)

    df = create_features(df)

    return df


if __name__ == "__main__":

    df = load_data("data/sample_flight_data.csv")

    inspect_data(df)

    df = clean_data(df)

    df = create_features(df)

    print("\nFeature Engineering Completed")

    print(
        df[
            [
                "FL_DATE",
                "AIRLINE",
                "ROUTE",
                "ARR_DELAY",
                "IS_DELAYED",
                "IS_SEVERELY_DELAYED"
            ]
        ].head(10)
    )