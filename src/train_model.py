from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_cleaning import prepare_data


# =========================================================
# PATHS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "sample_flight_data.csv"

MODEL_DIR = ROOT_DIR / "models"

MODEL_PATH = MODEL_DIR / "delay_model.joblib"

MODEL_INFO_PATH = MODEL_DIR / "model_info.joblib"


# =========================================================
# FEATURES
# =========================================================

CATEGORICAL_FEATURES = [
    "AIRLINE",
    "ORIGIN",
    "DEST",
    "MONTH_NAME",
    "DAY_OF_WEEK",
    "TIME_OF_DAY"
]

NUMERIC_FEATURES = [
    "DEP_HOUR",
    "DEP_MINUTE",
    "DISTANCE",
    "IS_WEEKEND"
]

MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


# =========================================================
# PREPROCESSING
# =========================================================

def build_preprocessor():

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES
            ),
            (
                "numeric",
                numeric_transformer,
                NUMERIC_FEATURES
            )
        ]
    )

    return preprocessor


# =========================================================
# MODEL DEFINITIONS
# =========================================================

def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            C=0.5,
            random_state=42
        ),
        

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
    }


# =========================================================
# BUILD PIPELINE
# =========================================================

def build_pipeline(model):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                model
            )
        ]
    )


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_model(
    model_name,
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test
):

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    if hasattr(
        pipeline,
        "predict_proba"
    ):

        probabilities = (
            pipeline
            .predict_proba(
                X_test
            )[:, 1]
        )

    else:

        probabilities = None


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    if probabilities is not None:

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

    else:

        roc_auc = 0


    print("\n" + "=" * 60)

    print(
        f"MODEL: {model_name}"
    )

    print("=" * 60)

    print(
        f"Accuracy:  {accuracy:.3f}"
    )

    print(
        f"Precision: {precision:.3f}"
    )

    print(
        f"Recall:    {recall:.3f}"
    )

    print(
        f"F1 Score:  {f1:.3f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.3f}"
    )


    print(
        "\nCLASSIFICATION REPORT"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


    print(
        "CONFUSION MATRIX"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


    return {
        "name": model_name,
        "pipeline": pipeline,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# =========================================================
# BEST MODEL SELECTION
# =========================================================

def calculate_model_score(result):
    """
    Custom scoring function.

    We care more about detecting delayed flights
    than raw overall accuracy.

    Weighting:
    ROC-AUC = 40%
    Recall  = 30%
    F1      = 20%
    Precision = 10%
    """

    return (
        result["roc_auc"] * 0.40
        + result["recall"] * 0.30
        + result["f1"] * 0.20
        + result["precision"] * 0.10
    )


# =========================================================
# TRAIN ALL MODELS
# =========================================================

def train_models():

    print("\nLoading and preparing flight data...")

    df = prepare_data(
        DATA_PATH
    )


    X = df[
        MODEL_FEATURES
    ]

    y = df[
        "IS_DELAYED"
    ]


    print(
        f"\nTotal Samples: {len(df)}"
    )

    print(
        f"Delayed Flights: {y.sum()}"
    )

    print(
        f"Not Delayed: {(y == 0).sum()}"
    )


    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )


    print(
        f"\nTraining Samples: {len(X_train)}"
    )

    print(
        f"Testing Samples: {len(X_test)}"
    )


    models = get_models()

    results = []


    for model_name, model in models.items():

        pipeline = build_pipeline(
            model
        )

        result = evaluate_model(
            model_name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test
        )

        result["model_score"] = (
            calculate_model_score(
                result
            )
        )

        results.append(
            result
        )


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    comparison_data = []

    for result in results:

        comparison_data.append(
            {
                "MODEL":
                    result["name"],

                "ACCURACY":
                    result["accuracy"],

                "PRECISION":
                    result["precision"],

                "RECALL":
                    result["recall"],

                "F1":
                    result["f1"],

                "ROC_AUC":
                    result["roc_auc"],

                "MODEL_SCORE":
                    result["model_score"]
            }
        )


    comparison_df = pd.DataFrame(
        comparison_data
    )


    comparison_df = (
        comparison_df
        .sort_values(
            "MODEL_SCORE",
            ascending=False
        )
    )


    print(
        "\n\nMODEL COMPARISON"
    )

    print(
        comparison_df
        .round(3)
        .to_string(
            index=False
        )
    )


    # =====================================================
    # BEST MODEL
    # =====================================================

    best_result = max(
        results,
        key=lambda result:
        result["model_score"]
    )


    best_model = (
        best_result["pipeline"]
    )


    print(
        "\nBEST MODEL"
    )

    print(
        f"Model: {best_result['name']}"
    )

    print(
        f"ROC-AUC: "
        f"{best_result['roc_auc']:.3f}"
    )

    print(
        f"Recall: "
        f"{best_result['recall']:.3f}"
    )

    print(
        f"F1 Score: "
        f"{best_result['f1']:.3f}"
    )

    print(
        f"Combined Model Score: "
        f"{best_result['model_score']:.3f}"
    )


    # =====================================================
    # SAVE MODEL
    # =====================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    joblib.dump(
        best_model,
        MODEL_PATH
    )


    model_info = {
        "model_name":
            best_result["name"],

        "accuracy":
            best_result["accuracy"],

        "precision":
            best_result["precision"],

        "recall":
            best_result["recall"],

        "f1":
            best_result["f1"],

        "roc_auc":
            best_result["roc_auc"],

        "model_score":
            best_result["model_score"],

        "features":
            MODEL_FEATURES
    }


    joblib.dump(
        model_info,
        MODEL_INFO_PATH
    )


    print(
        f"\nBest model saved to:\n{MODEL_PATH}"
    )

    print(
        f"\nModel information saved to:\n"
        f"{MODEL_INFO_PATH}"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    train_models()