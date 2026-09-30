import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)


DATA_FILE = "results/trade_forensics.csv"


# ==========================================
# FEATURES
# ==========================================

FEATURES = [
    "entry_rsi",
    "ema_distance_pct",
    "entry_volatility",
    "entry_return_1h"
]


def load_data():

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=[
            "entry_time",
            "exit_time"
        ]
    )

    df = (
        df
        .sort_values("entry_time")
        .reset_index(drop=True)
    )

    # Remove incomplete feature rows
    df = df.dropna(
        subset=FEATURES + ["winner"]
    )

    # Convert boolean target to 0/1
    df["target"] = (
        df["winner"]
        .astype(int)
    )

    return df


def split_data(df):
    """
    TIME-ORDERED split.

    First 70% = training
    Last 30% = validation

    We deliberately DO NOT randomly shuffle
    financial time-series observations.
    """

    split_index = int(
        len(df) * 0.70
    )

    train = (
        df.iloc[:split_index]
        .copy()
    )

    validation = (
        df.iloc[split_index:]
        .copy()
    )

    return train, validation


def build_model():

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    return model


def evaluate_model(
    model,
    train,
    validation
):

    X_train = train[FEATURES]
    y_train = train["target"]

    X_val = validation[FEATURES]
    y_val = validation["target"]

    # --------------------------------------
    # TRAIN
    # --------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------
    # PREDICTIONS
    # --------------------------------------

    probabilities = (
        model.predict_proba(X_val)[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # --------------------------------------
    # METRICS
    # --------------------------------------

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            y_val,
            probabilities
        )

    except ValueError:

        auc = np.nan

    cm = confusion_matrix(
        y_val,
        predictions
    )

    # --------------------------------------
    # ADD PROBABILITIES TO VALIDATION DATA
    # --------------------------------------

    validation = validation.copy()

    validation[
        "ml_probability"
    ] = probabilities

    validation[
        "ml_prediction"
    ] = predictions

    return (
        validation,
        accuracy,
        precision,
        recall,
        auc,
        cm
    )


def print_report(
    train,
    validation,
    accuracy,
    precision,
    recall,
    auc,
    cm
):

    print("\n" + "=" * 60)

    print(
        "       MACHINE LEARNING — TRADE FILTER"
    )

    print("=" * 60)

    print(
        f"Total trades:          "
        f"{len(train) + len(validation)}"
    )

    print(
        f"Training trades:       "
        f"{len(train)}"
    )

    print(
        f"Validation trades:     "
        f"{len(validation)}"
    )

    print("-" * 60)

    print(
        f"Training period:       "
        f"{train['entry_time'].min()} "
        f"→ "
        f"{train['entry_time'].max()}"
    )

    print(
        f"Validation period:     "
        f"{validation['entry_time'].min()} "
        f"→ "
        f"{validation['entry_time'].max()}"
    )

    print("-" * 60)

    print(
        f"Accuracy:              "
        f"{accuracy:.2%}"
    )

    print(
        f"Precision:             "
        f"{precision:.2%}"
    )

    print(
        f"Recall:                "
        f"{recall:.2%}"
    )

    print(
        f"ROC AUC:               "
        f"{auc:.3f}"
    )

    print("\nConfusion matrix:")

    print(cm)

    print("=" * 60)


def probability_analysis(validation):

    print(
        "\nML PROBABILITY ANALYSIS\n"
    )

    bins = [
        0,
        0.40,
        0.50,
        0.60,
        0.70,
        1.00
    ]

    labels = [
        "<40%",
        "40-50%",
        "50-60%",
        "60-70%",
        ">70%"
    ]

    validation = validation.copy()

    validation[
        "probability_bucket"
    ] = pd.cut(
        validation["ml_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    analysis = (
        validation
        .groupby(
            "probability_bucket",
            observed=True
        )
        .agg(
            trades=("target", "count"),
            actual_win_rate=(
                "target",
                "mean"
            ),
            avg_return=(
                "net_return",
                "mean"
            )
        )
    )

    print(analysis)


if __name__ == "__main__":

    data = load_data()

    train, validation = (
        split_data(data)
    )

    model = build_model()

    (
        validation,
        accuracy,
        precision,
        recall,
        auc,
        cm

    ) = evaluate_model(
        model,
        train,
        validation
    )

    print_report(
        train,
        validation,
        accuracy,
        precision,
        recall,
        auc,
        cm
    )

    probability_analysis(
        validation
    )

    validation.to_csv(
        "results/ml_validation.csv",
        index=False
    )