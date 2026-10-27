import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss
)


DATA_FILE = "data/BTC_ML_features.csv"

FEATURES = [
    "return_1h",
    "return_6h",
    "return_12h",
    "return_24h",
    "return_72h",

    "price_vs_ema20",
    "price_vs_ema50",
    "price_vs_ema200",
    "ema50_vs_ema200",

    "rsi_14",

    "volatility_24h",
    "volatility_72h",

    "range_pct",
    "volume_ratio"
]


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["timestamp"]
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df = df.dropna(
        subset=FEATURES + ["target"]
    )

    df["target"] = (
        df["target"].astype(int)
    )

    return df


# ==========================================
# MODELS
# ==========================================

def build_models():

    logistic = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])

    gradient_boosting = (
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=42
        )
    )

    return {
        "Logistic Regression":
            logistic,

        "Gradient Boosting":
            gradient_boosting
    }


# ==========================================
# WALK-FORWARD WINDOWS
# ==========================================

def create_walkforward_windows(df):

    """
    Expanding-window validation.

    We train on all history before
    each validation period.

    A 24-hour embargo separates training
    from validation because our target
    looks 24 hours into the future.
    """

    windows = [

        (
            "2024-01-01",
            "2024-06-30 23:00",
            "2024-07-02",
            "2024-09-30 23:00"
        ),

        (
            "2024-01-01",
            "2024-09-30 23:00",
            "2024-10-02",
            "2024-12-31 23:00"
        ),

        (
            "2024-01-01",
            "2024-12-31 23:00",
            "2025-01-02",
            "2025-03-31 23:00"
        ),

        (
            "2024-01-01",
            "2025-03-31 23:00",
            "2025-04-02",
            "2025-06-30 23:00"
        ),

        (
            "2024-01-01",
            "2025-06-30 23:00",
            "2025-07-02",
            "2025-09-30 23:00"
        ),

        (
            "2024-01-01",
            "2025-09-30 23:00",
            "2025-10-02",
            "2025-12-29 23:00"
        )
    ]

    return windows


# ==========================================
# METRICS
# ==========================================

def calculate_metrics(
    y_true,
    probabilities
):

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    return {

        "accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities
            ),

        "brier":
            brier_score_loss(
                y_true,
                probabilities
            )
    }


# ==========================================
# WALK FORWARD
# ==========================================

def run_walkforward(
    df,
    model_name,
    model
):

    windows = (
        create_walkforward_windows(df)
    )

    results = []

    predictions_all = []

    print("\n" + "=" * 70)

    print(
        f"MODEL: {model_name}"
    )

    print("=" * 70)

    for fold, (
        train_start,
        train_end,
        val_start,
        val_end

    ) in enumerate(
        windows,
        start=1
    ):

        train_mask = (
            (df["timestamp"] >= train_start)
            &
            (df["timestamp"] <= train_end)
        )

        validation_mask = (
            (df["timestamp"] >= val_start)
            &
            (df["timestamp"] <= val_end)
        )

        train = (
            df.loc[train_mask]
            .copy()
        )

        validation = (
            df.loc[validation_mask]
            .copy()
        )

        if (
            len(train) == 0
            or
            len(validation) == 0
        ):
            continue

        X_train = train[FEATURES]
        y_train = train["target"]

        X_val = validation[FEATURES]
        y_val = validation["target"]

        # ----------------------------------
        # TRAIN
        # ----------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------
        # PREDICT
        # ----------------------------------

        probabilities = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_val,
            probabilities
        )

        # ----------------------------------
        # STORE
        # ----------------------------------

        validation[
            "probability"
        ] = probabilities

        validation[
            "model"
        ] = model_name

        validation[
            "fold"
        ] = fold

        predictions_all.append(
            validation
        )

        results.append({

            "model":
                model_name,

            "fold":
                fold,

            "train_rows":
                len(train),

            "validation_rows":
                len(validation),

            **metrics
        })

        print(
            f"\nFold {fold}"
        )

        print(
            f"Train: "
            f"{train['timestamp'].min()} "
            f"→ "
            f"{train['timestamp'].max()}"
        )

        print(
            f"Validate: "
            f"{validation['timestamp'].min()} "
            f"→ "
            f"{validation['timestamp'].max()}"
        )

        print(
            f"Rows: "
            f"{len(train):,} train / "
            f"{len(validation):,} validation"
        )

        print(
            f"ROC AUC: "
            f"{metrics['roc_auc']:.3f}"
        )

        print(
            f"PR AUC:  "
            f"{metrics['pr_auc']:.3f}"
        )

        print(
            f"Precision:"
            f" {metrics['precision']:.2%}"
        )

        print(
            f"Recall:   "
            f"{metrics['recall']:.2%}"
        )

    return (
        pd.DataFrame(results),
        pd.concat(
            predictions_all,
            ignore_index=True
        )
    )


# ==========================================
# PROBABILITY ANALYSIS
# ==========================================

def probability_analysis(
    predictions,
    model_name
):

    print(
        f"\n{model_name}"
        f" — PROBABILITY BUCKETS"
    )

    bins = [
        0,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        1.00
    ]

    predictions = predictions.copy()

    predictions["bucket"] = pd.cut(
        predictions["probability"],
        bins=bins,
        include_lowest=True
    )

    analysis = (
        predictions
        .groupby(
            "bucket",
            observed=True
        )
        .agg(
            observations=(
                "target",
                "count"
            ),

            actual_positive_rate=(
                "target",
                "mean"
            ),

            avg_future_return=(
                "future_return_24h",
                "mean"
            )
        )
    )

    print(analysis)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    df = load_data()

    models = build_models()

    all_results = []
    all_predictions = []

    for (
        model_name,
        model
    ) in models.items():

        results, predictions = (
            run_walkforward(
                df,
                model_name,
                model
            )
        )

        all_results.append(
            results
        )

        all_predictions.append(
            predictions
        )

        probability_analysis(
            predictions,
            model_name
        )

    # ======================================
    # COMBINE RESULTS
    # ======================================

    results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    print("\n" + "=" * 70)

    print(
        "AVERAGE WALK-FORWARD PERFORMANCE"
    )

    print("=" * 70)

    summary = (
        results_df
        .groupby("model")
        [
            [
                "accuracy",
                "precision",
                "recall",
                "roc_auc",
                "pr_auc",
                "brier"
            ]
        ]
        .mean()
    )

    print(summary)

    results_df.to_csv(
        "results/ml_walkforward_results.csv",
        index=False
    )

    predictions_df.to_csv(
        "results/ml_walkforward_predictions.csv",
        index=False
    )