import pandas as pd
import numpy as np
from pathlib import Path
from features import DEVELOPMENT_END, normalize_time, validate_labels



FEATURE_FILE = "data/BTC_ML_features.csv"
PRICE_FILE = "data/BTC_USDT.csv"

THRESHOLD = 0.75

INITIAL_CAPITAL = 1000.0

TRADING_FEE = 0.001
SLIPPAGE = 0.0005

HOLDING_HOURS = 24


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


def build_model():

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    return Pipeline([
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


def load_data():

    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["timestamp"]
    )

    prices = pd.read_csv(
        PRICE_FILE,
        parse_dates=["timestamp"]
    )

    features = (
        features
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    prices = (
        prices
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return features, prices


def prepare_holdout(features):

    features = validate_labels(features)
    train = features[
        (features["timestamp"] < DEVELOPMENT_END)
        & (features["label_end_time"] < DEVELOPMENT_END)
    ].copy()

    holdout = features[
        features["timestamp"]
        >= DEVELOPMENT_END
    ].copy()

    train = train.dropna(
        subset=FEATURES + ["target"]
    )

    holdout = holdout.dropna(
        subset=FEATURES
    )

    return train, holdout


def generate_predictions(
    train,
    holdout
):

    train = validate_labels(train)
    if train.empty or holdout.empty:
        raise ValueError("Training and holdout rows are required.")
    boundary = min(DEVELOPMENT_END, pd.to_datetime(holdout["timestamp"], utc=True).min())
    if not (train["label_end_time"] < boundary).all():
        raise ValueError("Training labels cross the holdout boundary.")
    model = build_model()

    X_train = train[FEATURES]
    y_train = train["target"]

    model.fit(
        X_train,
        y_train
    )

    holdout = holdout.copy()

    holdout["probability"] = (
        model.predict_proba(
            holdout[FEATURES]
        )[:, 1]
    )

    return holdout


def simulate(
    predictions,
    prices
):

    predictions = normalize_time(predictions)
    prices = normalize_time(prices)

    price_lookup = (
        prices
        .set_index("timestamp")
    )

    capital = INITIAL_CAPITAL

    next_available_time = None

    trades = []

    for _, row in predictions.iterrows():

        probability = row["probability"]

        if not np.isfinite(probability) or probability < THRESHOLD:
            continue

        signal_time = row["timestamp"]

        if (
            next_available_time
            is not None
            and signal_time <
            next_available_time
        ):
            continue

        entry_time = (
            signal_time
            + pd.Timedelta(hours=1)
        )

        exit_time = (
            entry_time
            + pd.Timedelta(
                hours=HOLDING_HOURS
            )
        )

        if (
            entry_time not in
            price_lookup.index
            or
            exit_time not in
            price_lookup.index
        ):
            continue

        entry_open = (
            price_lookup.loc[
                entry_time,
                "open"
            ]
        )

        exit_open = (
            price_lookup.loc[
                exit_time,
                "open"
            ]
        )

        buy_price = (
            entry_open *
            (1 + SLIPPAGE)
        )

        sell_price = (
            exit_open *
            (1 - SLIPPAGE)
        )

        entry_fee = (
            capital *
            TRADING_FEE
        )

        investable = (
            capital -
            entry_fee
        )

        btc = (
            investable /
            buy_price
        )

        gross_exit = (
            btc *
            sell_price
        )

        exit_fee = (
            gross_exit *
            TRADING_FEE
        )

        new_capital = (
            gross_exit -
            exit_fee
        )

        net_return = (
            new_capital /
            capital - 1
        )

        trades.append({

            "signal_time":
                signal_time,

            "entry_time":
                entry_time,

            "exit_time":
                exit_time,

            "probability":
                probability,

            "net_return":
                net_return,

            "capital_before":
                capital,

            "capital_after":
                new_capital
        })

        capital = new_capital

        next_available_time = (
            exit_time
        )

    return (
        capital,
        pd.DataFrame(trades)
    )


def report(
    final_capital,
    trades
):

    print("\n" + "=" * 65)

    print(
        "     PREVIOUSLY VIEWED HOLDOUT — 2026 (fixed diagnostic)"
    )

    print("=" * 65)

    print(
        f"Threshold:          "
        f"{THRESHOLD:.0%}"
    )

    print(
        f"Starting capital:   "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final capital:      "
        f"${final_capital:,.2f}"
    )

    total_return = (
        final_capital /
        INITIAL_CAPITAL - 1
    )

    print(
        f"Total return:       "
        f"{total_return:.2%}"
    )

    print(
        f"Trades:             "
        f"{len(trades)}"
    )

    if len(trades) == 0:
        return

    winners = (
        trades["net_return"] > 0
    )

    print(
        f"Win rate:           "
        f"{winners.mean():.2%}"
    )

    print(
        f"Average trade:      "
        f"{trades['net_return'].mean():.2%}"
    )

    print(
        f"Median trade:       "
        f"{trades['net_return'].median():.2%}"
    )

    gross_profit = (
        trades.loc[
            trades["net_return"] > 0,
            "net_return"
        ].sum()
    )

    gross_loss = abs(
        trades.loc[
            trades["net_return"] <= 0,
            "net_return"
        ].sum()
    )

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    print(
        f"Profit factor:      "
        f"{pf:.2f}"
    )

    equity = pd.concat([
        pd.Series(
            [INITIAL_CAPITAL]
        ),
        trades["capital_after"]
    ]).reset_index(drop=True)

    peak = equity.cummax()

    drawdown = (
        equity / peak - 1
    )

    print(
        f"Maximum drawdown:   "
        f"{drawdown.min():.2%}"
    )

    print("\nTRADES\n")

    print(
        trades[
            [
                "entry_time",
                "exit_time",
                "probability",
                "net_return"
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":

    Path("results").mkdir(parents=True, exist_ok=True)
    features, prices = load_data()

    train, holdout = (
        prepare_holdout(
            features
        )
    )

    print(
        f"\nTraining observations: "
        f"{len(train):,}"
    )

    print(
        f"Holdout observations:  "
        f"{len(holdout):,}"
    )

    if len(holdout) == 0:

        raise ValueError(
            "No 2026 observations found. "
            "BTC_ML_features.csv must contain "
            "2026 feature rows."
        )

    predictions = (
        generate_predictions(
            train,
            holdout
        )
    )

    final_capital, trades = (
        simulate(
            predictions,
            prices
        )
    )

    report(
        final_capital,
        trades
    )

    trades.to_csv(
        "results/final_holdout_2026.csv",
        index=False
    )
