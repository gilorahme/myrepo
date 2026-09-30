import pandas as pd
import numpy as np


INPUT_FILE = "data/BTC_USDT.csv"
OUTPUT_FILE = "data/BTC_ML_features.csv"


def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (
        100 / (1 + rs)
    )


def build_features(df):

    df = df.copy()

    # =====================================
    # RETURNS / MOMENTUM
    # =====================================

    df["return_1h"] = (
        df["close"].pct_change(1)
    )

    df["return_6h"] = (
        df["close"].pct_change(6)
    )

    df["return_12h"] = (
        df["close"].pct_change(12)
    )

    df["return_24h"] = (
        df["close"].pct_change(24)
    )

    df["return_72h"] = (
        df["close"].pct_change(72)
    )

    # =====================================
    # TREND
    # =====================================

    df["ema_20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["ema_50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["ema_200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    df["price_vs_ema20"] = (
        df["close"] /
        df["ema_20"] - 1
    )

    df["price_vs_ema50"] = (
        df["close"] /
        df["ema_50"] - 1
    )

    df["price_vs_ema200"] = (
        df["close"] /
        df["ema_200"] - 1
    )

    df["ema50_vs_ema200"] = (
        df["ema_50"] /
        df["ema_200"] - 1
    )

    # =====================================
    # RSI
    # =====================================

    df["rsi_14"] = calculate_rsi(
        df["close"],
        14
    )

    # =====================================
    # VOLATILITY
    # =====================================

    df["volatility_24h"] = (
        df["return_1h"]
        .rolling(24)
        .std()
    )

    df["volatility_72h"] = (
        df["return_1h"]
        .rolling(72)
        .std()
    )

    # =====================================
    # CANDLE RANGE
    # =====================================

    df["range_pct"] = (
        (df["high"] - df["low"]) /
        df["close"]
    )

    # =====================================
    # VOLUME
    # =====================================

    df["volume_avg_24h"] = (
        df["volume"]
        .rolling(24)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg_24h"]
    )

    return df


def create_target(df):
    """
    Target:

    Will BTC be at least 1% higher
    24 hours from now?

    IMPORTANT:
    Future data is used ONLY to create
    the training label, never as an input.
    """

    df = df.copy()

    df["future_close_24h"] = (
        df["close"].shift(-24)
    )

    df["future_return_24h"] = (
        df["future_close_24h"] /
        df["close"] - 1
    )

    df["target"] = (
        df["future_return_24h"] > 0.01
    ).astype(int)

    return df


if __name__ == "__main__":

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"]
    )

    # =====================================
    # KEEP 2026 LOCKED
    # =====================================

    df = df[
        df["timestamp"] <
        "2026-01-01"
    ].copy()

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df = build_features(df)

    df = create_target(df)

    # Last 24 rows do not have a complete
    # future target and must be removed.

    df = df[
        df["future_close_24h"]
        .notna()
    ].copy()

    df = df.dropna()

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("             ML FEATURE DATASET")
    print("=" * 60)

    print(
        f"Observations:       "
        f"{len(df):,}"
    )

    print(
        f"Period:             "
        f"{df['timestamp'].min()} "
        f"→ "
        f"{df['timestamp'].max()}"
    )

    print(
        f"Positive targets:   "
        f"{df['target'].sum():,}"
    )

    print(
        f"Negative targets:   "
        f"{(df['target'] == 0).sum():,}"
    )

    print(
        f"Positive rate:      "
        f"{df['target'].mean():.2%}"
    )

    print("=" * 60)