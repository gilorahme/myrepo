import pandas as pd
import numpy as np
from pathlib import Path

LABEL_HOURS = 24
DEVELOPMENT_END = pd.Timestamp("2026-01-01", tz="UTC")


def normalize_time(df, hourly=False):
    """Reject ambiguous time ordering; never silently repair missing candles."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    times = df["timestamp"]
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Timestamps must be nonmissing, unique and increasing.")
    if hourly and not times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Expected consecutive hourly candles; repair source gaps first.")
    return df


def validate_labels(df):
    """Require regenerated label provenance instead of guessing from legacy CSVs."""
    df = normalize_time(df)
    if "label_end_time" not in df:
        raise ValueError("Missing label_end_time: regenerate with features.py.")
    df["label_end_time"] = pd.to_datetime(df["label_end_time"], utc=True)
    known = df["target"].notna()
    expected = df["timestamp"] + pd.Timedelta(hours=LABEL_HOURS)
    if not df.loc[known, "label_end_time"].eq(expected[known]).all():
        raise ValueError("Known labels must end exactly 24 hours after their feature row.")
    if not df.loc[known, "target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary or missing.")
    return df


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

    df = normalize_time(df, hourly=True)

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

    Will BTC be more than 1% higher
    24 hours from now?

    IMPORTANT:
    Future data is used ONLY to create
    the training label, never as an input.
    """

    df = normalize_time(df, hourly=True)
    df["label_end_time"] = df["timestamp"].shift(-LABEL_HOURS)

    df["future_close_24h"] = (
        df["close"].shift(-LABEL_HOURS)
    )

    df["future_return_24h"] = (
        df["future_close_24h"] /
        df["close"] - 1
    )

    df["target"] = (
        df["future_return_24h"] > 0.01
    ).astype("Int64")
    df.loc[df["future_return_24h"].isna(), "target"] = pd.NA

    return df


if __name__ == "__main__":

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"]
    )

    # Features are causal for all rows. Training consumers enforce label cutoffs.

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df = build_features(df)

    df = create_target(df)

    # Keep inference rows; unknown future outcomes stay missing.
    feature_columns = [c for c in df.columns if c not in {
        "future_close_24h", "future_return_24h", "label_end_time", "target"
    }]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

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
