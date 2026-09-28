import pandas as pd


def generate_signals(df):
    """
    Generate trading signals.

    Signal:
        1 = Buy / Long
        0 = No position
    """

    df = df.copy()

    # Default: no position
    df["signal"] = 0

    # -------------------------
    # TREND CONDITION
    # -------------------------

    bullish_trend = (
        df["ema_50"] > df["ema_200"]
    )

    # -------------------------
    # MOMENTUM CONDITION
    # -------------------------

    momentum_condition = (
        (df["rsi_14"] > 35) &
        (df["rsi_14"] < 55)
    )

    # -------------------------
    # BUY CONDITION
    # -------------------------

    buy_condition = (
        bullish_trend &
        momentum_condition
    )

    df.loc[
        buy_condition,
        "signal"
    ] = 1

    return df


if __name__ == "__main__":

    df = pd.read_csv(
        "data/BTC_USDT_indicators.csv",
        parse_dates=["timestamp"]
    )

    df = generate_signals(df)

    columns = [
        "timestamp",
        "close",
        "ema_50",
        "ema_200",
        "rsi_14",
        "signal"
    ]

    print(
        "\nLatest strategy signals:\n"
    )

    print(
        df[columns]
        .tail(20)
        .to_string(index=False)
    )

    print(
        "\nSignal summary:"
    )

    print(
        df["signal"]
        .value_counts()
        .sort_index()
    )

    df.to_csv(
        "data/BTC_USDT_signals.csv",
        index=False
    )