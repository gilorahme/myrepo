import pandas as pd


def generate_signals(df):
    """
    Stateful trend/momentum strategy.

    signal:
        1 = long BTC
        0 = cash

    Entry and exit conditions are deliberately
    different to reduce excessive switching.
    """

    df = df.copy()

    df["signal"] = 0

    # Current portfolio state
    in_position = False

    for i in range(len(df)):

        ema50 = df.loc[i, "ema_50"]
        ema200 = df.loc[i, "ema_200"]
        rsi = df.loc[i, "rsi_14"]

        # Skip rows where indicators aren't ready
        if pd.isna(ema50) or pd.isna(ema200) or pd.isna(rsi):
            continue

        # =====================================
        # ENTRY
        # =====================================

        if not in_position:

            bullish_trend = (
                ema50 > ema200
            )

            momentum_ok = (
                40 <= rsi <= 55
            )

            if bullish_trend and momentum_ok:
                in_position = True

        # =====================================
        # EXIT
        # =====================================

        else:

            trend_broken = (
                ema50 < ema200
            )

            momentum_extended = (
                rsi >= 70
            )

            if trend_broken or momentum_extended:
                in_position = False

        # Store portfolio state
        df.loc[i, "signal"] = int(in_position)

    return df


if __name__ == "__main__":

    df = pd.read_csv(
        "data/BTC_USDT_indicators.csv",
        parse_dates=["timestamp"]
    )

    # --------------------------------------
    # DEVELOPMENT DATA ONLY
    # --------------------------------------

    df = df[
        df["timestamp"] < "2026-01-01"
    ].copy()

    df = df.reset_index(drop=True)

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
        "\nLatest DEVELOPMENT signals:\n"
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

    # Count transitions
    transitions = (
        df["signal"]
        .diff()
        .abs()
        .fillna(0)
        .sum()
    )

    print(
        f"\nSignal transitions: "
        f"{int(transitions)}"
    )

    df.to_csv(
        "data/BTC_USDT_signals_dev.csv",
        index=False
    )