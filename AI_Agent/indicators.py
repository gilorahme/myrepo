import pandas as pd


def calculate_rsi(series, period=14):
    """
    Calculate the Relative Strength Index (RSI).

    RSI ranges from 0 to 100.
    """

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_indicators(df):
    """
    Add technical indicators used by our trading strategy.
    """

    df = df.copy()

    # Exponential Moving Averages
    df["ema_50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["ema_200"] = df["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # Relative Strength Index
    df["rsi_14"] = calculate_rsi(
        df["close"],
        period=14
    )

    # Percentage return for each 1-hour candle
    df["return_1h"] = df["close"].pct_change()

    # 24-hour rolling volatility
    df["volatility_24h"] = (
        df["return_1h"]
        .rolling(window=24)
        .std()
    )

    return df


if __name__ == "__main__":

    df = pd.read_csv(
        "data/BTC_USDT.csv",
        parse_dates=["timestamp"]
    )

    df = add_indicators(df)

    print("\nLatest BTC data with indicators:\n")

    columns = [
        "timestamp",
        "close",
        "ema_50",
        "ema_200",
        "rsi_14",
        "return_1h",
        "volatility_24h"
    ]

    print(df[columns].tail(10).to_string(index=False))

    df.to_csv(
        "data/BTC_USDT_indicators.csv",
        index=False
    )

    print(
        "\nSaved indicators to: "
        "data/BTC_USDT_indicators.csv"
    )