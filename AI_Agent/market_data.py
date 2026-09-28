import ccxt
import pandas as pd
import time
from pathlib import Path


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def get_historical_data(
    symbol="BTC/USDT",
    timeframe="1h",
    since="2024-01-01"
):
    """
    Download historical OHLCV data using pagination.
    """

    exchange = ccxt.binance({
        "enableRateLimit": True
    })

    since_ms = exchange.parse8601(
        f"{since}T00:00:00Z"
    )

    all_candles = []

    print(
        f"\nDownloading {symbol} {timeframe} "
        f"data since {since}..."
    )

    while True:

        candles = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=1000
        )

        if not candles:
            break

        all_candles.extend(candles)

        # Move starting point beyond the last candle
        since_ms = candles[-1][0] + 1

        print(
            f"\rDownloaded {len(all_candles):,} candles",
            end=""
        )

        # We have reached the latest available candle
        if len(candles) < 1000:
            break

        time.sleep(exchange.rateLimit / 1000)

    print()

    df = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    # Defensive cleanup
    df = (
        df
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


def save_market_data(df, symbol):

    filename = symbol.replace("/", "_")

    filepath = DATA_DIR / f"{filename}.csv"

    df.to_csv(filepath, index=False)

    print(f"Saved {len(df):,} candles to {filepath}")


if __name__ == "__main__":

    # Bitcoin
    btc = get_historical_data(
        symbol="BTC/USDT",
        timeframe="1h",
        since="2024-01-01"
    )

    save_market_data(
        btc,
        "BTC_USDT"
    )

    # Ethereum
    eth = get_historical_data(
        symbol="ETH/USDT",
        timeframe="1h",
        since="2024-01-01"
    )

    save_market_data(
        eth,
        "ETH_USDT"
    )

    print("\nDownload complete.")

    print(
        f"BTC: {len(btc):,} candles "
        f"from {btc['timestamp'].min()} "
        f"to {btc['timestamp'].max()}"
    )

    print(
        f"ETH: {len(eth):,} candles "
        f"from {eth['timestamp'].min()} "
        f"to {eth['timestamp'].max()}"
    )

import ccxt
import pandas as pd
import time
from pathlib import Path


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def get_historical_data(
    symbol="BTC/USDT",
    timeframe="1h",
    since="2024-01-01"
):
    """
    Download historical OHLCV data using pagination.
    """

    exchange = ccxt.binance({
        "enableRateLimit": True
    })

    since_ms = exchange.parse8601(
        f"{since}T00:00:00Z"
    )

    all_candles = []

    print(
        f"\nDownloading {symbol} {timeframe} "
        f"data since {since}..."
    )

    while True:

        candles = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=1000
        )

        if not candles:
            break

        all_candles.extend(candles)

        # Move starting point beyond the last candle
        since_ms = candles[-1][0] + 1

        print(
            f"\rDownloaded {len(all_candles):,} candles",
            end=""
        )

        # We have reached the latest available candle
        if len(candles) < 1000:
            break

        time.sleep(exchange.rateLimit / 1000)

    print()

    df = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    # Defensive cleanup
    df = (
        df
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


def save_market_data(df, symbol):

    filename = symbol.replace("/", "_")

    filepath = DATA_DIR / f"{filename}.csv"

    df.to_csv(filepath, index=False)

    print(f"Saved {len(df):,} candles to {filepath}")


if __name__ == "__main__":

    # Bitcoin
    btc = get_historical_data(
        symbol="BTC/USDT",
        timeframe="1h",
        since="2024-01-01"
    )

    save_market_data(
        btc,
        "BTC_USDT"
    )

    # Ethereum
    eth = get_historical_data(
        symbol="ETH/USDT",
        timeframe="1h",
        since="2024-01-01"
    )

    save_market_data(
        eth,
        "ETH_USDT"
    )

    print("\nDownload complete.")

    print(
        f"BTC: {len(btc):,} candles "
        f"from {btc['timestamp'].min()} "
        f"to {btc['timestamp'].max()}"
    )

    print(
        f"ETH: {len(eth):,} candles "
        f"from {eth['timestamp'].min()} "
        f"to {eth['timestamp'].max()}"
    )