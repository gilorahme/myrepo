import ccxt
import pandas as pd
from pathlib import Path


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def get_market_data(symbol="BTC/USDT", timeframe="1h", limit=1000):
    """
    Download historical OHLCV cryptocurrency data.

    OHLCV =
    Open
    High
    Low
    Close
    Volume
    """

    exchange = ccxt.binance({
        "enableRateLimit": True
    })

    print(f"Downloading {symbol} {timeframe} data...")

    candles = exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit
    )

    df = pd.DataFrame(
        candles,
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

    return df


def save_market_data(df, symbol):
    """Save downloaded market data to CSV."""

    filename = symbol.replace("/", "_")

    filepath = DATA_DIR / f"{filename}.csv"

    df.to_csv(filepath, index=False)

    print(f"Saved data to: {filepath}")


if __name__ == "__main__":

    btc = get_market_data("BTC/USDT")
    save_market_data(btc, "BTC_USDT")

    print("\nLatest BTC candles:")
    print(btc.tail())