import pandas as pd
import numpy as np


TRADES_FILE = "results/development_trades.csv"
SIGNALS_FILE = "data/BTC_USDT_signals_dev.csv"


def load_data():

    trades = pd.read_csv(
        TRADES_FILE,
        parse_dates=[
            "entry_time",
            "exit_time"
        ]
    )

    market = pd.read_csv(
        SIGNALS_FILE,
        parse_dates=["timestamp"]
    )

    return trades, market


def enrich_trades(trades, market):
    """
    Attach ONLY information that was available
    before the trade was executed.

    A trade executes at candle T open.
    Therefore features must come from candle T-1.
    """

    market = market.copy()

    # -----------------------------------------
    # CREATE PREVIOUS-CANDLE FEATURES
    # -----------------------------------------

    feature_columns = [
        "close",
        "ema_50",
        "ema_200",
        "rsi_14",
        "volatility_24h",
        "return_1h",
        "volume"
    ]

    for column in feature_columns:

        market[
            f"prev_{column}"
        ] = market[column].shift(1)

    features = market[
        [
            "timestamp",
            "prev_close",
            "prev_ema_50",
            "prev_ema_200",
            "prev_rsi_14",
            "prev_volatility_24h",
            "prev_return_1h",
            "prev_volume"
        ]
    ].copy()

    features = features.rename(
        columns={
            "timestamp":
                "entry_time",

            "prev_close":
                "signal_close",

            "prev_ema_50":
                "entry_ema50",

            "prev_ema_200":
                "entry_ema200",

            "prev_rsi_14":
                "entry_rsi",

            "prev_volatility_24h":
                "entry_volatility",

            "prev_return_1h":
                "entry_return_1h",

            "prev_volume":
                "entry_volume"
        }
    )

    trades = trades.merge(
        features,
        on="entry_time",
        how="left"
    )

    # -----------------------------------------
    # TREND STRENGTH
    # -----------------------------------------

    trades["ema_distance_pct"] = (
        (
            trades["entry_ema50"] -
            trades["entry_ema200"]
        )
        /
        trades["entry_ema200"]
    )

    # -----------------------------------------
    # HOLDING PERIOD
    # -----------------------------------------

    trades["holding_hours"] = (
        (
            trades["exit_time"] -
            trades["entry_time"]
        )
        .dt.total_seconds()
        / 3600
    )

    # -----------------------------------------
    # TARGET
    # -----------------------------------------

    trades["winner"] = (
        trades["net_return"] > 0
    )

    return trades


def print_summary(trades):

    winners = trades[
        trades["winner"]
    ]

    losers = trades[
        ~trades["winner"]
    ]

    print("\n" + "=" * 65)
    print("              TRADE FORENSICS")
    print("=" * 65)

    print(
        f"\nTotal trades: "
        f"{len(trades)}"
    )

    print(
        f"Winners:      "
        f"{len(winners)}"
    )

    print(
        f"Losers:       "
        f"{len(losers)}"
    )

    print("\n--- RETURN ---")

    print(
        f"Winner avg:   "
        f"{winners['net_return'].mean():.2%}"
    )

    print(
        f"Loser avg:    "
        f"{losers['net_return'].mean():.2%}"
    )

    print("\n--- RSI AT ENTRY ---")

    print(
        f"Winners:      "
        f"{winners['entry_rsi'].mean():.2f}"
    )

    print(
        f"Losers:       "
        f"{losers['entry_rsi'].mean():.2f}"
    )

    print("\n--- EMA TREND STRENGTH ---")

    print(
        f"Winners:      "
        f"{winners['ema_distance_pct'].mean():.2%}"
    )

    print(
        f"Losers:       "
        f"{losers['ema_distance_pct'].mean():.2%}"
    )

    print("\n--- VOLATILITY AT ENTRY ---")

    print(
        f"Winners:      "
        f"{winners['entry_volatility'].mean():.4%}"
    )

    print(
        f"Losers:       "
        f"{losers['entry_volatility'].mean():.4%}"
    )

    print("\n--- HOLDING PERIOD ---")

    print(
        f"Winners:      "
        f"{winners['holding_hours'].mean():.1f} hours"
    )

    print(
        f"Losers:       "
        f"{losers['holding_hours'].mean():.1f} hours"
    )

    print("\n--- WORST 10 TRADES ---")

    worst = (
        trades
        .sort_values("net_return")
        .head(10)
    )

    columns = [
        "entry_time",
        "exit_time",
        "net_return",
        "entry_rsi",
        "ema_distance_pct",
        "entry_volatility",
        "holding_hours"
    ]

    print(
        worst[columns]
        .to_string(index=False)
    )

    print("\n" + "=" * 65)


def bucket_analysis(trades):

    print("\nVOLATILITY BUCKET ANALYSIS\n")

    trades = trades.copy()

    trades["volatility_bucket"] = pd.qcut(
        trades["entry_volatility"],
        q=4,
        duplicates="drop"
    )

    volatility_analysis = (
        trades
        .groupby(
            "volatility_bucket",
            observed=True
        )
        .agg(
            trades=("net_return", "count"),
            win_rate=("winner", "mean"),
            avg_return=("net_return", "mean")
        )
    )

    print(volatility_analysis)

    print("\nTREND-STRENGTH BUCKET ANALYSIS\n")

    trades["trend_bucket"] = pd.qcut(
        trades["ema_distance_pct"],
        q=4,
        duplicates="drop"
    )

    trend_analysis = (
        trades
        .groupby(
            "trend_bucket",
            observed=True
        )
        .agg(
            trades=("net_return", "count"),
            win_rate=("winner", "mean"),
            avg_return=("net_return", "mean")
        )
    )

    print(trend_analysis)


if __name__ == "__main__":

    trades, market = load_data()

    trades = enrich_trades(
        trades,
        market
    )

    trades.to_csv(
        "results/trade_forensics.csv",
        index=False
    )

    print_summary(trades)

    bucket_analysis(trades)