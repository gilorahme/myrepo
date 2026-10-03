import pandas as pd
import numpy as np


INITIAL_CAPITAL = 1000.0

THRESHOLDS = [
    0.70,
    0.75,
    0.80
]


def analyze_threshold(threshold):

    file = (
        f"results/"
        f"ml_trades_{int(threshold * 100)}.csv"
    )

    trades = pd.read_csv(
        file,
        parse_dates=[
            "signal_time",
            "entry_time",
            "exit_time"
        ]
    )

    trades = (
        trades
        .sort_values("entry_time")
        .reset_index(drop=True)
    )

    # ----------------------------------
    # PERIOD LABELS
    # ----------------------------------

    trades["year"] = (
        trades["entry_time"].dt.year
    )

    trades["quarter"] = (
        trades["entry_time"]
        .dt.to_period("Q")
        .astype(str)
    )

    # ----------------------------------
    # PRINT INDIVIDUAL TRADES
    # ----------------------------------

    print("\n" + "=" * 75)

    print(
        f"THRESHOLD {threshold:.0%}"
    )

    print("=" * 75)

    print(
        f"Total trades: "
        f"{len(trades)}"
    )

    # ----------------------------------
    # YEAR ANALYSIS
    # ----------------------------------

    yearly = (
        trades
        .groupby("year")
        .agg(
            trades=(
                "net_return",
                "count"
            ),

            win_rate=(
                "net_return",
                lambda x:
                    (x > 0).mean()
            ),

            avg_trade=(
                "net_return",
                "mean"
            ),

            median_trade=(
                "net_return",
                "median"
            ),

            best_trade=(
                "net_return",
                "max"
            ),

            worst_trade=(
                "net_return",
                "min"
            )
        )
    )

    print(
        "\nYEARLY PERFORMANCE\n"
    )

    print(yearly)

    # ----------------------------------
    # QUARTER ANALYSIS
    # ----------------------------------

    quarterly = (
        trades
        .groupby("quarter")
        .agg(
            trades=(
                "net_return",
                "count"
            ),

            win_rate=(
                "net_return",
                lambda x:
                    (x > 0).mean()
            ),

            avg_trade=(
                "net_return",
                "mean"
            ),

            median_trade=(
                "net_return",
                "median"
            ),

            total_trade_return=(
                "net_return",
                "sum"
            ),

            best_trade=(
                "net_return",
                "max"
            ),

            worst_trade=(
                "net_return",
                "min"
            )
        )
    )

    print(
        "\nQUARTERLY PERFORMANCE\n"
    )

    print(quarterly)

    # ----------------------------------
    # CONCENTRATION ANALYSIS
    # ----------------------------------

    sorted_returns = (
        trades["net_return"]
        .sort_values(
            ascending=False
        )
        .reset_index(drop=True)
    )

    total_positive = (
        trades.loc[
            trades["net_return"] > 0,
            "net_return"
        ].sum()
    )

    top_1 = (
        sorted_returns
        .head(1)
        .sum()
    )

    top_3 = (
        sorted_returns
        .head(3)
        .sum()
    )

    top_5 = (
        sorted_returns
        .head(5)
        .sum()
    )

    print(
        "\nPROFIT CONCENTRATION\n"
    )

    if total_positive > 0:

        print(
            f"Top 1 trade / gross profits: "
            f"{top_1 / total_positive:.2%}"
        )

        print(
            f"Top 3 trades / gross profits: "
            f"{top_3 / total_positive:.2%}"
        )

        print(
            f"Top 5 trades / gross profits: "
            f"{top_5 / total_positive:.2%}"
        )

    # ----------------------------------
    # REMOVE BEST TRADES
    # ----------------------------------

    print(
        "\nROBUSTNESS TO REMOVING WINNERS\n"
    )

    for n in [1, 3, 5]:

        remaining = (
            trades
            .sort_values(
                "net_return",
                ascending=False
            )
            .iloc[n:]
        )

        compounded = (
            (1 + remaining["net_return"])
            .prod() - 1
        )

        print(
            f"Return without top {n}: "
            f"{compounded:.2%}"
        )


if __name__ == "__main__":

    for threshold in THRESHOLDS:

        analyze_threshold(
            threshold
        )