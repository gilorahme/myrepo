import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

INITIAL_CAPITAL = 1000.0

# Assumptions for our first simulation.
# We will replace these with actual exchange costs later.
TRADING_FEE = 0.001       # 0.10%
SLIPPAGE = 0.0005         # 0.05%

COST_PER_TRANSACTION = TRADING_FEE + SLIPPAGE

HOURS_PER_YEAR = 365.25 * 24


def load_data():
    """Load BTC data containing our strategy signals."""

    df = pd.read_csv(
        "data/BTC_USDT_signals_dev.csv",
        parse_dates=["timestamp"]
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )

    return df


def run_backtest(df):
    """
    Run a long/cash backtest.

    Important:
    The signal generated from candle T is applied
    starting with candle T+1.

    This prevents look-ahead bias.
    """

    df = df.copy()

    # --------------------------------------
    # MARKET RETURN
    # --------------------------------------

    df["market_return"] = df["close"].pct_change()

    # --------------------------------------
    # POSITION
    # --------------------------------------
    # Shift by one candle:
    #
    # Signal at T
    #     ↓
    # Position at T+1
    #
    # This prevents the strategy from using
    # information that was not yet available.

    df["position"] = (
        df["signal"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------
    # IDENTIFY TRANSACTIONS
    # --------------------------------------

    df["position_change"] = (
        df["position"]
        .diff()
        .fillna(df["position"])
    )

    # +1 = enter BTC
    # -1 = exit BTC
    #  0 = no transaction

    df["transaction"] = (
        df["position_change"]
        .abs()
    )

    # --------------------------------------
    # GROSS STRATEGY RETURN
    # --------------------------------------

    df["gross_strategy_return"] = (
        df["position"] *
        df["market_return"]
    )

    # --------------------------------------
    # TRANSACTION COST
    # --------------------------------------

    df["transaction_cost"] = (
        df["transaction"] *
        COST_PER_TRANSACTION
    )

    # --------------------------------------
    # NET STRATEGY RETURN
    # --------------------------------------

    df["strategy_return"] = (
        df["gross_strategy_return"] -
        df["transaction_cost"]
    )

    # --------------------------------------
    # EQUITY CURVES
    # --------------------------------------

    df["strategy_equity"] = (
        INITIAL_CAPITAL *
        (1 + df["strategy_return"].fillna(0))
        .cumprod()
    )

    df["buy_hold_equity"] = (
        INITIAL_CAPITAL *
        (1 + df["market_return"].fillna(0))
        .cumprod()
    )

    return df


def calculate_max_drawdown(equity):
    """Calculate maximum peak-to-trough drawdown."""

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
    ) - 1

    return drawdown.min()


def calculate_sharpe(returns):
    """
    Calculate annualized Sharpe ratio.

    Risk-free rate is set to zero for this
    first baseline implementation.
    """

    returns = returns.dropna()

    if returns.std() == 0:
        return np.nan

    return (
        returns.mean() /
        returns.std()
    ) * np.sqrt(HOURS_PER_YEAR)


def extract_trades(df):
    """
    Extract completed long trades.

    Entry occurs when position changes 0 -> 1.
    Exit occurs when position changes 1 -> 0.
    """

    trades = []

    entry_index = None

    for i in range(len(df)):

        change = df.iloc[i]["position_change"]

        if change == 1:
            entry_index = i

        elif change == -1 and entry_index is not None:

            entry = df.iloc[entry_index]
            exit_ = df.iloc[i]

            entry_price = entry["close"]
            exit_price = exit_["close"]

            gross_return = (
                exit_price / entry_price
            ) - 1

            # Entry + exit transaction costs
            net_return = (
                (1 + gross_return) *
                (1 - COST_PER_TRANSACTION) ** 2
            ) - 1

            trades.append({
                "entry_time":
                    entry["timestamp"],

                "exit_time":
                    exit_["timestamp"],

                "entry_price":
                    entry_price,

                "exit_price":
                    exit_price,

                "gross_return":
                    gross_return,

                "net_return":
                    net_return
            })

            entry_index = None

    return pd.DataFrame(trades)


def performance_report(df, trades):

    strategy_final = (
        df["strategy_equity"].iloc[-1]
    )

    hold_final = (
        df["buy_hold_equity"].iloc[-1]
    )

    strategy_return = (
        strategy_final /
        INITIAL_CAPITAL - 1
    )

    hold_return = (
        hold_final /
        INITIAL_CAPITAL - 1
    )

    strategy_dd = calculate_max_drawdown(
        df["strategy_equity"]
    )

    hold_dd = calculate_max_drawdown(
        df["buy_hold_equity"]
    )

    sharpe = calculate_sharpe(
        df["strategy_return"]
    )

    transactions = int(
        df["transaction"].sum()
    )

    estimated_cost = (
        df["transaction_cost"].sum()
    )

    if len(trades) > 0:

        winners = trades[
            trades["net_return"] > 0
        ]

        losers = trades[
            trades["net_return"] <= 0
        ]

        win_rate = (
            len(winners) /
            len(trades)
        )

        avg_win = (
            winners["net_return"].mean()
            if len(winners) > 0
            else 0
        )

        avg_loss = (
            losers["net_return"].mean()
            if len(losers) > 0
            else 0
        )

        gross_profit = (
            winners["net_return"].sum()
        )

        gross_loss = abs(
            losers["net_return"].sum()
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else np.inf
        )

    else:

        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    print("\n")
    print("=" * 55)
    print("          AI TRADING AGENT — BACKTEST")
    print("=" * 55)

    print(
        f"Period:              "
        f"{df['timestamp'].iloc[0]} "
        f"→ {df['timestamp'].iloc[-1]}"
    )

    print("-" * 55)

    print(
        f"Starting capital:     "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Strategy final:       "
        f"${strategy_final:,.2f}"
    )

    print(
        f"Buy & Hold final:     "
        f"${hold_final:,.2f}"
    )

    print("-" * 55)

    print(
        f"Strategy return:      "
        f"{strategy_return:.2%}"
    )

    print(
        f"Buy & Hold return:    "
        f"{hold_return:.2%}"
    )

    print("-" * 55)

    print(
        f"Completed trades:     "
        f"{len(trades)}"
    )

    print(
        f"Transactions:         "
        f"{transactions}"
    )

    print(
        f"Win rate:             "
        f"{win_rate:.2%}"
    )

    print(
        f"Average winner:       "
        f"{avg_win:.2%}"
    )

    print(
        f"Average loser:        "
        f"{avg_loss:.2%}"
    )

    print(
        f"Profit factor:        "
        f"{profit_factor:.2f}"
    )

    print("-" * 55)

    print(
        f"Strategy max DD:      "
        f"{strategy_dd:.2%}"
    )

    print(
        f"Buy & Hold max DD:    "
        f"{hold_dd:.2%}"
    )

    print(
        f"Strategy Sharpe:      "
        f"{sharpe:.2f}"
    )

    print("-" * 55)

    print(
        f"Cost assumption:      "
        f"{COST_PER_TRANSACTION:.2%} "
        f"per transaction"
    )

    print(
        f"Accumulated cost*:    "
        f"{estimated_cost:.2%}"
    )

    print(
        "*Sum of modeled return deductions, "
        "not literal dollars."
    )

    print("=" * 55)


def plot_results(df):

    Path("results").mkdir(
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        df["timestamp"],
        df["strategy_equity"],
        label="Trading Strategy"
    )

    plt.plot(
        df["timestamp"],
        df["buy_hold_equity"],
        label="BTC Buy & Hold"
    )

    plt.title(
        "AI Trading Agent vs BTC Buy & Hold"
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "results/equity_curve.png",
        dpi=150
    )

    plt.show()


if __name__ == "__main__":

    data = load_data()

    results = run_backtest(data)

    trades = extract_trades(results)

    Path("results").mkdir(
        exist_ok=True
    )

    results.to_csv(
        "results/backtest_results.csv",
        index=False
    )

    trades.to_csv(
        "results/trades.csv",
        index=False
    )

    performance_report(
        results,
        trades
    )

    plot_results(results)