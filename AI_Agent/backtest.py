import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


INITIAL_CAPITAL = 1000.0

TRADING_FEE = 0.001
SLIPPAGE = 0.0005

ENTRY_COST = TRADING_FEE + SLIPPAGE
EXIT_COST = TRADING_FEE + SLIPPAGE

HOURS_PER_YEAR = 365.25 * 24


def load_data():
    """
    Load development data containing:
    OHLCV prices, indicators and strategy signals.
    """

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

    # Safety check
    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "signal"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    print(
        f"Loaded {len(df):,} development candles."
    )

    print(
        f"Period: "
        f"{df['timestamp'].iloc[0]} "
        f"→ "
        f"{df['timestamp'].iloc[-1]}"
    )

    return df


def run_backtest(df):

    df = df.copy()

    cash = INITIAL_CAPITAL
    btc = 0.0

    in_position = False

    equity_history = []
    trades = []

    entry_price = None
    entry_time = None
    entry_equity = None

    # Signal from previous completed candle
    previous_signal = 0

    for i in range(1, len(df)):

        row = df.iloc[i]

        current_open = row["open"]

        # =====================================
        # ENTRY
        # =====================================

        if previous_signal == 1 and not in_position:

            execution_price = (
                current_open *
                (1 + SLIPPAGE)
            )

            fee = (
                cash *
                TRADING_FEE
            )

            investable_cash = cash - fee

            btc = (
                investable_cash /
                execution_price
            )

            entry_price = execution_price
            entry_time = row["timestamp"]
            entry_equity = cash

            cash = 0.0
            in_position = True

        # =====================================
        # EXIT
        # =====================================

        elif previous_signal == 0 and in_position:

            execution_price = (
                current_open *
                (1 - SLIPPAGE)
            )

            gross_cash = (
                btc *
                execution_price
            )

            fee = (
                gross_cash *
                TRADING_FEE
            )

            cash = gross_cash - fee

            trade_return = (
                cash / entry_equity
            ) - 1

            trades.append({
                "entry_time": entry_time,
                "exit_time": row["timestamp"],
                "entry_price": entry_price,
                "exit_price": execution_price,
                "net_return": trade_return
            })

            btc = 0.0
            in_position = False

        # =====================================
        # PORTFOLIO VALUE
        # =====================================

        if in_position:

            equity = (
                btc *
                row["close"]
            )

        else:

            equity = cash

        equity_history.append({
            "timestamp": row["timestamp"],
            "equity": equity
        })

        # Signal becomes actionable next candle
        previous_signal = row["signal"]

    equity_df = pd.DataFrame(
        equity_history
    )

    trades_df = pd.DataFrame(
        trades
    )

    return equity_df, trades_df


def max_drawdown(equity):

    peak = equity.cummax()

    drawdown = (
        equity / peak
    ) - 1

    return drawdown.min()


def calculate_statistics(equity, trades):

    final_value = equity["equity"].iloc[-1]

    total_return = (
        final_value /
        INITIAL_CAPITAL - 1
    )

    dd = max_drawdown(
        equity["equity"]
    )

    hourly_returns = (
        equity["equity"]
        .pct_change()
        .dropna()
    )

    if hourly_returns.std() > 0:

        sharpe = (
            hourly_returns.mean() /
            hourly_returns.std()
        ) * np.sqrt(HOURS_PER_YEAR)

    else:

        sharpe = np.nan

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
            if len(winners)
            else 0
        )

        avg_loss = (
            losers["net_return"].mean()
            if len(losers)
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

    return {
        "final_value": final_value,
        "return": total_return,
        "max_drawdown": dd,
        "sharpe": sharpe,
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor
    }


def print_report(stats):

    print("\n" + "=" * 55)
    print("       TRADING AGENT — EXECUTION BACKTEST")
    print("=" * 55)

    print(
        f"Starting capital:     "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final capital:        "
        f"${stats['final_value']:,.2f}"
    )

    print(
        f"Total return:         "
        f"{stats['return']:.2%}"
    )

    print("-" * 55)

    print(
        f"Completed trades:     "
        f"{stats['trades']}"
    )

    print(
        f"Win rate:             "
        f"{stats['win_rate']:.2%}"
    )

    print(
        f"Average winner:       "
        f"{stats['avg_win']:.2%}"
    )

    print(
        f"Average loser:        "
        f"{stats['avg_loss']:.2%}"
    )

    print(
        f"Profit factor:        "
        f"{stats['profit_factor']:.2f}"
    )

    print("-" * 55)

    print(
        f"Maximum drawdown:     "
        f"{stats['max_drawdown']:.2%}"
    )

    print(
        f"Sharpe ratio:         "
        f"{stats['sharpe']:.2f}"
    )

    print("=" * 55)


def plot_equity(equity):

    Path("results").mkdir(
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        equity["timestamp"],
        equity["equity"],
        label="Trading Strategy"
    )

    plt.axhline(
        INITIAL_CAPITAL,
        linestyle="--",
        alpha=0.5,
        label="Starting Capital"
    )

    plt.title(
        "Trading Agent — Development Equity"
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "results/development_equity.png",
        dpi=150
    )

    plt.show()


if __name__ == "__main__":

    data = load_data()

    equity, trades = run_backtest(data)

    stats = calculate_statistics(
        equity,
        trades
    )

    Path("results").mkdir(
        exist_ok=True
    )

    equity.to_csv(
        "results/development_equity.csv",
        index=False
    )

    trades.to_csv(
        "results/development_trades.csv",
        index=False
    )

    print_report(stats)

    plot_equity(equity)