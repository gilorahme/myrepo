import pandas as pd
import numpy as np
from pathlib import Path
from features import DEVELOPMENT_END, normalize_time


def development_only(predictions, prices):
    predictions = normalize_time(predictions)
    prices = normalize_time(prices)
    # Exclude signals AND execution prices from the viewed holdout.
    return (predictions[predictions["timestamp"] < DEVELOPMENT_END].copy(),
            prices[prices["timestamp"] < DEVELOPMENT_END].copy())


INITIAL_CAPITAL = 1000.0

TRADING_FEE = 0.001
SLIPPAGE = 0.0005

HOLDING_HOURS = 24

PREDICTIONS_FILE = (
    "results/ml_walkforward_predictions.csv"
)

PRICE_FILE = "data/BTC_USDT.csv"


def load_data():

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        parse_dates=["timestamp"]
    )

    prices = pd.read_csv(
        PRICE_FILE,
        parse_dates=["timestamp"]
    )

    # Logistic Regression only
    predictions = predictions[
        predictions["model"]
        == "Logistic Regression"
    ].copy()

    predictions = (
        predictions
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
        .reset_index(drop=True)
    )

    prices = (
        prices
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return development_only(predictions, prices)


def simulate_threshold(
    predictions,
    prices,
    threshold,
    trading_fee=None,
    slippage=None
):
    """
    Simple non-overlapping strategy:

    Signal at candle T close.

    If probability >= threshold:
        enter at T+1 open
        hold exactly 24 hours
        exit at open 24 hours later

    No overlapping positions.
    """

    predictions, prices = development_only(predictions, prices)
    trading_fee = TRADING_FEE if trading_fee is None else trading_fee
    slippage = SLIPPAGE if slippage is None else slippage
    capital = INITIAL_CAPITAL

    trades = []

    price_lookup = (
        prices
        .set_index("timestamp")
    )

    next_available_time = None

    for _, row in predictions.iterrows():

        signal_time = row["timestamp"]

        probability = row["probability"]

        if not np.isfinite(probability) or probability < threshold:
            continue

        # Prevent overlapping trades
        if (
            next_available_time
            is not None
            and signal_time <
            next_available_time
        ):
            continue

        # Entry occurs next hour
        entry_time = (
            signal_time
            + pd.Timedelta(hours=1)
        )

        # Hold 24 hours
        exit_time = (
            entry_time
            + pd.Timedelta(
                hours=HOLDING_HOURS
            )
        )

        if (
            entry_time
            not in price_lookup.index
            or
            exit_time
            not in price_lookup.index
        ):
            continue

        entry_open = (
            price_lookup
            .loc[entry_time, "open"]
        )

        exit_open = (
            price_lookup
            .loc[exit_time, "open"]
        )

        # ---------------------------------
        # REALISTIC EXECUTION
        # ---------------------------------

        buy_price = (
            entry_open *
            (1 + slippage)
        )

        sell_price = (
            exit_open *
            (1 - slippage)
        )

        # Entry fee
        entry_fee = (
            capital *
            trading_fee
        )

        investable = (
            capital -
            entry_fee
        )

        btc = (
            investable /
            buy_price
        )

        gross_exit = (
            btc *
            sell_price
        )

        # Exit fee
        exit_fee = (
            gross_exit *
            trading_fee
        )

        new_capital = (
            gross_exit -
            exit_fee
        )

        trade_return = (
            new_capital /
            capital - 1
        )

        trades.append({

            "signal_time":
                signal_time,

            "entry_time":
                entry_time,

            "exit_time":
                exit_time,

            "probability":
                probability,

            "entry_price":
                buy_price,

            "exit_price":
                sell_price,

            "net_return":
                trade_return,

            "capital_before":
                capital,

            "capital_after":
                new_capital
        })

        capital = new_capital

        next_available_time = exit_time

    return capital, pd.DataFrame(trades)


def calculate_stats(
    final_capital,
    trades
):

    if len(trades) == 0:

        return {
            "trades": 0,
            "final": INITIAL_CAPITAL,
            "return": 0,
            "win_rate": 0,
            "avg_trade": 0,
            "profit_factor": 0,
            "max_drawdown": 0
        }

    total_return = (
        final_capital /
        INITIAL_CAPITAL - 1
    )

    win_rate = (
        trades["net_return"] > 0
    ).mean()

    avg_trade = (
        trades["net_return"].mean()
    )

    winners = trades[
        trades["net_return"] > 0
    ]

    losers = trades[
        trades["net_return"] <= 0
    ]

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

    equity = pd.concat([
        pd.Series(
            [INITIAL_CAPITAL]
        ),
        trades["capital_after"]
    ]).reset_index(drop=True)

    peak = equity.cummax()

    drawdown = (
        equity / peak - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    return {

        "trades":
            len(trades),

        "final":
            final_capital,

        "return":
            total_return,

        "win_rate":
            win_rate,

        "avg_trade":
            avg_trade,

        "profit_factor":
            profit_factor,

        "max_drawdown":
            max_drawdown
    }


def print_stats(
    threshold,
    stats
):

    print("\n" + "=" * 60)

    print(
        f"ML THRESHOLD: "
        f"{threshold:.0%}"
    )

    print("=" * 60)

    print(
        f"Completed trades:    "
        f"{stats['trades']}"
    )

    print(
        f"Final capital:       "
        f"${stats['final']:,.2f}"
    )

    print(
        f"Total return:        "
        f"{stats['return']:.2%}"
    )

    print(
        f"Win rate:            "
        f"{stats['win_rate']:.2%}"
    )

    print(
        f"Average trade:       "
        f"{stats['avg_trade']:.2%}"
    )

    print(
        f"Profit factor:       "
        f"{stats['profit_factor']:.2f}"
    )

    print(
        f"Maximum drawdown:    "
        f"{stats['max_drawdown']:.2%}"
    )

def robustness_test(predictions, prices):

    thresholds = [
        0.65,
        0.70,
        0.75,
        0.80,
        0.85
    ]

    scenarios = [
        ("Normal", 0.0010, 0.0005),
        ("Stress 1.5x", 0.0015, 0.00075),
        ("Stress 2x", 0.0020, 0.0010)
    ]

    results = []

    Path("results").mkdir(parents=True, exist_ok=True)

    for (
        scenario,
        fee,
        slippage
    ) in scenarios:

        for threshold in thresholds:

            final_capital, trades = (
                simulate_threshold(
                    predictions,
                    prices,
                    threshold,
                    trading_fee=fee,
                    slippage=slippage
                )
            )

            stats = calculate_stats(
                final_capital,
                trades
            )

            if scenario == "Normal":
                trades.to_csv(
                    f"results/ml_trades_{int(threshold * 100)}.csv",
                    index=False
                )

            results.append({

                "scenario":
                    scenario,

                "threshold":
                    threshold,

                **stats
            })

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 90)
    print("ROBUSTNESS TEST")
    print("=" * 90)

    columns = [
        "scenario",
        "threshold",
        "trades",
        "return",
        "win_rate",
        "avg_trade",
        "profit_factor",
        "max_drawdown"
    ]

    print(
        results_df[columns]
        .to_string(
            index=False,
            formatters={
                "threshold":
                    "{:.0%}".format,

                "return":
                    "{:.2%}".format,

                "win_rate":
                    "{:.2%}".format,

                "avg_trade":
                    "{:.2%}".format,

                "profit_factor":
                    "{:.2f}".format,

                "max_drawdown":
                    "{:.2%}".format
            }
        )
    )

    results_df.to_csv(
        "results/ml_robustness.csv",
        index=False
    )

    return results_df

if __name__ == "__main__":

    Path("results").mkdir(parents=True, exist_ok=True)
    predictions, prices = (
        load_data()
    )

    # Diagnostic thresholds only.
    # Do NOT choose a final threshold yet.

    thresholds = [
        0.50,
        0.60,
        0.70,
        0.75,
        0.80
    ]

    summary = []

    for threshold in thresholds:

        final_capital, trades = (
            simulate_threshold(
                predictions,
                prices,
                threshold
            )
        )

        stats = calculate_stats(
            final_capital,
            trades
        )

        print_stats(
            threshold,
            stats
        )

        summary.append({
            "threshold":
                threshold,
            **stats
        })

        trades.to_csv(
            f"results/"
            f"ml_trades_"
            f"{int(threshold*100)}.csv",
            index=False
        )

    summary_df = pd.DataFrame(
        summary
    )

    print("\n" + "=" * 75)

    print(
        "THRESHOLD COMPARISON"
    )

    print("=" * 75)

    print(
        summary_df.to_string(
            index=False
        )
    )

    summary_df.to_csv(
        "results/"
        "ml_threshold_comparison.csv",
        index=False
    )

    robustness_test(
        predictions,
        prices
    )