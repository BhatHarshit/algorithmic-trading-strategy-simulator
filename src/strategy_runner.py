from src.data_loader import load_stock_data
from src.strategies.sma import sma_strategy
from src.strategies.rsi import rsi_strategy
from src.strategies.momentum import momentum_strategy
from src.backtest import run_backtest
from src.performance import calculate_metrics


def run_all_strategies(ticker="AAPL"):
    df = load_stock_data(ticker)

    strategies = {
        "SMA": sma_strategy(df),
        "RSI": rsi_strategy(df),
        "Momentum": momentum_strategy(df),
    }

    results = {}

    for name, strategy_df in strategies.items():
        backtest_df = run_backtest(strategy_df)
        metrics = calculate_metrics(backtest_df)
        results[name] = metrics

    return results


if __name__ == "__main__":
    results = run_all_strategies("AAPL")
    for strat, metrics in results.items():
        print(strat, metrics)
