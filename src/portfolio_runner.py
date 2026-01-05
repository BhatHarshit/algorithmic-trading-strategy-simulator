from src.data_loader import load_stock_data
from src.strategies.sma import sma_strategy
from src.backtest import run_backtest
from src.portfolio import run_portfolio_backtest
from src.performance import calculate_metrics


def run_portfolio(tickers):
    strategy_results = {}

    for ticker in tickers:
        df = load_stock_data(ticker)
        signals = sma_strategy(df)
        backtest_df = run_backtest(signals)
        strategy_results[ticker] = backtest_df

    portfolio_df = run_portfolio_backtest(strategy_results)
    metrics = calculate_metrics(portfolio_df, column="Portfolio_Return")

    return metrics, portfolio_df


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    metrics, portfolio_df = run_portfolio(tickers)
    print("Portfolio Metrics:", metrics)
