from src.data_loader import load_stock_data
from src.strategies.sma import sma_strategy
from src.strategies.rsi import rsi_strategy
from src.strategies.momentum import momentum_strategy
from src.backtest import run_backtest
from src.portfolio import run_portfolio_backtest
from src.performance import calculate_metrics


def run_portfolio(tickers):
    strategy_results = {}

    # =========================
    # Loop through tickers
    # =========================
    for ticker in tickers:
        df = load_stock_data(ticker)

        # =========================
        # Individual Strategies
        # =========================
        sma_df = sma_strategy(df)
        rsi_df = rsi_strategy(df)
        mom_df = momentum_strategy(df)

        # =========================
        # Combine Signals (Alpha Stacking)
        # =========================
        combined = df.copy()
        combined["Signal"] = (
            0.5 * sma_df["Signal"] +
            0.25 * rsi_df["Signal"] +
            0.25 * mom_df["Signal"]
        )

        # =========================
        # Backtest Combined Strategy
        # =========================
        backtest_df = run_backtest(combined)

        strategy_results[ticker] = backtest_df

    # =========================
    # Portfolio Construction
    # =========================
    portfolio_df = run_portfolio_backtest(strategy_results)
    metrics = calculate_metrics(portfolio_df, column="Portfolio_Return")

    return metrics, portfolio_df


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    metrics, portfolio_df = run_portfolio(tickers)
    print("Portfolio Metrics:", metrics)
