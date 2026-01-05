import pandas as pd


def run_portfolio_backtest(strategy_results: dict) -> pd.DataFrame:
    """
    Combines multiple strategy backtest DataFrames into a portfolio.

    strategy_results: dict[str, pd.DataFrame]
    """
    portfolio = pd.DataFrame()

    for ticker, df in strategy_results.items():
        portfolio[ticker] = df["Strategy_Return"]

    # Equal-weight portfolio
    portfolio["Portfolio_Return"] = portfolio.mean(axis=1)

    portfolio["Cumulative_Return"] = (1 + portfolio["Portfolio_Return"]).cumprod()

    return portfolio
