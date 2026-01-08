# src/portfolio.py

import pandas as pd
import numpy as np


def run_portfolio_backtest(
    strategy_results: dict,
    vol_window: int = 20,
    target_vol: float = 0.10  # 6.3% annualized target volatility
) -> pd.DataFrame:
    """
    Combines multiple strategy backtest DataFrames into a portfolio
    using:
    - Inverse volatility (risk parity) weighting
    - Portfolio volatility targeting
    """

    returns = pd.DataFrame()

    # ============================
    # Collect Strategy Returns
    # ============================
    for ticker, df in strategy_results.items():
        returns[ticker] = df["Strategy_Return"]

    returns = returns.dropna()

    # ============================
    # Asset Volatility Estimation
    # ============================
    asset_vol = returns.rolling(vol_window).std()

    # ============================
    # Inverse Volatility Weights
    # ============================
    inv_vol = 1 / asset_vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # ============================
    # Portfolio Raw Return
    # ============================
    portfolio_return = (weights * returns).sum(axis=1)
    # After computing portfolio returns
    equity = (1 + portfolio_return).cumprod()
    dd = equity / equity.cummax() - 1
    portfolio_return[dd < -0.10] *= 0.7



    # ============================
    # Portfolio Volatility Targeting
    # ============================
    portfolio_vol = portfolio_return.rolling(vol_window).std() * np.sqrt(252)

    scaling_factor = target_vol / portfolio_vol
    scaling_factor = scaling_factor.clip(upper=1.5)

    portfolio_return_scaled = portfolio_return * scaling_factor

    # ============================
    # Final Portfolio DataFrame
    # ============================
    portfolio = pd.DataFrame({
        "Portfolio_Return": portfolio_return_scaled
    })

    portfolio["Cumulative_Return"] = (1 + portfolio["Portfolio_Return"]).cumprod()

    return portfolio
