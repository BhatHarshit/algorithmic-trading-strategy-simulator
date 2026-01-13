# src/portfolio.py

import pandas as pd
import numpy as np


def run_portfolio_backtest(
    strategy_results: dict,
    vol_window: int = 20,
    target_vol: float = 0.10,   # annualized target volatility
    regime_window: int = 60,    # Step B: regime detection window
    high_vol_multiplier: float = 1.3,
    min_exposure: float = 0.6,
    loss_window: int = 10,      # Step C: loss asymmetry window
    loss_threshold: float = -0.02,
    loss_cut: float = 0.8,
    recovery_rate: float = 0.02
) -> pd.DataFrame:
    """
    Combines multiple strategy backtest DataFrames into a portfolio using:
    - Inverse volatility (risk parity) weighting
    - Portfolio volatility targeting
    - Step A: drawdown-aware risk shaping
    - Step B: regime-aware volatility scaling
    - Step C: loss asymmetry dampening
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

    # ============================
    # Step A: Drawdown-Aware Risk Shaping
    # ============================
    equity = (1 + portfolio_return).cumprod()
    drawdown = equity / equity.cummax() - 1
    portfolio_return[drawdown < -0.10] *= 0.7

    # ============================
    # Portfolio Volatility Targeting
    # ============================
    portfolio_vol = portfolio_return.rolling(vol_window).std() * np.sqrt(252)
    scaling_factor = target_vol / portfolio_vol
    scaling_factor = scaling_factor.clip(upper=1.5)
    portfolio_return_scaled = portfolio_return * scaling_factor

    # ============================
    # Step B: Regime-Aware Scaling
    # ============================
    long_term_vol = (
        portfolio_return_scaled
        .rolling(regime_window)
        .std()
        * np.sqrt(252)
    )

    regime_ratio = portfolio_vol / long_term_vol
    regime_exposure = 1 / regime_ratio
    regime_exposure = regime_exposure.clip(lower=min_exposure, upper=1.0)
    portfolio_return_regime = portfolio_return_scaled * regime_exposure

    # ============================
    # Step C: Loss Asymmetry Dampening
    # ============================
    rolling_perf = (
        portfolio_return_regime
        .rolling(loss_window)
        .sum()
    )

    loss_exposure = pd.Series(1.0, index=portfolio_return_regime.index)

    for i in range(1, len(loss_exposure)):
        if rolling_perf.iloc[i] < loss_threshold:
            loss_exposure.iloc[i] = loss_exposure.iloc[i-1] * loss_cut
        else:
            loss_exposure.iloc[i] = min(
                1.0,
                loss_exposure.iloc[i-1] + recovery_rate
            )

    portfolio_return_final = portfolio_return_regime * loss_exposure

    # ============================
    # Final Portfolio DataFrame
    # ============================
    portfolio = pd.DataFrame({
        "Portfolio_Return": portfolio_return_final
    })

    portfolio["Cumulative_Return"] = (1 + portfolio["Portfolio_Return"]).cumprod()

    return portfolio
