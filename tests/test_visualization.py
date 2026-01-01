from src.data_loader import load_stock_data
from src.strategies.sma import sma_strategy
from src.backtest import run_backtest
from src.visualization import plot_equity_curve

df = load_stock_data("AAPL")
signals = sma_strategy(df)
bt = run_backtest(signals)

plot_equity_curve(bt, ticker="AAPL", save=True, show=False)
