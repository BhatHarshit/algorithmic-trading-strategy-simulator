from src.data_loader import load_stock_data
from src.strategies.sma import sma_strategy
from src.backtest import run_backtest
from src.performance import calculate_strategy_metrics


df = load_stock_data("AAPL")
signals = sma_strategy(df)

bt = run_backtest(signals)
metrics = calculate_strategy_metrics(bt)

print(metrics)
