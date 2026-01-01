from src.data_loader import load_stock_data
from src.strategies.sma import sma_strategy

# Load one stock
df = load_stock_data("AAPL")

# Apply SMA strategy
signals = sma_strategy(df)

# View last 5 rows
print(signals.tail())
