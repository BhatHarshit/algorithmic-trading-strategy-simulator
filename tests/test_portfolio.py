from src.portfolio_runner import run_portfolio

metrics, portfolio_df = run_portfolio(["AAPL", "MSFT", "GOOGL"])

print("Portfolio Metrics:")
print(metrics)
print("\nPortfolio Tail:")
print(portfolio_df.tail())
