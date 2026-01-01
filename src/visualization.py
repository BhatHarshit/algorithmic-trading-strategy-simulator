# src/visualization.py

import matplotlib.pyplot as plt
import os

FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_equity_curve(
    df,
    ticker="",
    save: bool = True,
    show: bool = True
):
    plt.figure(figsize=(12, 6))

    plt.plot(df["Cumulative_Market"], label="Market")
    plt.plot(df["Cumulative_Strategy"], label="Strategy")

    plt.title(f"Equity Curve Comparison - {ticker}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)

    if save:
        filename = f"{FIGURES_DIR}/equity_curve_{ticker}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved figure → {filename}")

    if show:
        plt.show()

    plt.close()
