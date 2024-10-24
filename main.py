import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def monte_carlo_sim(stock_symbol, num_sims, num_days):
    # Step 1: Data Collection
    data = yf.download(stock_symbol, start='2015-01-01')
    closing_prices = data['Close']

    # Step 2: Data Processing
    daily_returns = closing_prices.pct_change().dropna()
    mean_return = daily_returns.mean()
    std_dev = daily_returns.std()

    # Step 3: Simulation Setup
    sims = np.zeros((num_sims, num_days))

    # Step 4: Monte Carlo Simulatioon
    for i in range (num_sims):
        simulated_prices = [closing_prices.iloc[-1]]
        for _ in range(num_days):
            random_return = np.random.normal(mean_return, std_dev)
            simulated_price = simulated_prices[-1] * (1 + random_return)
            simulated_prices.append(simulated_price)
        sims[i, :] = simulated_prices[1:]

    # Step 5: Analysis
    mean_simulated_end_price = np.mean(sims[:, -1])
    print(f"Mean simulated price after {num_days} days: ${mean_simulated_end_price:.2f}")

    # # Calculate the median path
    # median_path = np.median(sims, axis=0)

    # # Step 6: Visualization
    # plt.figure(figsize=(12,6))
    # plt.plot(sims.T, color='lightblue', alpha=0.3)
    # plt.plot(median_path, color='red', linewidth=2, label='Most Likely Path (Median)')
    # plt.title(f'Monte Carlo Simulation: {stock_symbol} Stock Price Prediction')
    # plt.xlabel('Days')
    # plt.ylabel('Price ($)')
    # plt.show()

monte_carlo_sim('AAPL', num_sims=10_000, num_days=50)
