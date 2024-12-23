import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def monte_carlo_simulation(stock_symbol, num_simulations, num_days):
    # Step 1: Data Collection
    data = yf.download(stock_symbol, start="2015-01-01")
    closing_prices = data['Close']['AAPL']

    # Step 2: Data Preprocessing
    daily_returns = closing_prices.pct_change().dropna()
    mean_return = daily_returns.mean()
    std_dev = daily_returns.std()

    # Step 3: Simulation Setup using Geometric Brownian Motion
    simulations = np.zeros((num_simulations, num_days))
    S0 = closing_prices.iloc[-1].item()

    for i in range(num_simulations):
        simulated_prices = [S0]
        for _ in range(num_days):
            random_shock = np.random.normal(0, 1)
            drift = (mean_return - 0.5 * std_dev**2)
            diffusion = std_dev * random_shock
            simulated_price = simulated_prices[-1] * np.exp(drift + diffusion)
            simulated_prices.append(simulated_price)
        simulations[i, :] = simulated_prices[1:]

    # Step 4: Analysis (Optional: Calculate metrics such as average price at the end of the period)
    mean_simulated_end_price = np.mean(simulations[:, -1])
    print(f"Mean simulated price after {num_days} days: ${mean_simulated_end_price:.2f}")

    # Step 5: Option Pricing using Black-Scholes Model
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    risk_free_rate = 0.05       # Assumed risk-free rate
    strike_price = 250         # Example strike price (10% above current price)
    time_to_expiration = 1      # One year until option expiration

    call_option_price = black_scholes_call(S0, strike_price, time_to_expiration, risk_free_rate, sigma=std_dev*np.sqrt(252))
    print(f"Estimated Call Option Price: ${call_option_price:.2f}")

    # Step 6: Risk Analysis (Value at Risk)
    percentile_5 = np.percentile(simulations[:, -1], 5)
    var_95 = S0 - percentile_5
    print(f"95% Value at Risk: ${var_95:.2f}")

    # Step 7: Backtesting
    # def backtest_strategy(prices, simulated_mean):
    #     returns = prices.pct_change().dropna()
    #     X = returns.index.astype("int64").values.reshape(-1, 1)
    #     Y = returns.values

    #     # Train-Test Split
    #     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #     # Linear Regression Model
    #     model = LinearRegression().fit(X_train, y_train)
    #     score = model.score(X_test, y_test)
    #     print(f"Backtesting model accuracy (R^2 score): {score:.2f}")

    # backtest_strategy(closing_prices, mean_simulated_end_price)

    # Step 8: Parameter Optimization Using Machine Learning
    def optimize_parameters(data, num_simulations, num_days):
        # Simulate multiple scenarios with different parameters and evaluate the outcomes
        results = []
        for factor in np.linspace(0.5, 1.5, 10):
            adjusted_std_dev = std_dev * factor
            simulations = np.zeros((num_simulations, num_days))

            for i in range(num_simulations):
                simulated_prices = [S0]
                for _ in range(num_days):
                    random_shock = np.random.normal(0, 1)
                    drift = (mean_return - 0.5 * adjusted_std_dev**2)
                    diffusion = adjusted_std_dev * random_shock
                    simulated_price = simulated_prices[-1] * np.exp(drift + diffusion)
                    simulated_prices.append(simulated_price)
                results.append((factor, np.mean(simulations[:, -1])))

        # Find the factor that leads to the best result
        optimal_factor = max(results, key=lambda x: x[1])[0]
        print(f"Optimal standard deviation adjustment factor: {optimal_factor:.2f}")

    optimize_parameters(closing_prices, num_simulations, num_days)

    # Step 9: Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(simulations.T, color='lightblue', alpha=0.3)
    plt.title(f'Monte Carlo Simulation: {stock_symbol} Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.show()

# Example Usage
monte_carlo_simulation('AAPL', num_simulations=1000, num_days=252)
