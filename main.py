import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2020-01-01'
end_date = '2025-06-01'

data = yf.download(tickers, start=start_date, end=end_date)['Close']


returns = np.log(data / data.shift(1)).dropna()

def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    returns = np.dot(weights, mean_returns)#portfolio return
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # portfolio volatility
    sharpe = (returns - risk_free_rate) / std_dev  # Sharpe ratio
    return returns, std_dev, sharpe

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    return -portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[2]
def check_sum(weights):
    return np.sum(weights) - 1

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0):
    num_assets = len(mean_returns)
    initail_weights = num_assets * [1. / num_assets,]
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': check_sum})

    result = minimize(
        neg_sharpe_ratio, 
        initail_weights, 
        args=(mean_returns, cov_matrix, risk_free_rate), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    return result

def simulate_portfolios(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.0):
    results = np.zeros((3, num_portfolios))
    weights_list = []

    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_list.append(weights)

        portfolio_return, portfolio_std_dev, sharpe_ratio = portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = sharpe_ratio
    
    return results, weights_list

mean_returns = returns.mean()
cov_matrix = returns.cov()
risk_free_rate = 0.01

results, weights_list = simulate_portfolios(mean_returns, cov_matrix)

opt = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
opt_ret, opt_vol, _ = portfolio_stats(opt.x, mean_returns, cov_matrix, risk_free_rate)

plt.figure(figsize=(12,8))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(opt_vol, opt_ret, color='red', marker='*', s=100, label='Max Sharpe Ratio Portfolio')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.show()

opt_weights = pd.Series(opt.x, index=tickers)
print("Optimal Portfolio Weights:\n", opt_weights)

