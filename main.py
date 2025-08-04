import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import inv

def load_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = np.log(data / data.shift(1)).dropna()
    return returns

def calculate_bl_returns(returns, market_weights, tau, P, Q, confidence=2.0):
    cov_matrix = returns.cov()
    cov_np = cov_matrix.values
    pi = tau * cov_matrix.dot(market_weights)
    omega = np.diag(np.diag(P @ (tau * cov_matrix) @ P.T)) * confidence
    middle_term = inv(inv(tau * cov_np) + P.T @ inv(omega) @ P)
    adjusted_returns = middle_term @ (inv(tau * cov_np) @ pi + P.T @ inv(omega) @ Q)
    bl_mean_returns = pd.Series(adjusted_returns, index=returns.columns)
    return bl_mean_returns, cov_matrix

def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return portfolio_return, portfolio_std_dev, sharpe

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    return -portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def check_sum(weights):
    return np.sum(weights) - 1

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0, bounds=None):
    num_assets = len(mean_returns)
    init_weights = num_assets * [1. / num_assets]
    if bounds is None:
        bounds = tuple((0.05, 0.3) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': check_sum}
    result = minimize(neg_sharpe_ratio, init_weights,
                      args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds,
                      constraints=constraints)
    return result

def simulate_portfolios(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.0):
    results = np.zeros((3, num_portfolios))
    weights_list = []
    num_assets = len(mean_returns)

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_list.append(weights)

        ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe
    
    return results, weights_list

def plot_efficient_frontier(results, opt_vol, opt_ret, bl_vol, bl_ret):
    plt.figure(figsize=(12,8))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(opt_vol, opt_ret, color='red', marker='*', s=200, label='Traditional Max Sharpe')
    plt.scatter(bl_vol, bl_ret, color='blue', marker='X', s=200, label='Black-Litterman Portfolio')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start_date = '2020-01-01'
    end_date = '2025-06-01'
    market_weights = np.array([0.25, 0.25, 0.20, 0.20, 0.10])
    tau = 0.05
    risk_free_rate = 0.01

    returns = load_data(tickers, start_date, end_date)

    P = np.array([
        [1, -1, 0, 0, 0],
        [0,  0, 1, 0, 0],
    ])
    Q = np.array([0.06, 0.04])

    bl_mean_returns, cov_matrix = calculate_bl_returns(returns, market_weights, tau, P, Q)

    mean_returns = returns.mean()

    results, weights_list = simulate_portfolios(mean_returns, cov_matrix)

    opt = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
    bl_opt = optimize_portfolio(bl_mean_returns, cov_matrix, risk_free_rate)

    opt_ret, opt_vol, _ = portfolio_stats(opt.x, mean_returns, cov_matrix, risk_free_rate)
    bl_ret, bl_vol, _ = portfolio_stats(bl_opt.x, bl_mean_returns, cov_matrix, risk_free_rate)

    plot_efficient_frontier(results, opt_vol, opt_ret, bl_vol, bl_ret)

    opt_weights = pd.Series(opt.x, index=tickers)
    bl_opt_weights = pd.Series(bl_opt.x, index=tickers)

    print("🔴 Traditional Optimal Portfolio Weights:\n", opt_weights.round(4))
    print("\n🔵 Black-Litterman Optimal Portfolio Weights:\n", bl_opt_weights.round(4))

if __name__ == "__main__":
    main()
