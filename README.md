# portfolio-optimizer
This repository contains a Python tool for portfolio optimization combining traditional mean-variance optimization with the Black-Litterman model. It allows you to incorporate your own market views and confidence levels into the portfolio construction process, producing more robust and personalized asset allocations.

---

## Features

- Download historical price data using `yfinance`
- Calculate log returns and covariance matrix
- Incorporate user views with the Black-Litterman model
- Optimize portfolios to maximize Sharpe ratio with diversification constraints
- Simulate random portfolios to visualize the efficient frontier
- Visualize portfolios and efficient frontier with Matplotlib
