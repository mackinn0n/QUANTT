
import numpy as np

# Parameters
S0 = 178.86  # initial Tesla stock price
K = 100  # strike price
T = 1  # time to maturity in years
r = 0.01  # risk-free interest rate, assume 1%
sigma = 0.30  # volatility of Tesla stock
N = 100  # number of steps

# Calculations
dt = T / N  # length of each step
u = np.exp(sigma * np.sqrt(dt))  # up-factor
d = 1 / u  # down-factor
p = (np.exp(r * dt) - d) / (u - d)  # risk-neutral probability

# Initialize the Stock Price Tree
stock_price_tree = np.zeros((N + 1, N + 1))
stock_price_tree[0, 0] = S0
for i in range(1, N + 1):
    stock_price_tree[i, 0] = stock_price_tree[i - 1, 0] * u
    for j in range(1, i + 1):
        stock_price_tree[i, j] = stock_price_tree[i - 1, j - 1] * d

# Initialize the Option Value Tree for a Call Option
option_value_tree = np.zeros((N + 1, N + 1))
for j in range(N + 1):
    option_value_tree[N, j] = max(0, stock_price_tree[N, j] - K)

# Perform Backward Induction
for i in range(N - 1, -1, -1):
    for j in range(i + 1):
        option_value_tree[i, j] = np.exp(-r * dt) * (p * option_value_tree[i + 1, j] + (1 - p) * option_value_tree[i + 1, j + 1])

# Result
option_price = option_value_tree[0, 0]
option_price
print(f"The option price is: {option_price}")
