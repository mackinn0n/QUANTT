# Black Scholes Model

# print("Test commit2")
from scipy.stats import norm
import numpy as np

def black_scholes(S, X, T, r, sigma):
    """
    S: Current stock price
    X: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate (expressed as a decimal)
    sigma: Volatility of the stock's return
    """
    # Calculate d1 and d2
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
   
    # Calculate the call option price
    call_price = (S * norm.cdf(d1)) - (X * np.exp(-r * T) * norm.cdf(d2))
   
    # Calculate the put option price
    put_price = (X * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
   
    return call_price, put_price

# Example values for the Tesla stock (these would need to be updated with real values)
# Assuming these example values:
# Current stock price (S)
S = 166.63 #USD

# Strike price (X)
X = 165 # USD

# Time to maturity (T - t), assuming this is for an option expiring in 1 year
T = 3/365 

# Risk-free interest rate (r), assuming 1%
r = 0.0436

# Volatility of the stock's return (sigma), assuming 30%
sigma = 0.6146

# Calculate the call and put prices
call_price, put_price = black_scholes(S, X, T, r, sigma)
print("The call value of the option is ", call_price)
