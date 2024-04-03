import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from scipy.stats import norm
import tensorflow as tf


S = 168.38
X = 100
T = 2/365
r = 0.0436
sigma = 2.3672

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
   
    return round(call_price,2)

def monte_carlo(S, X, T, r, sigma, N=10, M=1_000):
    # Constants. These apply to the formula -----> delta(x) = u*delta(t) + sigma*delta(z)
    dt = T/N                                #   timestep. I.E., if T is 5 years and N is 10, this will give each time step as 0.5 years
    u_dt = (r-0.5*sigma**2)*dt                #   drift term (u)  
    sigma_sqrt_dt = sigma*np.sqrt(dt)           #   From equation on line 43, delta(z) becomes sqrt(delta(t)) when time step is applied
                                            #   -----> StockPriceNow = StockPricePrevious*e^(u*delta(t) + sigma*delta(z))
    # Standard error placeholders
    totalCT = 0
    totalCT2 = 0

    # MONTE CARLO METHOD
    for i in range(M):  #M is total simulations
        lnS = np.log(S)
        for j in range(N):  #N is total timesteps
            lnS = lnS + u_dt + sigma_sqrt_dt*np.random.normal()

        ST = np.exp(lnS) # e^ln cancels out and leaves the stock price as ST
        CT = max(0, ST - X) 
        totalCT = totalCT + CT
        totalCT2 = totalCT2 +CT*CT

    # Find the call value and the SE
    C0 = np.exp(-r*T)*totalCT/M    # C0 is call value. Comes from formula ------> C0 = (1/M)*(SIGMA(  M(top of sigma)...i=1(bottom of sigma) C0 ))
    sigma = np.sqrt((totalCT2 - totalCT*totalCT/M)*np.exp(-2*r*T) / (M-1)) # for standard error. Has nothing to do with uppercase SIGMA from line 65
    SE = sigma/np.sqrt(M) #find the standard error. This is essentially based on number of sims and call value at each time point

    return round(C0,2)

def binomial_tree(S, X, T, r, sigma, N = 100):
    
    # Calculations
    dt = T / N  # length of each step
    u = np.exp(sigma * np.sqrt(dt))  # up-factor
    d = 1 / u  # down-factor
    p = (np.exp(r * dt) - d) / (u - d)  # risk-neutral probability

    # Initialize the Stock Price Tree
    stock_price_tree = np.zeros((N + 1, N + 1))
    stock_price_tree[0, 0] = S
    for i in range(1, N + 1):
        stock_price_tree[i, 0] = stock_price_tree[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_price_tree[i, j] = stock_price_tree[i - 1, j - 1] * d

    # Initialize the Option Value Tree for a Call Option
    option_value_tree = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        option_value_tree[N, j] = max(0, stock_price_tree[N, j] - X)

    # Perform Backward Induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_value_tree[i, j] = np.exp(-r * dt) * (p * option_value_tree[i + 1, j] + (1 - p) * option_value_tree[i + 1, j + 1])

    # Result
    option_price = option_value_tree[0, 0]

    return round(option_price,2)

def neural_network(S, X, T, r, sigma):
    model = tf.keras.models.load_model("neural_network_model.keras")
    model.load_weights('neural_network_model.weights.h5')
    return round(model.predict(np.array([[S, X, T, r, sigma]]), verbose=0)[0][0],2)


print(f"Black Scholes:\t{black_scholes(S, X, T, r, sigma):.2f}")
print(f"Monte Carlo:\t{monte_carlo(S, X, T, r, sigma):.2f}")
print(f"Binomial Tree:\t{binomial_tree(S, X, T, r, sigma):.2f}")
print(f"Neural Network:\t{neural_network(S, X, T, r, sigma):.2f}")