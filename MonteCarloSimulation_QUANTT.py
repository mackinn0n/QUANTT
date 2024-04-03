

# Monte-Carlo Simulation

import math
import numpy as np 
import pandas as pd 
import datetime
import scipy.stats as stats 
import matplotlib.pyplot as plt 
from pandas_datareader import data as pdr 


# PROCESS
#   Gather options from Yahoo Finance TSLA options
#   Implement formula
#       - Risk-neutral pricing/probabilites
#           - Risk is implied through Geometric Brownian motion, and therefore risk term can be dropped
#       - Apply Girsanov Theorem
#       - Apply Radon-Nikodym derivative (removes drift term, ensures a martingale)
#   Write loop to iterate through 10,000 simulations for the option
# 
# IMPROVING MONTE-CARLO ---- for future adaptations
#   Variance Reduction Methods
#       - Antithetic Variates
#       - Control Variates
#   Quasi-random numbers (deterministic series) compared to pseudo random numbers


# Initial Stock Conditions - Can pull from Yahoo Finance for TSLA
# Below stock is a real yahoo finance option, expires on feb 2nd 
Stock_P = 9.9867 #stock price
Strike_P = 10.48083960978591 #strike price
vol = 0.5527    #implied volatility (in %)
r = 0.03        #risk-free rate (%)
N = 10          #number of times steps
M = 10_000       #number of simulations

T = 21/365 #The +1 is because you can still trade on the day the option expires
# Above gives time in years
def monte_carlo(Stock_P, Strike_P, vol, r, T, N=10, M=1_000):
    # Constants. These apply to the formula -----> delta(x) = u*delta(t) + sigma*delta(z)
    dt = T/N                                #   timestep. I.E., if T is 5 years and N is 10, this will give each time step as 0.5 years
    u_dt = (r-0.5*vol**2)*dt                #   drift term (u)  
    vol_sqrt_dt = vol*np.sqrt(dt)           #   From equation on line 43, delta(z) becomes sqrt(delta(t)) when time step is applied
                                            #   -----> StockPriceNow = StockPricePrevious*e^(u*delta(t) + sigma*delta(z))
    # Standard error placeholders
    totalCT = 0
    totalCT2 = 0

    # MONTE CARLO METHOD
    for i in range(M):  #M is total simulations
        lnStock_P = np.log(Stock_P)
        for j in range(N):  #N is total timesteps
            lnStock_P = lnStock_P + u_dt + vol_sqrt_dt*np.random.normal()

        ST = np.exp(lnStock_P) # e^ln cancels out and leaves the stock price as ST
        CT = max(0, ST - Strike_P) 
        totalCT = totalCT + CT
        totalCT2 = totalCT2 +CT*CT

    # Find the call value and the SE
    C0 = np.exp(-r*T)*totalCT/M    # C0 is call value. Comes from formula ------> C0 = (1/M)*(SIGMA(  M(top of sigma)...i=1(bottom of sigma) C0 ))
    sigma = np.sqrt((totalCT2 - totalCT*totalCT/M)*np.exp(-2*r*T) / (M-1)) # for standard error. Has nothing to do with uppercase SIGMA from line 65
    SE = sigma/np.sqrt(M) #find the standard error. This is essentially based on number of sims and call value at each time point

    return C0, SE

C0, SE = monte_carlo(Stock_P, Strike_P, vol, r, T)
print("The call value of the option is ", C0, "with a standard error of ", SE)


# NEXT STEPS

#   - The above framework generates one call value per run. Find a way to generate many at a time to form 
#       one of the typical monte carlo derivative pricing graphs
#   - It may also have a lot of bugs, needs to be debugged
#   - I manually enterred the stock initial condditions (lines 32-40) here. The code needs to pull options from online so they can be compared
#       - Yahoo Finance has all the initial conditions information

