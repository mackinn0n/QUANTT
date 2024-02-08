import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import tqdm

# Simulation Parameters:
num_simulations = 10_000  # Number of simulations to run
prediction_steps = 100  # Number of steps in each simulation
prediction_history = 365 # Number of days before simulation to use as predicition data
graph_history = 100 # Number of days before simulation to graph
simulation_display_count = 1000# Number of simulations to graph


# Step 1: Read the CSV file
df = pd.read_csv('TSLA.csv')
close_prices = df['Close']

# Step 2: Calculate the daily price changes
price_changes = close_prices.diff().dropna().to_list()[-prediction_history:]

# Fit KDE to the data
kde = gaussian_kde(price_changes, bw_method=0.1)

plt.figure(figsize=(10, 6))
x_grid = np.linspace(min(price_changes), max(price_changes), 1000)
plt.plot(x_grid, kde.evaluate(x_grid), label='Probability Distribution Function')
plt.fill_between(x_grid, kde.evaluate(x_grid), alpha=0.5)
plt.xlabel('Change in Price')
plt.ylabel('Density')
plt.title('Probability Density Function of Price Changes')
plt.grid(True)
plt.legend()
plt.show(block=False)

# Function to sample from the KDE
def sample_from_kde(kde, n=1):
    """Sample from the KDE."""
    return kde.resample(n)[0][0]

# Step 4: Monte Carlo Simulation
def monte_carlo_simulation(prev_prices, kde, steps):
    price_path = prev_prices.copy()
    current_price = prev_prices[-1]
    
    for _ in range(steps):
        # Sample a change from the KDE
        change = sample_from_kde(kde)  # Resample returns a 2D array, get the first value
        # Update the current price
        current_price += change
        price_path.append(current_price)
    
    return price_path


prev_prices = close_prices.to_list()[-graph_history:]  # Starting with the last available close price

simulation_results = []

x_axis = list(range(-graph_history + 1, 1)) + list(range(1, prediction_steps + 1))
plt.figure(figsize=(10, 6))

for i in tqdm.tqdm(range(num_simulations)):
    simulated_prices = monte_carlo_simulation(prev_prices, kde, prediction_steps)
    simulation_results.append(simulated_prices)
    if i < simulation_display_count:
        plt.plot(x_axis, simulated_prices, linewidth=1, alpha=0.5)

plt.title(f'First {simulation_display_count} of {num_simulations} Monte Carlo Simulations of TSLA Price Changes')
plt.xlabel('Time Steps')
plt.ylabel('Simulated Price')
plt.grid(True)
plt.show(block=False)

simulation_mean = np.mean(np.array(simulation_results), axis=0)

plt.figure(figsize=(10, 6))
plt.plot(x_axis[:graph_history], simulated_prices[:graph_history], linewidth=1, label='Historical Price')
plt.plot(x_axis[graph_history-1:], simulated_prices[graph_history-1:], linewidth=1, label='Simulated Price')
plt.title(f'{num_simulations} Monte Carlo Simulations of TSLA Price Changes')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()