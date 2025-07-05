#!/usr/bin/env python3
"""
Plot H-ratio (imine/aldehyde) vs N (monomer count) from all_graphimine_counts.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Read the data
df = pd.read_csv('all_graphimine_counts_new.csv')

# Convert to numeric to handle any string issues
df['monomers'] = pd.to_numeric(df['monomers'], errors='coerce')
df['imines'] = pd.to_numeric(df['imines'], errors='coerce')
df['aldehydes'] = pd.to_numeric(df['aldehydes'], errors='coerce')

# Remove any rows with NaN values
df = df.dropna()

# Calculate H-ratio (imine/aldehyde)
df['H_ratio'] = df['imines'] / df['aldehydes']

# Set up the plotting style
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

# Create the plot
plt.figure(figsize=(10, 8))

# Plot H-ratio vs N (monomers)
plt.scatter(df['monomers'], df['H_ratio'], alpha=0.6, s=20, color='green', edgecolors='darkgreen', linewidth=0.3)

# Define power law function for fitting
def power_law(x, a, b):
    return a * x**b

# Fit power law function
x_data = df['monomers'].values
y_data = df['H_ratio'].values
popt, _ = curve_fit(power_law, x_data, y_data, p0=[1, 0.5])

# Generate fit line
x_fit = np.linspace(x_data.min(), x_data.max(), 100)
y_fit = power_law(x_fit, *popt)

# Plot the fit
plt.plot(x_fit, y_fit, 'red', linewidth=2, 
         label=f'Fit: y = {popt[0]:.3f}x^{popt[1]:.3f}')

# Add derived circular model
# For circular platelets: H-ratio ∝ √N (theoretical derivation)
# Fit the circular model: y = C * x^0.5
def circular_model(x, c):
    return c * x**0.5

# Fit circular model parameters
popt_circular, _ = curve_fit(circular_model, x_data, y_data, p0=[1])

# Generate circular model line
y_circular = circular_model(x_fit, *popt_circular)

# Plot the circular model
plt.plot(x_fit, y_circular, 'blue', linewidth=2, linestyle='--',
         label=f'Circular Model: y = {popt_circular[0]:.3f}x^0.5')

plt.xlabel('Monomer Count (N)')
plt.ylabel('H-ratio (Imine/Aldehyde)')
plt.title('Random Graphimine: H-ratio vs Monomer Count')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ticklabel_format(style='plain', axis='both')

# Adjust layout and save
plt.tight_layout()
plt.savefig('hratio_vs_n_random.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Plotted {len(df)} data points")
print(f"N range: {df['monomers'].min()} to {df['monomers'].max()}")
print(f"H-ratio range: {df['H_ratio'].min():.1f} to {df['H_ratio'].max():.1f}")
print(f"Fitted function: y = {popt[0]:.3f}x^{popt[1]:.3f}")
print(f"Circular model: y = {popt_circular[0]:.3f}x^0.5")

# Print some statistics
print(f"\nStatistics:")
print(f"Mean H-ratio: {df['H_ratio'].mean():.1f}")
print(f"Median H-ratio: {df['H_ratio'].median():.1f}")
print(f"Std H-ratio: {df['H_ratio'].std():.1f}")