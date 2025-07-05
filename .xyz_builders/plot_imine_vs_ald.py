#!/usr/bin/env python3
"""Plot Imine vs Aldehyde counts"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Read the data
df = pd.read_csv('all_graphimine_counts_new.csv')

# Convert to numeric 
df['imines'] = pd.to_numeric(df['imines'], errors='coerce')
df['aldehydes'] = pd.to_numeric(df['aldehydes'], errors='coerce')
df = df.dropna()

# Create plot
plt.figure(figsize=(10, 8))
plt.scatter(df['aldehydes'], df['imines'], alpha=0.6, s=20, color='purple')

# Quadratic fit
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

x_data = df['aldehydes'].values
y_data = df['imines'].values
popt_quad, _ = curve_fit(quadratic, x_data, y_data)

x_fit = np.linspace(x_data.min(), x_data.max(), 100)
y_fit_quad = quadratic(x_fit, *popt_quad)

plt.plot(x_fit, y_fit_quad, 'red', linewidth=2, 
         label=f'Quadratic: y = {popt_quad[0]:.4f}x² + {popt_quad[1]:.1f}x + {popt_quad[2]:.0f}')

plt.xlabel('Aldehyde Count (Edge)')
plt.ylabel('Imine Count (Interior)')
plt.title('Graphimine: Imine vs Aldehyde')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('imine_vs_aldehyde.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Plotted {len(df)} points")
print(f"Quadratic fit: y = {popt_quad[0]:.4f}x² + {popt_quad[1]:.1f}x + {popt_quad[2]:.0f}")
correlation = np.corrcoef(df['aldehydes'], df['imines'])[0, 1]
print(f"Correlation: {correlation:.3f}")

# Calculate R² for quadratic fit
y_pred_quad = quadratic(x_data, *popt_quad)
ss_res_quad = np.sum((y_data - y_pred_quad)**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared_quad = 1 - (ss_res_quad / ss_tot)

print(f"R² for quadratic fit: {r_squared_quad:.4f}")