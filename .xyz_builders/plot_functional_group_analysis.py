#!/usr/bin/env python3
"""
Create scientific plots for functional group analysis of graphimine structures
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Read the data
df = pd.read_csv('functional_group_counts.txt', sep='\t')

# Separate circle and hexagon data
circle_data = df[df['Script_Type'] == 'circular'].copy()
hexagon_data = df[df['Script_Type'] == 'hexagon'].copy()

# Calculate imine/aldehyde ratio
circle_data['Imine_Aldehyde_Ratio'] = circle_data['Imine_Count'] / circle_data['Aldehyde_Count']
hexagon_data['Imine_Aldehyde_Ratio'] = hexagon_data['Imine_Count'] / hexagon_data['Aldehyde_Count']

# Sort data for fitting
circle_data = circle_data.sort_values('G_Target')
hexagon_data = hexagon_data.sort_values('G_Target')

# Define fitting functions
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

def power_law(x, a, b):
    return a * x**b

# Set up the plotting style
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Create the four plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Imine Count vs Aldehyde Count for Circle
ax1.scatter(circle_data['Aldehyde_Count'], circle_data['Imine_Count'], 
           color='blue', alpha=0.7, s=30, edgecolors='darkblue', linewidth=0.5, label='Data')

# Fit quadratic function
x1 = circle_data['Aldehyde_Count'].values
y1 = circle_data['Imine_Count'].values
popt1, _ = curve_fit(quadratic, x1, y1)
x1_fit = np.linspace(x1.min(), x1.max(), 100)
y1_fit = quadratic(x1_fit, *popt1)
ax1.plot(x1_fit, y1_fit, 'r-', linewidth=2, 
         label=f'Fit: y = {popt1[0]:.1f}x² + {popt1[1]:.1f}x + {popt1[2]:.0f}')

ax1.set_xlabel('Aldehyde Count')
ax1.set_ylabel('Imine Count')
ax1.set_title('Circular Graphimine: Imine vs Aldehyde Count')
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(style='plain', axis='both')
ax1.legend(fontsize=10)

# Plot 2: Imine Count vs Aldehyde Count for Hexagon
ax2.scatter(hexagon_data['Aldehyde_Count'], hexagon_data['Imine_Count'], 
           color='red', alpha=0.7, s=30, edgecolors='darkred', linewidth=0.5, label='Data')

# Fit quadratic function (should be curved like circular)
x2 = hexagon_data['Aldehyde_Count'].values
y2 = hexagon_data['Imine_Count'].values
popt2, _ = curve_fit(quadratic, x2, y2)
x2_fit = np.linspace(x2.min(), x2.max(), 100)
y2_fit = quadratic(x2_fit, *popt2)
ax2.plot(x2_fit, y2_fit, 'g-', linewidth=2, 
         label=f'Fit: y = {popt2[0]:.3f}x² + {popt2[1]:.1f}x + {popt2[2]:.0f}')

ax2.set_xlabel('Aldehyde Count')
ax2.set_ylabel('Imine Count')
ax2.set_title('Hexagonal Graphimine: Imine vs Aldehyde Count')
ax2.grid(True, alpha=0.3)
ax2.ticklabel_format(style='plain', axis='both')
ax2.legend(fontsize=10)

# Plot 3: Imine/Aldehyde Ratio vs Monomer Count for Circle
ax3.scatter(circle_data['Monomer_Count'], circle_data['Imine_Aldehyde_Ratio'], 
           color='green', alpha=0.7, s=30, edgecolors='darkgreen', linewidth=0.5, label='Data')

# Fit power law function for ratio vs monomer count
x3 = circle_data['Monomer_Count'].values
y3 = circle_data['Imine_Aldehyde_Ratio'].values
popt3, _ = curve_fit(power_law, x3, y3, p0=[1, 0.5])
x3_fit = np.linspace(x3.min(), x3.max(), 100)
y3_fit = power_law(x3_fit, *popt3)
ax3.plot(x3_fit, y3_fit, 'm-', linewidth=2, 
         label=f'Fit: y = {popt3[0]:.2f}x^{popt3[1]:.3f}')

ax3.set_xlabel('Monomer Count (N)')
ax3.set_ylabel('Imine/Aldehyde Ratio')
ax3.set_title('Circular Graphimine: I/A Ratio vs Monomer Count')
ax3.grid(True, alpha=0.3)
ax3.ticklabel_format(style='plain', axis='both')
ax3.legend(fontsize=10)

# Plot 4: Imine/Aldehyde Ratio vs Monomer Count for Hexagon
ax4.scatter(hexagon_data['Monomer_Count'], hexagon_data['Imine_Aldehyde_Ratio'], 
           color='purple', alpha=0.7, s=30, edgecolors='indigo', linewidth=0.5, label='Data')

# Fit power law function for hexagon ratio vs monomer count
x4 = hexagon_data['Monomer_Count'].values
y4 = hexagon_data['Imine_Aldehyde_Ratio'].values
popt4, _ = curve_fit(power_law, x4, y4, p0=[1, 0.5])
x4_fit = np.linspace(x4.min(), x4.max(), 100)
y4_fit = power_law(x4_fit, *popt4)
ax4.plot(x4_fit, y4_fit, 'orange', linewidth=2, 
         label=f'Fit: y = {popt4[0]:.2f}x^{popt4[1]:.3f}')

ax4.set_xlabel('Monomer Count (N)')
ax4.set_ylabel('Imine/Aldehyde Ratio')
ax4.set_title('Hexagonal Graphimine: I/A Ratio vs Monomer Count')
ax4.grid(True, alpha=0.3)
ax4.ticklabel_format(style='plain', axis='both')
ax4.legend(fontsize=10)

# Adjust layout and save
plt.tight_layout(pad=3.0)
plt.savefig('functional_group_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print fitted function parameters
print("=== Fitted Functions ===")
print(f"\n1. Circular Imine vs Aldehyde:")
print(f"   Quadratic: y = {popt1[0]:.1f}x² + {popt1[1]:.1f}x + {popt1[2]:.0f}")

print(f"\n2. Hexagonal Imine vs Aldehyde:")
print(f"   Quadratic: y = {popt2[0]:.3f}x² + {popt2[1]:.1f}x + {popt2[2]:.0f}")

print(f"\n3. Circular I/A Ratio vs Monomer Count:")
print(f"   Power Law: y = {popt3[0]:.3f}x^{popt3[1]:.3f}")

print(f"\n4. Hexagonal I/A Ratio vs Monomer Count:")
print(f"   Power Law: y = {popt4[0]:.3f}x^{popt4[1]:.3f}")

# Print some summary statistics
print("\n=== Summary Statistics ===")
print(f"\nCircular Graphimine:")
print(f"G range: {circle_data['G_Target'].min()} - {circle_data['G_Target'].max()}")
print(f"Monomer count range: {circle_data['Monomer_Count'].min()} - {circle_data['Monomer_Count'].max()}")
print(f"Imine/Aldehyde ratio range: {circle_data['Imine_Aldehyde_Ratio'].min():.1f} - {circle_data['Imine_Aldehyde_Ratio'].max():.1f}")

print(f"\nHexagonal Graphimine:")
print(f"G range: {hexagon_data['G_Target'].min()} - {hexagon_data['G_Target'].max()}")
print(f"Monomer count range: {hexagon_data['Monomer_Count'].min()} - {hexagon_data['Monomer_Count'].max()}")
print(f"Imine/Aldehyde ratio range: {hexagon_data['Imine_Aldehyde_Ratio'].min():.1f} - {hexagon_data['Imine_Aldehyde_Ratio'].max():.1f}")