#!/usr/bin/env python3
"""
Plot functional group data from functional_group_counts.txt
1) Imine vs Aldehyde for circular structures
2) Imine vs Aldehyde for hexagon structures  
3) H-Ratio (imine/aldehyde) vs N for circular
4) H-Ratio vs N for hexagon
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('functional_group_counts.txt', sep='\t')

# Filter for circular and hexagon structures from G1-150 range
circular = df[df['Script_Type'] == 'circular'].copy()
hexagon = df[df['Script_Type'] == 'hexagon'].copy()

# For circular/hexagon structures, N_Target is actually the generation G
# Calculate total monomers N from the relationship N = 3*G*(G+1) for hexagon, similar for circular
# For circular: approximately N ≈ 3*G^2 + 3*G + 1
# For hexagon: exactly N = 3*G*(G+1) + 1

# Estimate total monomers from functional group counts (more accurate)
# Each monomer contributes ~3 functional groups on average
circular['N_Total'] = (circular['Imine_Count'] + circular['Aldehyde_Count']) / 3
hexagon['N_Total'] = (hexagon['Imine_Count'] + hexagon['Aldehyde_Count']) / 3

# Calculate H-Ratio (imine/aldehyde)
circular['H_Ratio'] = circular['Imine_Count'] / circular['Aldehyde_Count']
hexagon['H_Ratio'] = hexagon['Imine_Count'] / hexagon['Aldehyde_Count']

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Imine vs Aldehyde for circular
ax1.scatter(circular['Aldehyde_Count'], circular['Imine_Count'], alpha=0.7, color='blue')
ax1.set_xlabel('Aldehyde Count')
ax1.set_ylabel('Imine Count')
ax1.set_title('Imine vs Aldehyde (Circular Structures)')
ax1.grid(True, alpha=0.3)

# Plot 2: Imine vs Aldehyde for hexagon
ax2.scatter(hexagon['Aldehyde_Count'], hexagon['Imine_Count'], alpha=0.7, color='red')
ax2.set_xlabel('Aldehyde Count')
ax2.set_ylabel('Imine Count')
ax2.set_title('Imine vs Aldehyde (Hexagon Structures)')
ax2.grid(True, alpha=0.3)

# Plot 3: H-Ratio vs N for circular
ax3.scatter(circular['N_Total'], circular['H_Ratio'], alpha=0.7, color='blue')
ax3.set_xlabel('Number of Monomers (N)')
ax3.set_ylabel('H-Ratio (Imine/Aldehyde)')
ax3.set_title('H-Ratio vs N (Circular Structures)')
ax3.grid(True, alpha=0.3)

# Plot 4: H-Ratio vs N for hexagon
ax4.scatter(hexagon['N_Total'], hexagon['H_Ratio'], alpha=0.7, color='red')
ax4.set_xlabel('Number of Monomers (N)')
ax4.set_ylabel('H-Ratio (Imine/Aldehyde)')
ax4.set_title('H-Ratio vs N (Hexagon Structures)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('functional_group_analysis.png', dpi=300, bbox_inches='tight')
# plt.show()  # Comment out to avoid timeout

# Print some statistics
print("=== Circular Structures ===")
print(f"G range: {circular['N_Target'].min()} - {circular['N_Target'].max()}")
print(f"N range: {circular['N_Total'].min():.0f} - {circular['N_Total'].max():.0f}")
print(f"H-Ratio range: {circular['H_Ratio'].min():.2f} - {circular['H_Ratio'].max():.2f}")
print(f"Mean H-Ratio: {circular['H_Ratio'].mean():.2f}")

print("\n=== Hexagon Structures ===")
print(f"G range: {hexagon['N_Target'].min()} - {hexagon['N_Target'].max()}")
print(f"N range: {hexagon['N_Total'].min():.0f} - {hexagon['N_Total'].max():.0f}")
print(f"H-Ratio range: {hexagon['H_Ratio'].min():.2f} - {hexagon['H_Ratio'].max():.2f}")
print(f"Mean H-Ratio: {hexagon['H_Ratio'].mean():.2f}")