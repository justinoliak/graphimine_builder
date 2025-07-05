#!/usr/bin/env python3
"""Plot nematic order parameter S vs time."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_S_vs_time(csv_file: str, output_file: str = None):
    """Plot S vs time from CSV file."""
    
    # Read data
    df = pd.read_csv(csv_file)
    
    # Convert timestep to time in LJ units (timestep * dt)
    dt = 0.002  # from LAMMPS input
    df['time'] = df['timestep'] * dt
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot S vs time
    plt.plot(df['time'], df['S'], 'b-', linewidth=2, alpha=0.8)
    plt.scatter(df['time'], df['S'], c='red', s=30, alpha=0.7, zorder=5)
    
    # Formatting
    plt.xlabel('Time (τ)', fontsize=14)
    plt.ylabel('Nematic Order Parameter S', fontsize=14)
    plt.title('Nematic Order Evolution: G=5, φ=0.10', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal lines for reference
    plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High order (S > 0.7)')
    plt.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate order (S > 0.3)')
    plt.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    
    # Add text annotations
    plt.text(0.02, 0.95, f'Initial S = {df["S"].iloc[0]:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.text(0.02, 0.85, f'Final S = {df["S"].iloc[-1]:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.text(0.02, 0.75, f'Average S = {df["S"].mean():.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    # Legend
    plt.legend(loc='upper right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    # Print statistics
    print(f"\n=== Nematic Evolution Statistics ===")
    print(f"Time range: {df['time'].min():.1f} - {df['time'].max():.1f} τ")
    print(f"S range: {df['S'].min():.4f} - {df['S'].max():.4f}")
    print(f"Initial S: {df['S'].iloc[0]:.4f}")
    print(f"Final S: {df['S'].iloc[-1]:.4f}")
    print(f"Average S: {df['S'].mean():.4f}")
    print(f"Standard deviation: {df['S'].std():.4f}")
    
    # Find equilibration time (when S drops below 50% of initial value)
    initial_S = df['S'].iloc[0]
    equilibrium_threshold = 0.5 * initial_S
    equilibrium_idx = np.where(df['S'] < equilibrium_threshold)[0]
    
    if len(equilibrium_idx) > 0:
        eq_time = df['time'].iloc[equilibrium_idx[0]]
        print(f"Approximate equilibration time: {eq_time:.1f} τ")
    
    return df

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Plot nematic order vs time")
    parser.add_argument("csv_file", help="CSV file with timestep,S data")
    parser.add_argument("--output", help="Output plot file (PNG/PDF)")
    args = parser.parse_args()
    
    # Plot the data
    df = plot_S_vs_time(args.csv_file, args.output)

if __name__ == "__main__":
    main()