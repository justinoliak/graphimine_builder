#!/usr/bin/env python3

import sys
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def read_xyz(filename):
    """Read XYZ file and return atoms and coordinates."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    
    atoms = []
    coords = []
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].strip().split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return atoms, np.array(coords), comment

def write_xyz(filename, atoms, coords, comment="Optimized structure"):
    """Write XYZ file."""
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom, coord in zip(atoms, coords):
            f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

def analyze_and_optimize_bonds(atoms, coords, max_iter=500):
    """Analyze bond lengths and optimize if needed."""
    
    bond_targets = {
        ('C', 'C'): 1.40,
        ('C', 'N'): 1.35,
        ('C', 'H'): 1.09,
        ('N', 'H'): 1.01
    }
    
    bond_thresholds = {
        ('C', 'C'): 1.8,
        ('C', 'N'): 1.7,
        ('C', 'H'): 1.3,
        ('N', 'H'): 1.2
    }
    
    # First, analyze the current structure
    bond_stats = {}
    total_bonds = 0
    
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            vec = coords[j] - coords[i]
            current_length = np.linalg.norm(vec)
            
            pair = tuple(sorted([atoms[i], atoms[j]]))
            threshold = bond_thresholds.get(pair, 3.0)
            
            if current_length < threshold:
                if pair not in bond_stats:
                    bond_stats[pair] = []
                bond_stats[pair].append(current_length)
                total_bonds += 1
    
    print(f"Found {total_bonds} bonds in structure:")
    for pair, lengths in bond_stats.items():
        avg_length = np.mean(lengths)
        target = bond_targets.get(pair, "N/A")
        print(f"  {pair[0]}-{pair[1]}: {len(lengths)} bonds, avg={avg_length:.3f}Å, target={target}Å")
    
    # Now optimize
    opt_coords = coords.copy()
    
    for iteration in range(max_iter):
        adjustments_made = 0
        
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                vec = opt_coords[j] - opt_coords[i]
                current_length = np.linalg.norm(vec)
                
                pair = tuple(sorted([atoms[i], atoms[j]]))
                threshold = bond_thresholds.get(pair, 3.0)
                target = bond_targets.get(pair)
                
                if current_length < threshold and target is not None:
                    deviation = current_length - target
                    
                    # Adjust if deviation is >5% (more sensitive)
                    if abs(deviation) > 0.05 * target:
                        adjustment_factor = 0.02  # Slightly more aggressive
                        new_length = current_length - deviation * adjustment_factor
                        
                        if current_length > 1e-8:
                            unit_vec = vec / current_length
                            displacement = unit_vec * (new_length - current_length) * 0.5
                            
                            opt_coords[i] -= displacement
                            opt_coords[j] += displacement
                            adjustments_made += 1
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Made {adjustments_made} adjustments")
        
        if adjustments_made < 50:  # More sensitive stopping
            print(f"Converged after {iteration+1} iterations")
            break
    
    return opt_coords

def optimize_with_simple_method(xyz_file, output_file=None, quick=False):
    """Optimize structure using simple geometry optimization."""
    
    # Read XYZ file
    atoms, coords, comment = read_xyz(xyz_file)
    
    print(f"Optimizing {len(atoms)} atoms...")
    
    # Analyze and optimize
    max_iter = 100 if quick else 500
    opt_coords = analyze_and_optimize_bonds(atoms, coords, max_iter)
    
    # Write output
    if output_file is None:
        base, ext = os.path.splitext(xyz_file)
        output_file = base + "_optimized" + ext
    
    write_xyz(output_file, atoms, opt_coords, 
              comment + " - Gently optimized")
    
    print(f"Optimized structure written to: {output_file}")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_structure.py <xyz_file> [output_file] [--quick]")
        print("  --quick: Use faster optimization (fewer iterations)")
        sys.exit(1)
    
    xyz_file = sys.argv[1]
    output_file = None
    quick = False
    
    # Parse arguments
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--quick":
            quick = True
        elif not arg.startswith("--"):
            output_file = arg
    
    if not os.path.exists(xyz_file):
        print(f"Error: File {xyz_file} not found")
        sys.exit(1)
    
    print(f"Optimizing {xyz_file}...")
    if quick:
        print("Using quick optimization mode")
    
    success = optimize_with_simple_method(xyz_file, output_file, quick)
    
    if success:
        print("Optimization completed successfully!")
    else:
        print("Optimization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()