#!/usr/bin/env python3
"""Calculate nematic order parameter S for hexagonal platelets from LAMMPS data files."""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

def read_lammps_data(filename: str) -> Tuple[np.ndarray, List[int]]:
    """Read atom coordinates and molecule IDs from LAMMPS data file."""
    atoms = []
    mol_ids = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find atoms section
    atoms_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            atoms_start = i + 2
            break
    
    if atoms_start is None:
        raise ValueError("Atoms section not found")
    
    # Read atom data
    for line in lines[atoms_start:]:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Bonds"):
            continue
            
        parts = line.split()
        if len(parts) >= 6:
            mol_id = int(parts[1])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            atoms.append([x, y, z])
            mol_ids.append(mol_id)
    
    return np.array(atoms), mol_ids

def group_by_molecule(coords: np.ndarray, mol_ids: List[int]) -> Dict[int, np.ndarray]:
    """Group coordinates by molecule ID."""
    molecules = {}
    for i, mol_id in enumerate(mol_ids):
        if mol_id not in molecules:
            molecules[mol_id] = []
        molecules[mol_id].append(coords[i])
    
    return {mol_id: np.array(coords) for mol_id, coords in molecules.items()}

def get_platelet_normal(coords: np.ndarray) -> np.ndarray:
    """Calculate platelet normal vector using PCA."""
    centered = coords - np.mean(coords, axis=0)
    cov = np.cov(centered.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Normal is eigenvector with smallest eigenvalue
    normal = eigenvecs[:, 0]
    return normal / np.linalg.norm(normal)

def calculate_nematic_s(normals: List[np.ndarray]) -> Tuple[float, np.ndarray]:
    """Calculate nematic order parameter S using S = (3⟨cos²θ⟩ - 1)/2."""
    normals = np.array(normals)
    N = len(normals)
    
    # Method 1: Direct calculation using S = (3⟨cos²θ⟩ - 1)/2
    # First estimate director as average orientation
    director_estimate = np.mean(normals, axis=0)
    director_estimate /= np.linalg.norm(director_estimate)
    
    # Calculate cos²θ for each molecule
    cos_theta_sq = []
    for n in normals:
        cos_theta = abs(np.dot(n, director_estimate))  # abs for nematic symmetry
        cos_theta_sq.append(cos_theta**2)
    
    # Calculate S = (3⟨cos²θ⟩ - 1)/2
    avg_cos_theta_sq = np.mean(cos_theta_sq)
    S_direct = (3 * avg_cos_theta_sq - 1) / 2
    
    # Method 2: Q-tensor approach for comparison
    Q = np.zeros((3, 3))
    for n in normals:
        Q += 1.5 * np.outer(n, n) - 0.5 * np.eye(3)
    Q /= N
    
    eigenvals, eigenvecs = np.linalg.eigh(Q)
    max_idx = np.argmax(eigenvals)
    S_tensor = eigenvals[max_idx]
    director = eigenvecs[:, max_idx]
    
    print(f"S (direct formula): {S_direct:.4f}")
    print(f"S (Q-tensor): {S_tensor:.4f}")
    
    return S_tensor, director

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Calculate nematic order parameter S")
    parser.add_argument("input", help="LAMMPS data file")
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        return
    
    # Read data
    coords, mol_ids = read_lammps_data(args.input)
    molecules = group_by_molecule(coords, mol_ids)
    
    print(f"Found {len(molecules)} platelets")
    
    # Calculate normal vectors
    normals = []
    for mol_coords in molecules.values():
        if len(mol_coords) >= 3:
            normal = get_platelet_normal(mol_coords)
            normals.append(normal)
    
    if len(normals) < 2:
        print("Need at least 2 platelets")
        return
    
    # Calculate nematic order
    S, director = calculate_nematic_s(normals)
    
    print(f"Nematic order parameter S = {S:.4f}")
    print(f"Director = [{director[0]:.4f}, {director[1]:.4f}, {director[2]:.4f}]")
    
    # Interpret result
    if S > 0.7:
        print("High order - well aligned")
    elif S > 0.3:
        print("Moderate order - partial alignment")
    else:
        print("Low order - random orientation")

if __name__ == "__main__":
    main()