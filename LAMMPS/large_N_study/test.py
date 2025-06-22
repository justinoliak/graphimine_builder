#!/usr/bin/env python3
"""
Test Script: Hexagonal Lattice Generator
========================================
Creates hexagonal lattice of beads in xy-planes with z-spacing of 1.06σ
Box size = 5 × (2G + 1) for input generation G
"""
import math
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple
from scipy.spatial import cKDTree as KDTree

SIGMA = 1.0

def calculate_box_size(G: int, b: float = 1.0) -> float:
    """Calculate box size based on G generations"""
    platelet_diameter = (2 * G + 1) * b  # 2*G + 1 center monomer
    box_size = 5.0 * platelet_diameter
    return box_size, platelet_diameter

def generate_hexagonal_lattice_xy(box_size: float, spacing: float) -> List[Tuple[float, float]]:
    """Generate hexagonal lattice points in xy-plane with different orientation"""
    points = []
    
    # Hexagonal lattice vectors - rotated orientation
    # a1 points along y-axis, a2 at 60 degrees to a1
    a1 = np.array([0, spacing])
    a2 = np.array([spacing * math.sqrt(3) / 2, spacing * 0.5])
    
    # Determine grid bounds
    max_n = int(box_size / spacing) + 1
    
    for n in range(-max_n, max_n + 1):
        for m in range(-max_n, max_n + 1):
            point = n * a1 + m * a2
            x, y = point[0], point[1]
            
            # Check if point is within box bounds
            if abs(x) <= box_size / 2 and abs(y) <= box_size / 2:
                points.append((x, y))
    
    return points

def generate_z_layers(box_size: float, z_spacing: float) -> List[float]:
    """Generate z-coordinates for layers"""
    n_layers = int(box_size / z_spacing)
    if n_layers % 2 == 0:
        n_layers -= 1  # Ensure odd number for symmetric placement
    
    z_coords = []
    for i in range(n_layers):
        z = -box_size / 2 + (i + 0.5) * box_size / n_layers
        z_coords.append(z)
    
    return z_coords

def generate_hex_flake(g: int, b: float) -> np.ndarray:
    """Generate hexagonal flake with g generations"""
    coords = []
    for n in range(-g, g + 1):
        for m in range(-g, g + 1):
            if max(abs(n), abs(m), abs(-n - m)) <= g:
                x = b * (m + 0.5 * n)
                y = b * (math.sqrt(3) / 2 * n)
                coords.append([x, y, 0.0])
    return np.asarray(coords)

def bond_pairs(centres: np.ndarray, cut: float) -> List[Tuple[int, int]]:
    """Find bond pairs within cutoff distance"""
    kd = KDTree(centres)
    return [tuple(sorted(p)) for p in kd.query_pairs(cut)]

def generate_all_lattice_centers(box_size: float, xy_spacing: float, z_spacing: float) -> List[np.ndarray]:
    """Generate all 3D lattice center points"""
    # Generate xy hexagonal lattice
    xy_points = generate_hexagonal_lattice_xy(box_size, xy_spacing)
    print(f"[info] XY hex lattice: {len(xy_points)} centers, spacing={xy_spacing:.3f}σ")
    
    # Generate z layers
    z_layers = generate_z_layers(box_size, z_spacing)
    print(f"[info] Z layers: {len(z_layers)} layers, spacing={z_spacing:.3f}σ")
    
    # Combine to get all 3D centers
    all_centers = []
    for z in z_layers:
        for x, y in xy_points:
            center = np.array([x, y, z])
            all_centers.append(center)
    
    print(f"[info] Total hexagon centers: {len(all_centers)}")
    return all_centers

def rotate_45_degrees(coords: np.ndarray) -> np.ndarray:
    """Rotate coordinates 45 degrees around z-axis"""
    angle = np.pi / 4  # 45 degrees in radians
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    # 2D rotation matrix for xy plane
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # Apply rotation to each coordinate
    rotated = np.dot(coords, rotation_matrix.T)
    return rotated

def build_hexagons_at_centers(centers: List[np.ndarray], G: int, bond_length: float = 1.0) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[int]]:
    """Build hexagonal flakes at each center"""
    # Generate template hexagon
    hex_template = generate_hex_flake(G, bond_length)
    print(f"[info] Hexagon template: G={G}, {len(hex_template)} beads per flake")
    
    all_coords = []
    all_bonds = []
    flake_sizes = []
    offset = 0
    
    for center in centers:
        # Position hexagon at center
        positioned_hex = hex_template + center
        
        # Find bonds within this hexagon
        hex_bonds = bond_pairs(positioned_hex, 1.05 * bond_length)
        offset_bonds = [(i + offset, j + offset) for i, j in hex_bonds]
        
        # Add to global lists
        all_coords.extend(positioned_hex.tolist())
        all_bonds.extend(offset_bonds)
        flake_sizes.append(len(positioned_hex))
        offset += len(positioned_hex)
    
    print(f"[info] Built {len(centers)} hexagons")
    print(f"[info] Total atoms: {len(all_coords)}, Total bonds: {len(all_bonds)}")
    
    return all_coords, all_bonds, flake_sizes

def write_lammps_hexagons(path: Path, coords: List[np.ndarray], bonds: List[Tuple[int, int]], 
                         box_size: float, flake_sizes: List[int]):
    """Write LAMMPS data file for hexagonal structures"""
    with path.open("w") as f:
        f.write(f"Hexagonal Structures – {len(coords)} atoms, {len(bonds)} bonds\n\n")
        f.write(f"{len(coords)} atoms\n{len(bonds)} bonds\n\n")
        f.write("1 atom types\n1 bond types\n\n")
        f.write(f"{-box_size/2:.6f} {box_size/2:.6f} xlo xhi\n")
        f.write(f"{-box_size/2:.6f} {box_size/2:.6f} ylo yhi\n")
        f.write(f"{-box_size/2:.6f} {box_size/2:.6f} zlo zhi\n\n")
        f.write("Masses\n\n1 1.0\n\n")
        f.write("Atoms  # id mol type x y z\n\n")
        
        atom_id, mol_id, start = 1, 1, 0
        for flake_size in flake_sizes:
            for i in range(flake_size):
                x, y, z = coords[start + i]
                f.write(f"{atom_id} {mol_id} 1 {x:.6f} {y:.6f} {z:.6f}\n")
                atom_id += 1
            mol_id += 1
            start += flake_size
        
        f.write("\nBonds\n\n")
        for b_id, (i, j) in enumerate(bonds, 1):
            f.write(f"{b_id} 1 {i + 1} {j + 1}\n")

def parse_cli():
    """Parse command line arguments"""
    ap = argparse.ArgumentParser(description="Generate hexagonal lattice (test script)")
    ap.add_argument("--G", type=int, required=True, help="generation number for box size calculation")
    ap.add_argument("--output", type=str, default="test_lattice.data", help="output filename")
    return ap.parse_args()

if __name__ == "__main__":
    P = parse_cli()
    
    # Calculate box dimensions
    box_size, platelet_diameter = calculate_box_size(P.G)
    z_spacing = 1.06  # Fixed z-spacing
    
    # Use short diameter (face-to-face) instead of long diameter (vertex-to-vertex)
    short_diameter = platelet_diameter * math.sqrt(3) / 2
    xy_spacing = short_diameter + 0.06  # Short diameter + 0.06σ clearance
    
    print(f"[info] G={P.G}")
    print(f"[info] Platelet diameter: {platelet_diameter:.2f}σ")
    print(f"[info] Box size: {box_size:.2f}σ")
    print(f"[info] Short diameter: {short_diameter:.2f}σ, XY spacing: {xy_spacing:.3f}σ, Z spacing: {z_spacing:.3f}σ")
    
    # Generate hexagon centers
    hex_centers = generate_all_lattice_centers(box_size, xy_spacing, z_spacing)
    
    # Build hexagons at each center
    coords, bonds, flake_sizes = build_hexagons_at_centers(hex_centers, P.G, bond_length=1.0)
    
    # Write LAMMPS file
    write_lammps_hexagons(Path(P.output), coords, bonds, box_size, flake_sizes)
    
    # Calculate density
    bead_vol = math.pi * SIGMA**3 / 6
    box_vol = box_size**3
    phi = len(coords) * bead_vol / box_vol
    
    print(f"[done] Wrote {P.output} — {len(coords)} atoms, {len(bonds)} bonds")
    print(f"[info] {len(hex_centers)} hexagons, {len(coords)//len(hex_centers)} atoms each")
    print(f"[info] Volume fraction φ = {phi:.4f}")