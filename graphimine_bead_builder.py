#!/usr/bin/env python3
"""Hexagonal lattice generator for LAMMPS with volume fraction control."""

# Standard library imports
import argparse
import math
import random
from pathlib import Path
from typing import List, Tuple

# Third-party imports
import numpy as np
from scipy.spatial import cKDTree as KDTree

# Constants
SIGMA = 1.0  # Bead diameter
Z_SPACING_DEFAULT = 1.06  # Default z-layer spacing
BOX_SIZE_MULTIPLIER = 5.0  # Box size multiplier for platelet diameter


# =============================================================================
# GEOMETRY AND LATTICE GENERATION FUNCTIONS
# =============================================================================

def calculate_box_size(G: int, b: float = 1.0) -> Tuple[float, float]:
    """Calculate box size and platelet diameter for generation G."""
    platelet_diameter = (2 * G + 1) * b  # 2*G + 1 center monomer
    box_size = BOX_SIZE_MULTIPLIER * platelet_diameter
    return box_size, platelet_diameter

def generate_hexagonal_lattice_xy(box_size: float, spacing: float) -> List[Tuple[float, float]]:
    """Generate 2D hexagonal lattice points."""
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
    """Generate z-coordinates for evenly spaced layers."""
    n_layers = int(box_size / z_spacing)
    if n_layers % 2 == 0:
        n_layers -= 1  # Ensure odd number for symmetric placement
    
    z_coords = []
    for i in range(n_layers):
        z = -box_size / 2 + (i + 0.5) * box_size / n_layers
        z_coords.append(z)
    
    return z_coords

def generate_all_lattice_centers(box_size: float, xy_spacing: float, z_spacing: float) -> List[np.ndarray]:
    """Generate all 3D lattice center points for platelet placement."""
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


# =============================================================================
# HEXAGON AND STRUCTURE BUILDING FUNCTIONS
# =============================================================================

def generate_hex_flake(g: int, b: float) -> np.ndarray:
    """Generate a single hexagonal flake with specified generations."""
    coords = []
    for n in range(-g, g + 1):
        for m in range(-g, g + 1):
            if max(abs(n), abs(m), abs(-n - m)) <= g:
                x = b * (m + 0.5 * n)
                y = b * (math.sqrt(3) / 2 * n)
                coords.append([x, y, 0.0])
    return np.asarray(coords)

def bond_pairs(centres: np.ndarray, cut: float) -> List[Tuple[int, int]]:
    """Find all pairs of beads within bonding distance."""
    kd = KDTree(centres)
    return [tuple(sorted(p)) for p in kd.query_pairs(cut)]

def build_hexagons_at_centers(centers: List[np.ndarray], G: int, bond_length: float = 1.0) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[int]]:
    """Build hexagonal platelets at specified 3D positions."""
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


# =============================================================================
# CALCULATION AND ANALYSIS FUNCTIONS
# =============================================================================

def calculate_platelets_for_initial_box(G: int, concentration: float, initial_box_size: float) -> Tuple[int, int]:
    """Calculate number of platelets needed for target concentration in initial box size."""
    # Calculate platelet properties
    beads_per_platelet = 3 * G**2 + 3 * G + 1  # Number of beads per platelet
    bead_volume = math.pi * SIGMA**3 / 6  # Volume of single bead
    platelet_volume = beads_per_platelet * bead_volume  # Volume per platelet
    
    # Calculate box volume and target volume fraction
    box_volume = initial_box_size**3
    target_solid_volume = concentration * box_volume
    
    # Calculate number of platelets needed (rounded UP)
    n_platelets = target_solid_volume / platelet_volume
    n_platelets_rounded_up = math.ceil(n_platelets)  # Round UP to ensure we meet/exceed target
    
    return n_platelets_rounded_up, beads_per_platelet

def calculate_final_box_size(G: int, concentration: float, n_platelets: int, beads_per_platelet: int) -> Tuple[float, float]:
    """Calculate final box size needed to achieve exact concentration with given number of platelets."""
    # Calculate volumes
    bead_volume = math.pi * SIGMA**3 / 6  # Volume of single bead
    platelet_volume = beads_per_platelet * bead_volume  # Volume per platelet
    total_solid_volume = n_platelets * platelet_volume
    
    # Calculate required box volume for exact concentration
    required_box_volume = total_solid_volume / concentration
    final_box_size = required_box_volume**(1/3)  # Cube root for cubic box
    
    # Calculate actual concentration achieved
    actual_concentration = total_solid_volume / (final_box_size**3)
    
    return final_box_size, actual_concentration

def calculate_optimal_spacing(platelet_diameter: float, clearance: float = 0.06) -> Tuple[float, float]:
    """Calculate optimal lattice spacing for platelets."""
    # Use short diameter (face-to-face) instead of long diameter (vertex-to-vertex)
    short_diameter = platelet_diameter * math.sqrt(3) / 2
    xy_spacing = short_diameter + clearance
    return short_diameter, xy_spacing


# =============================================================================
# I/O AND UTILITY FUNCTIONS
# =============================================================================

def write_lammps_hexagons(path: Path, coords: List[np.ndarray], bonds: List[Tuple[int, int]], 
                         box_size: float, flake_sizes: List[int]) -> None:
    """Write LAMMPS data file for hexagonal platelet structures."""
    with path.open("w") as f:
        f.write(f"Hexagonal Platelet Structures – {len(coords)} atoms, {len(bonds)} bonds\n\n")
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

def parse_cli() -> argparse.Namespace:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Generate hexagonal lattice with concentration control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--G", type=int, required=True, 
                    help="Generation number for platelet size (0=1 bead, 1=7 beads, etc.)")
    ap.add_argument("--concentration", type=float, required=True, 
                    help="Target volume fraction (0-1)")
    ap.add_argument("--output", type=str, default=None, 
                    help="Output LAMMPS data filename (default: auto-generated from G and concentration)")
    return ap.parse_args()

def get_script_dir() -> Path:
    """Get the directory where this script is located."""
    return Path(__file__).parent.absolute()

def print_simulation_info(G: int, concentration: float, box_size: float, 
                         platelet_diameter: float, beads_per_platelet: int,
                         n_platelets: int, actual_concentration: float) -> None:
    """Print comprehensive simulation setup information."""
    print(f"[info] G={G}, Target concentration: {concentration:.4f}")
    print(f"[info] Platelet diameter: {platelet_diameter:.2f}σ, Beads per platelet: {beads_per_platelet}")
    print(f"[info] Box size: {box_size:.2f}σ")
    print(f"[info] Target platelets: {n_platelets}, Actual concentration: {actual_concentration:.4f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution function."""
    # Parse command line arguments
    args = parse_cli()
    
    # Validate input parameters
    if not (0 < args.concentration <= 1):
        raise ValueError("Concentration must be between 0 and 1")
    if args.G < 0:
        raise ValueError("Generation number G must be non-negative")
    
    # Calculate box dimensions and platelet properties
    initial_box_size, platelet_diameter = calculate_box_size(args.G)
    
    # Calculate number of platelets needed (rounded up)
    n_platelets, beads_per_platelet = calculate_platelets_for_initial_box(
        args.G, args.concentration, initial_box_size)
    
    # Calculate final box size to achieve exact concentration
    box_size, actual_concentration = calculate_final_box_size(
        args.G, args.concentration, n_platelets, beads_per_platelet)
    
    # Print simulation setup information
    print_simulation_info(args.G, args.concentration, box_size, 
                         platelet_diameter, beads_per_platelet, 
                         n_platelets, actual_concentration)
    
    # Calculate optimal lattice spacing
    short_diameter, xy_spacing = calculate_optimal_spacing(platelet_diameter)
    z_spacing = Z_SPACING_DEFAULT
    
    print(f"[info] Short diameter: {short_diameter:.2f}σ, XY spacing: {xy_spacing:.3f}σ, Z spacing: {z_spacing:.3f}σ")
    
    # Generate all possible lattice centers
    all_hex_centers = generate_all_lattice_centers(box_size, xy_spacing, z_spacing)
    
    # Select appropriate number of centers
    if len(all_hex_centers) < n_platelets:
        print(f"[warning] Only {len(all_hex_centers)} lattice sites available, but {n_platelets} platelets needed")
        print(f"[warning] Using all available sites. Actual concentration will be lower.")
        hex_centers = all_hex_centers
        n_platelets = len(all_hex_centers)
        # Recalculate actual concentration
        actual_solid_volume = n_platelets * beads_per_platelet * (math.pi * SIGMA**3 / 6)
        actual_concentration = actual_solid_volume / (box_size**3)
    else:
        # Randomly select the required number of centers
        hex_centers = random.sample(all_hex_centers, n_platelets)
        print(f"[info] Selected {n_platelets} centers from {len(all_hex_centers)} available lattice sites")
    
    # Build hexagonal platelets at selected centers
    coords, bonds, flake_sizes = build_hexagons_at_centers(hex_centers, args.G, bond_length=1.0)
    
    # Generate output filename if not provided
    if args.output is None:
        output_filename = f"hexlattice_G{args.G}_phi{args.concentration:.3f}.data"
    else:
        output_filename = args.output
    
    # Write LAMMPS data file
    script_dir = get_script_dir()
    output_path = script_dir / output_filename
    write_lammps_hexagons(output_path, coords, bonds, box_size, flake_sizes)
    
    # Print final statistics
    print(f"[done] Wrote {output_path} — {len(coords)} atoms, {len(bonds)} bonds")
    print(f"[info] {len(hex_centers)} platelets, {len(coords)//len(hex_centers)} atoms each")
    print(f"[info] Final volume fraction φ = {actual_concentration:.4f}")
    print(f"[info] Concentration error: {abs(actual_concentration - args.concentration)/args.concentration*100:.2f}%")


if __name__ == "__main__":
    main()