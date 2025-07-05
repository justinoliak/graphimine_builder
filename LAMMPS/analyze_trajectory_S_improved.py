#!/usr/bin/env python3
"""Calculate S(t) from LAMMPS trajectory file using cross product method."""

import numpy as np
import gzip
import sys
from pathlib import Path

def read_trajectory_frame(f):
    """Read one frame from LAMMPS trajectory file."""
    # Read timestep
    line = f.readline()
    if not line:
        return None, None
    
    if 'ITEM: TIMESTEP' not in line:
        return None, None
    
    timestep = int(f.readline().strip())
    
    # Read number of atoms
    f.readline()  # ITEM: NUMBER OF ATOMS
    n_atoms = int(f.readline().strip())
    
    # Read box bounds
    f.readline()  # ITEM: BOX BOUNDS
    for _ in range(3):
        f.readline()
    
    # Read atoms
    f.readline()  # ITEM: ATOMS
    atoms = []
    mol_ids = []
    for _ in range(n_atoms):
        parts = f.readline().split()
        mol_ids.append(int(parts[1]))
        atoms.append([float(parts[3]), float(parts[4]), float(parts[5])])
    
    return timestep, {'coords': np.array(atoms), 'mol_ids': mol_ids}

def get_platelet_normal(coords):
    """Calculate platelet normal vector using cross product of two edges."""
    # For hexagonal platelets, use cross product of two edge vectors
    center = np.mean(coords, axis=0)
    
    # Find two points farthest from center in different directions
    distances = np.linalg.norm(coords - center, axis=1)
    max_idx = np.argmax(distances)
    vec1 = coords[max_idx] - center
    
    # Find another point roughly perpendicular to first
    dots = np.abs(np.dot(coords - center, vec1) / (np.linalg.norm(coords - center, axis=1) * np.linalg.norm(vec1)))
    min_dot_idx = np.argmin(dots)
    vec2 = coords[min_dot_idx] - center
    
    # Normal is cross product
    normal = np.cross(vec1, vec2)
    norm = np.linalg.norm(normal)
    
    if norm > 1e-10:
        return normal / norm
    else:
        # Fallback: use z-axis if cross product fails
        return np.array([0.0, 0.0, 1.0])

def calculate_S_from_frame(coords, mol_ids):
    """Calculate S for one frame using Q-tensor method."""
    # Group by molecule
    molecules = {}
    for i, mol_id in enumerate(mol_ids):
        if mol_id not in molecules:
            molecules[mol_id] = []
        molecules[mol_id].append(coords[i])
    
    # Get normals
    normals = []
    for mol_id, mol_coords in molecules.items():
        mol_coords = np.array(mol_coords)
        if len(mol_coords) >= 3:
            normal = get_platelet_normal(mol_coords)
            normals.append(normal)
    
    if len(normals) < 2:
        return 0.0
    
    # Calculate Q-tensor
    normals = np.array(normals)
    Q = np.zeros((3, 3))
    for n in normals:
        Q += 1.5 * np.outer(n, n) - 0.5 * np.eye(3)
    Q /= len(normals)
    
    # Get S (largest eigenvalue)
    eigenvals = np.linalg.eigvalsh(Q)
    return np.max(eigenvals)

def analyze_trajectory_silent(filename, sample_interval=1):
    """Analyze trajectory file without creating individual output files."""
    # Determine if gzipped
    if filename.endswith('.gz'):
        f = gzip.open(filename, 'rt')
    else:
        f = open(filename, 'r')
    
    timesteps = []
    S_values = []
    
    frame_count = 0
    while True:
        timestep, frame_data = read_trajectory_frame(f)
        if timestep is None:
            break
        
        # Sample every nth frame to speed up analysis
        if frame_count % sample_interval == 0:
            # Calculate S for this frame
            S = calculate_S_from_frame(frame_data['coords'], frame_data['mol_ids'])
            timesteps.append(timestep)
            S_values.append(S)
        
        frame_count += 1
    
    f.close()
    return timesteps, S_values

def analyze_trajectory(filename, sample_interval=1):
    """Analyze entire trajectory file."""
    print(f"Analyzing {filename}")
    
    # Determine if gzipped
    if filename.endswith('.gz'):
        f = gzip.open(filename, 'rt')
    else:
        f = open(filename, 'r')
    
    timesteps = []
    S_values = []
    
    frame_count = 0
    while True:
        timestep, frame_data = read_trajectory_frame(f)
        if timestep is None:
            break
        
        # Sample every nth frame to speed up analysis
        if frame_count % sample_interval == 0:
            # Calculate S for this frame
            S = calculate_S_from_frame(frame_data['coords'], frame_data['mol_ids'])
            timesteps.append(timestep)
            S_values.append(S)
            
            if len(S_values) % 10 == 0:
                print(f"  Frame {len(S_values)}: t={timestep}, S={S:.4f}")
        
        frame_count += 1
    
    f.close()
    
    if len(S_values) == 0:
        print("No frames analyzed!")
        return [], []
    
    # Calculate statistics
    S_values = np.array(S_values)
    equilibration = len(S_values) // 2  # Use second half as equilibrated
    S_eq = S_values[equilibration:]
    
    print(f"\nAnalyzed {len(S_values)} frames (every {sample_interval} frames)")
    print(f"Full trajectory: S = {np.mean(S_values):.4f} ± {np.std(S_values):.4f}")
    if len(S_eq) > 0:
        print(f"Equilibrated (last half): S = {np.mean(S_eq):.4f} ± {np.std(S_eq):.4f}")
    print(f"Final S = {S_values[-1]:.4f}")
    
    # Extract phi value from filename for output
    filename_path = Path(filename)
    if 'phi' in filename_path.name:
        # Extract phi value from filename like traj_phi0.10_rep1.lammpstrj
        phi_part = [part for part in filename_path.name.split('_') if part.startswith('phi')][0]
        phi_value = phi_part[3:]  # Remove 'phi' prefix
        output_base = f"S_vs_time_phi{phi_value}"
    else:
        output_base = filename_path.stem + "_S_vs_time"
    
    # Save time series
    output_file = filename_path.parent / f"{output_base}.dat"
    np.savetxt(output_file, np.column_stack([timesteps, S_values]), 
               header='timestep S', comments='')
    print(f"Saved time series to {output_file}")
    
    return timesteps, S_values

def batch_analyze_G_folder(G_folder, sample_interval=1):
    """Analyze all trajectory files in a G folder and create comprehensive CSV."""
    G_path = Path(G_folder)
    
    if not G_path.exists():
        print(f"Folder {G_folder} does not exist!")
        return
    
    # Find all trajectory files (both compressed and uncompressed)
    traj_files = list(G_path.glob("traj_phi*.lammpstrj*"))
    
    if not traj_files:
        print(f"No trajectory files found in {G_folder}")
        return
    
    print(f"Found {len(traj_files)} trajectory files in {G_folder}")
    
    # Extract G value from folder name (handles both G5 and /scratch/.../G5/)
    G_value = G_path.name.replace('G', '') if 'G' in G_path.name else 'unknown'
    if G_value == 'unknown':
        # Try to extract from parent path
        for part in G_path.parts:
            if part.startswith('G') and part[1:].isdigit():
                G_value = part[1:]
                break
    
    # Prepare comprehensive data storage
    all_data = []
    
    # Analyze each trajectory
    for traj_file in sorted(traj_files):
        print(f"\n{'='*60}")
        print(f"Analyzing {traj_file.name}")
        
        # Analyze trajectory (suppress individual output files in batch mode)
        timesteps, S_values = analyze_trajectory_silent(str(traj_file), sample_interval)
        
        # Extract phi value
        if 'phi' in traj_file.name:
            phi_part = [part for part in traj_file.name.split('_') if part.startswith('phi')][0]
            phi_value = float(phi_part[3:])
        else:
            print(f"Warning: Could not extract phi from {traj_file.name}")
            continue
        
        if len(S_values) == 0:
            print(f"Warning: No S values calculated for {traj_file.name}")
            continue
        
        # Add each timestep to the comprehensive dataset
        for t, S in zip(timesteps, S_values):
            all_data.append({
                'G': G_value,
                'phi': phi_value,
                'timestep': t,
                'S': S
            })
        
        # Calculate and print summary for this trajectory
        equilibration = len(S_values) // 2
        S_eq = S_values[equilibration:] if equilibration < len(S_values) else S_values
        print(f"  phi={phi_value:.2f}: {len(S_values)} frames, S_eq = {np.mean(S_eq):.4f} ± {np.std(S_eq):.4f}")
    
    # Save comprehensive CSV
    if all_data:
        import pandas as pd
        
        df = pd.DataFrame(all_data)
        csv_file = G_path / f"nematic_S_data_G{G_value}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"Comprehensive CSV saved to {csv_file}")
        print(f"Total data points: {len(all_data)}")
        print(f"Concentrations: {sorted(df['phi'].unique())}")
        
        # Also save summary statistics
        summary_data = []
        for phi in sorted(df['phi'].unique()):
            phi_data = df[df['phi'] == phi]['S']
            equilibration = len(phi_data) // 2
            S_eq = phi_data.iloc[equilibration:] if equilibration < len(phi_data) else phi_data
            
            summary_data.append({
                'G': G_value,
                'phi': phi,
                'S_mean_eq': S_eq.mean(),
                'S_std_eq': S_eq.std(),
                'S_final': phi_data.iloc[-1],
                'n_frames': len(phi_data)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = G_path / f"nematic_S_summary_G{G_value}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Summary statistics saved to {summary_file}")
        print("\nEquilibrated S values:")
        for _, row in summary_df.iterrows():
            print(f"  phi={row['phi']:.2f}: S = {row['S_mean_eq']:.4f} ± {row['S_std_eq']:.4f}")
    
    else:
        print("No data to save!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python analyze_trajectory_S_improved.py trajectory_file.lammpstrj")
        print("  Batch mode:  python analyze_trajectory_S_improved.py /path/to/G5/folder/ [sample_interval]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    sample_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    if Path(input_path).is_dir():
        # Batch mode
        batch_analyze_G_folder(input_path, sample_interval)
    else:
        # Single file mode
        analyze_trajectory(input_path, sample_interval)