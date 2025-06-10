#!/usr/bin/env python3

import sys
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

def count_imine_and_aldehyde_accurate_fast(atoms, coords):
    """Accurate and fast count using vectorized operations."""
    
    atoms = np.array(atoms)
    coords = np.array(coords)
    
    # Get indices for different atom types
    c_indices = np.where(atoms == 'C')[0]
    n_indices = np.where(atoms == 'N')[0] 
    o_indices = np.where(atoms == 'O')[0]
    h_indices = np.where(atoms == 'H')[0]
    
    print(f"Found {len(c_indices)} C, {len(n_indices)} N, {len(o_indices)} O, {len(h_indices)} H atoms")
    
    # Bond distance thresholds
    cn_imine_threshold = 1.4  # C=N double bond
    co_double_threshold = 1.25  # C=O double bond  
    ch_threshold = 1.15  # C-H bond
    cc_threshold = 1.6  # C-C bond
    
    imine_count = 0
    aldehyde_count = 0
    
    print("Analyzing carbon atoms...")
    
    # Process carbons in batches for progress
    batch_size = 1000
    for batch_start in range(0, len(c_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(c_indices))
        batch_c_indices = c_indices[batch_start:batch_end]
        
        if batch_start % 10000 == 0:
            print(f"Processing carbons {batch_start}-{batch_end} of {len(c_indices)}")
        
        for c_idx in batch_c_indices:
            c_coord = coords[c_idx]
            
            # Count neighbors using vectorized operations
            neighbor_counts = {'C': 0, 'N': 0, 'O': 0, 'H': 0}
            
            # Check C neighbors
            if len(c_indices) > 1:
                c_coords = coords[c_indices]
                c_distances = np.linalg.norm(c_coords - c_coord, axis=1)
                c_neighbors = np.sum((c_distances < cc_threshold) & (c_distances > 0))
                neighbor_counts['C'] = c_neighbors
            
            # Check N neighbors
            if len(n_indices) > 0:
                n_coords = coords[n_indices]
                n_distances = np.linalg.norm(n_coords - c_coord, axis=1)
                n_neighbors = np.sum(n_distances < cn_imine_threshold)
                neighbor_counts['N'] = n_neighbors
            
            # Check O neighbors
            if len(o_indices) > 0:
                o_coords = coords[o_indices]
                o_distances = np.linalg.norm(o_coords - c_coord, axis=1)
                o_neighbors = np.sum(o_distances < co_double_threshold)
                neighbor_counts['O'] = o_neighbors
            
            # Check H neighbors
            if len(h_indices) > 0:
                h_coords = coords[h_indices]
                h_distances = np.linalg.norm(h_coords - c_coord, axis=1)
                h_neighbors = np.sum(h_distances < ch_threshold)
                neighbor_counts['H'] = h_neighbors
            
            # Identify imine carbons: 1 N neighbor, no O, reasonable total connectivity
            if (neighbor_counts['N'] == 1 and 
                neighbor_counts['O'] == 0 and 
                neighbor_counts['C'] >= 1 and neighbor_counts['C'] <= 2):
                imine_count += 1
            
            # Identify aldehyde carbons: 1 O, 1 H, 1 C, 0 N
            elif (neighbor_counts['O'] == 1 and 
                  neighbor_counts['H'] == 1 and 
                  neighbor_counts['C'] == 1 and 
                  neighbor_counts['N'] == 0):
                aldehyde_count += 1
    
    return imine_count, aldehyde_count

def main():
    if len(sys.argv) != 2:
        print("Usage: python Generate+imine_ald_counts.py <xyz_file>")
        sys.exit(1)
    
    xyz_file = sys.argv[1]
    
    try:
        atoms, coords, comment = read_xyz(xyz_file)
        print(f"Read {len(atoms)} atoms from {xyz_file}")
        
        imine_count, aldehyde_count = count_imine_and_aldehyde_accurate_fast(atoms, coords)
        
        print(f"Imine groups (C=N): {imine_count}")
        print(f"Aldehyde groups (CHO): {aldehyde_count}")
        print(f"Total functional groups: {imine_count + aldehyde_count}")
        
        return imine_count, aldehyde_count
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()