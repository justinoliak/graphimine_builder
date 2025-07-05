# Rigid Body Molecular Dynamics Scripts

This folder contains all the scripts needed to run rigid body molecular dynamics simulations of graphimine platelets.

## Files

### Core Scripts
- **`LAMMPS_bead_builder.py`** - Generates hexagonal platelet structures with specified G and concentration
- **`production_traj_only.sh`** - SLURM production script for cluster runs (self-contained)
- **`graphimine_rigid_LJ.in`** - LAMMPS template for rigid body simulations (optional reference)

### Test Scripts
- **`test_rigid_single.sh`** - Test script to verify rigid body dynamics works locally

## Usage

### Local Testing
```bash
# Test single simulation
bash test_rigid_single.sh
```

### Cluster Production
```bash
# Copy files to cluster
scp LAMMPS_bead_builder.py production_traj_only.sh user@cluster:/scratch/path/

# On cluster
cd /scratch/path/
mkdir -p logs
sbatch production_traj_only.sh
```

## Features

### Rigid Body Dynamics
- **Perfect rigidity**: Platelets cannot deform internally
- **Better performance**: ~2.6x faster than harmonic bonds
- **Cleaner dynamics**: Each platelet moves/rotates as single object
- **Automatic scaling**: Communication cutoff adjusts for G5-G10

### Production Parameters
- **G values**: 5, 6, 7, 8, 9, 10
- **Concentrations**: φ = 0.01 to 0.56 (56 values)
- **Total jobs**: 336 (6 G × 56 φ × 1 replicate)
- **Simulation length**: 400,000 steps
- **Output frequency**: Every 1000 steps

### Output
- **Trajectories**: `G{G}/traj_phi{phi}_rep{rep}.lammpstrj`
- **Organized by G**: Separate folders for each generation
- **Analysis ready**: Compatible with nematic order analysis scripts

## Technical Details

### Communication Cutoff
Automatically calculated as `(2*G + 1) * 0.6` to handle increasing platelet sizes:
- G5: 6.6σ
- G6: 7.8σ  
- G7: 9.0σ
- G8: 10.2σ
- G9: 11.4σ
- G10: 12.6σ

### Rigid Body Setup
- **Integration**: `fix rigid/small molecule langevin`
- **Thermostat**: Langevin with γ=1.0, T=1.0
- **Timestep**: 0.005τ (larger than harmonic bonds)
- **Potential**: WCA (purely repulsive)

### Performance
- **Speed**: ~285k τ/day for G5 φ=0.1
- **Memory**: ~50 MB per simulation
- **Scaling**: Handles G5-G10 without issues