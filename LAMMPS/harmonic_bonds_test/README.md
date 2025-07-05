# Strong Harmonic Bonds Molecular Dynamics Scripts

This folder contains all the scripts needed to run strong harmonic bonds molecular dynamics simulations of graphimine platelets.

## Files

### Core Scripts
- **`LAMMPS_bead_builder.py`** - Generates hexagonal platelet structures with specified G and concentration
- **`production_harmonic_bonds.sh`** - SLURM production script for cluster runs (self-contained)
- **`graphimine_LJ.in`** - LAMMPS template for harmonic bond simulations (reference)

### Test Scripts
- **`test_harmonic_single.sh`** - Test script to verify strong harmonic bonds work locally

## Usage

### Local Testing
```bash
# Test single simulation
bash test_harmonic_single.sh
```

### Cluster Production
```bash
# Copy files to cluster
scp LAMMPS_bead_builder.py production_harmonic_bonds.sh user@cluster:/scratch/path/

# On cluster
cd /scratch/path/
mkdir -p logs
sbatch production_harmonic_bonds.sh
```

## Features

### Strong Harmonic Bonds
- **Near rigidity**: Very stiff bonds (k=4225) prevent most deformation
- **Standard MD**: Uses conventional Langevin + NVE integration
- **Compatibility**: Works with all analysis tools
- **Flexibility**: Allows small deformations under extreme forces

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

### Bond Parameters
- **Force constant**: k = 4225 ε/σ² (C=N imine bond strength)
- **Equilibrium length**: r₀ = 1.0σ
- **Physical basis**: Derived from AMBER/CHARMM force fields

### Integration Setup
- **Thermostat**: Langevin with γ=1.0, T=1.0
- **Integration**: NVE (velocity Verlet)
- **Timestep**: 0.002τ (smaller than rigid bodies)
- **Potential**: WCA (purely repulsive)

### Communication
- **Fixed cutoff**: 3.0σ (sufficient for all G values with bonds)
- **Memory**: Similar to rigid bodies (~50 MB per simulation)
- **Performance**: ~110k τ/day for G5 φ=0.1

## Comparison with Rigid Bodies

### Advantages
- **Standard MD**: Uses familiar integration methods
- **Tool compatibility**: Works with all MD analysis tools
- **Gradual failure**: Extreme forces cause stretching, not breaking

### Disadvantages
- **2.6x slower**: More force calculations due to bonds
- **Near rigidity**: Small deformations still possible
- **Energy overhead**: Bond potential energy always present

### When to Use
- **Analysis validation**: Compare with rigid body results
- **Tool compatibility**: When rigid body support is limited
- **Physical realism**: When small bond flexibility is desired