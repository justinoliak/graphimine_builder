# Large N Graphimine Study

Self-contained study of large graphimine platelets (N144, N256) at high volume fractions.

## Files

- `parallel_builder.py` - Discotic structure generator (parallel layers)
- `run_bulk.in` - LAMMPS simulation template  
- `large_N_study.yml` - Study configuration
- `run_large_N_production.py` - Main production script
- `nematic_order.py` - Advanced 3D nematic order calculator

## Study Parameters

- **Platelet sizes**: N144 (50 flakes), N256 (50 flakes)
- **Volume fractions**: φ = 0.05 to 0.65 (13 values)
- **Replicates**: 3 per φ value
- **Total simulations**: 78

## Usage

```bash
python run_large_N_production.py
```

This will:
1. Generate all data files using the parallel discotic builder
2. Run LAMMPS simulations for all configurations  
3. Save trajectories and logs

The parallel builder creates initially aligned platelets in regular layers, avoiding random packing algorithms that struggle at high densities.

## Analysis

```bash
python nematic_order.py trajectories/traj_phi0.50_N144_rep1.lammpstrj --geometry disc --skip 1000 --every 10
```