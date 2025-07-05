#!/bin/bash
#SBATCH --job-name=harmonic_sweep
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --array=0-335%50
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Load modules
module load LAMMPS/29Aug2024

# Change to scratch directory where scripts are located
cd /scratch/oliak.j/graphimine_temp

# Define parameters
G_VALUES=(5 6 7 8 9 10)
PHI_VALUES=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
            0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 \
            0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30 \
            0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.40 \
            0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.50 \
            0.51 0.52 0.53 0.54 0.55 0.56)

# Calculate indices
N_PHI=56
N_REP=1

G_IDX=$((SLURM_ARRAY_TASK_ID / (N_PHI * N_REP)))
PHI_IDX=$(((SLURM_ARRAY_TASK_ID / N_REP) % N_PHI))
REP=$((SLURM_ARRAY_TASK_ID % N_REP + 1))

G=${G_VALUES[$G_IDX]}
PHI=${PHI_VALUES[$PHI_IDX]}

# Create job directory
JOB_DIR="run_G${G}_phi${PHI}_rep${REP}"
mkdir -p $JOB_DIR

# Minimal output
echo "Job $SLURM_ARRAY_TASK_ID: G=$G, phi=$PHI, rep=$REP"

# Generate structure (suppress output)
python3 LAMMPS_bead_builder.py \
    --G $G \
    --concentration $PHI \
    --output ${JOB_DIR}/structure.data > /dev/null 2>&1

# Change to job directory
cd $JOB_DIR

# Verify structure file exists
if [ ! -f structure.data ]; then
    echo "ERROR: Structure generation failed"
    exit 1
fi

# Set parameters
OUTPUT_PREFIX="sim"
SEED=$((49281 + ${PHI_IDX} * 1000 + ${REP} * 10))

# Create LAMMPS input file with strong harmonic bonds
cat > lammps_input.in << EOFLAMMPS
# Strong harmonic bonds approach
units       lj
atom_style  molecular
boundary    p p p

log ${OUTPUT_PREFIX}.log

read_data   structure.data
mass        * 1.0

# Communication for large systems
comm_modify cutoff 3.0

# WCA potential
pair_style      lj/cut 1.12246
pair_modify     shift yes
pair_coeff      * * 1.0 1.0 1.12246

neighbor     1.0 bin
neigh_modify every 1 delay 0 check yes

# Strong harmonic bonds for rigidity
bond_style   harmonic
bond_coeff   * 4225.0 1.0

# Langevin thermostat + NVE integration
fix     brown all langevin 1.0 1.0 1.0 ${SEED}
fix     int   all nve
timestep 0.002

# Minimal thermo output
thermo_style custom step temp pe ke etotal press
thermo       10000

# Minimization
min_style    cg
minimize     1.0e-4 1.0e-6  5000 50000

reset_timestep 0

# Trajectory output
dump         traj all custom 1000 ${OUTPUT_PREFIX}.lammpstrj id mol type x y z
dump_modify  traj sort id

# Production run
run     400000
EOFLAMMPS

# Run LAMMPS with minimal screen output
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
lmp_serial -in lammps_input.in -screen none

# Check if LAMMPS succeeded by looking for trajectory file
if [ ! -f ${OUTPUT_PREFIX}.lammpstrj ]; then
    echo "ERROR: LAMMPS failed - no trajectory file"
    exit 1
fi

# Move results to organized structure
cd ..
mkdir -p G${G}

# Move trajectory only
mv ${JOB_DIR}/${OUTPUT_PREFIX}.lammpstrj G${G}/traj_phi${PHI}_rep${REP}.lammpstrj

# Clean up
rm -rf ${JOB_DIR}

echo "Completed: G=$G, phi=$PHI, rep=$REP"