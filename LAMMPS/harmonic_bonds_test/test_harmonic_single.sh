#!/bin/bash
# Test script for harmonic bonds approach - single job simulation

# Test parameters
G=5
PHI=0.1
REP=1
PHI_IDX=9

echo "Testing strong harmonic bonds with G=$G, phi=$PHI"

# Create test structure
echo "Generating structure..."
python3 LAMMPS_bead_builder.py --G $G --concentration $PHI --output test_structure.data

# Create job directory (mimicking cluster script behavior)
JOB_DIR="run_G${G}_phi${PHI}_rep${REP}"
mkdir -p $JOB_DIR
cp test_structure.data ${JOB_DIR}/structure.data
cd $JOB_DIR

# Set parameters (from production script)
OUTPUT_PREFIX="sim"
SEED=$((49281 + ${PHI_IDX} * 1000 + ${REP} * 10))

echo "Creating LAMMPS input with strong harmonic bonds..."

# Create LAMMPS input file with strong harmonic bonds (from production script)
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
thermo       1000

# Short minimization for test
min_style    cg
minimize     1.0e-4 1.0e-6  1000 5000

reset_timestep 0

# Trajectory output
dump         traj all custom 1000 ${OUTPUT_PREFIX}.lammpstrj id mol type x y z
dump_modify  traj sort id

# Short test run
run     5000

# Write final configuration
write_data  ${OUTPUT_PREFIX}_final.data
EOFLAMMPS

echo "Running LAMMPS simulation..."
if command -v lmp &> /dev/null; then
    lmp -in lammps_input.in
elif command -v lammps &> /dev/null; then
    lammps -in lammps_input.in
else
    echo "LAMMPS not found. Please install LAMMPS or ensure it's in your PATH."
    exit 1
fi

# Check results
echo ""
echo "Test Results:"
echo "============="
if [ -f ${OUTPUT_PREFIX}.lammpstrj ]; then
    echo "✓ Trajectory file created: ${OUTPUT_PREFIX}.lammpstrj"
    frames=$(grep -c "ITEM: TIMESTEP" ${OUTPUT_PREFIX}.lammpstrj)
    echo "  Number of frames: $frames"
else
    echo "✗ Trajectory file missing"
fi

if [ -f ${OUTPUT_PREFIX}_final.data ]; then
    echo "✓ Final configuration created: ${OUTPUT_PREFIX}_final.data"
    size=$(du -h ${OUTPUT_PREFIX}_final.data | cut -f1)
    echo "  File size: $size"
else
    echo "✗ Final configuration missing"
fi

if [ -f ${OUTPUT_PREFIX}.log ]; then
    echo "✓ Log file created: ${OUTPUT_PREFIX}.log"
    if tail -5 ${OUTPUT_PREFIX}.log | grep -q "Total wall time"; then
        runtime=$(tail -5 ${OUTPUT_PREFIX}.log | grep "Total wall time" | cut -d: -f2-)
        echo "  Runtime:$runtime"
    fi
    if grep -q "bond_coeff.*4225" lammps_input.in; then
        echo "  Bond strength: 4225 (strong harmonic bonds)"
    fi
else
    echo "✗ Log file missing"
fi

cd ..
echo ""
echo "Test completed. Files are in: $JOB_DIR/"