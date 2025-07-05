#!/bin/bash
# Test script for rigid body dynamics - single job simulation

# Test parameters
G=5
PHI=0.1
REP=1
PHI_IDX=9

echo "Testing rigid body dynamics with G=$G, phi=$PHI"

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

echo "Creating LAMMPS input with rigid body dynamics..."

# Create LAMMPS input file with rigid body dynamics (from production script)
cat > lammps_input.in << EOFLAMMPS
# Rigid body platelets
units       lj
atom_style  molecular
boundary    p p p

log ${OUTPUT_PREFIX}.log

read_data   structure.data
mass        * 1.0

# Group all platelet atoms
group plates type 1

# Automatic communication cutoff based on G value
variable comm_cutoff equal (2*${G}+1)*0.6
comm_modify cutoff \${comm_cutoff}

# WCA potential
pair_style      lj/cut 1.12246
pair_modify     shift yes
pair_coeff      * * 1.0 1.0 1.12246

neighbor     1.0 bin
neigh_modify every 1 delay 0 check yes

# Harmonic bonds (ignored by rigid fix)
bond_style   harmonic
bond_coeff   * 1000.0 1.0

# Rigid body integration with Langevin thermostat
fix  rig plates rigid/small molecule langevin 1.0 1.0 1.0 ${SEED}

# Initial velocities
velocity plates create 1.0 ${SEED} mom yes rot yes dist gaussian

timestep 0.005

# Minimal thermo output
thermo_style custom step temp etotal press
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
    if grep -q "rigid bodies" ${OUTPUT_PREFIX}.log; then
        bodies=$(grep "rigid bodies" ${OUTPUT_PREFIX}.log | head -1)
        echo "  $bodies"
    fi
    if tail -5 ${OUTPUT_PREFIX}.log | grep -q "Total wall time"; then
        runtime=$(tail -5 ${OUTPUT_PREFIX}.log | grep "Total wall time" | cut -d: -f2-)
        echo "  Runtime:$runtime"
    fi
else
    echo "✗ Log file missing"
fi

cd ..
echo ""
echo "Test completed. Files are in: $JOB_DIR/"