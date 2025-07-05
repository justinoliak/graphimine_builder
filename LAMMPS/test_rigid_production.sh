#!/bin/bash
# Test the rigid body production script logic locally

# Simulate the variables that would be set by SLURM
G=5
PHI=0.1
REP=1
PHI_IDX=9  # 0.1 is the 10th value (index 9)

# Set parameters (from production script)
OUTPUT_PREFIX="sim"
SEED=$((49281 + ${PHI_IDX} * 1000 + ${REP} * 10))

echo "Testing rigid production script with:"
echo "G=$G, PHI=$PHI, REP=$REP"
echo "SEED=$SEED"

# Create test directory
TEST_DIR="test_rigid_production"
mkdir -p $TEST_DIR
cd $TEST_DIR

# Generate structure using existing data
cp ../test_G5_phi01.data structure.data

echo "Generating LAMMPS input file..."

# Create LAMMPS input file with rigid body dynamics (copied from production script)
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

# Minimization
min_style    cg
minimize     1.0e-4 1.0e-6  1000 5000

reset_timestep 0

# Trajectory output
dump         traj all custom 1000 ${OUTPUT_PREFIX}.lammpstrj id mol type x y z
dump_modify  traj sort id

# Short test run
run     5000
EOFLAMMPS

echo "Generated lammps_input.in"
echo "Communication cutoff should be: (2*$G+1)*0.6 = $((2*G+1))*0.6 = $(echo "scale=1; (2*$G+1)*0.6" | bc)"

# Show the key parts of the input file
echo ""
echo "Key lines from generated input:"
grep -A1 "comm_cutoff" lammps_input.in
grep "fix.*rig" lammps_input.in
grep "timestep" lammps_input.in

echo ""
echo "Ready to test with: lmp -in lammps_input.in"