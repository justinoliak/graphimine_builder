#!/usr/bin/env python3
"""
Launch production runs for large N graphimine study.
N144 and N256 platelets, φ = 0.05-0.65, 3 replicates each.
"""
import yaml, subprocess, pathlib, datetime, sys

JOB = pathlib.Path("large_N_study.yml")
TEMPLATE = pathlib.Path("run_bulk.in")

with open(JOB) as f:
    cfg = yaml.safe_load(f)

# Generate data files first
print("Generating large N data files...")

for grid_item in cfg["grid"]:
    for phi in cfg["phi_list"]:
        for rep in range(1, cfg["replicas"] + 1):
            data_file = f"configs/{cfg['out_prefix']}_{grid_item['tag']}_phi{phi:.2f}_rep{rep}.data"
            
            # Skip if data file already exists
            if pathlib.Path(data_file).exists():
                print(f"[exists] {data_file}")
                continue
                
            # Generate data file using bead builder
            seed = cfg['seed_base'] + int(phi * 1e5) + rep * 1000
            cmd_builder = [
                "python", "graphimine_beadbuilder.py",
                "--N", str(grid_item['N']),
                "--phi", str(phi),
                "--copies", str(grid_item['copies']),
                "--output", data_file,
                "--seed", str(seed)
            ]
            
            try:
                print(f"Generating {data_file}...")
                subprocess.run(cmd_builder, check=True)
                print(f"✓ Generated {data_file}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to generate {data_file}: exit code {e.returncode}")
                continue

print("✓ All data files generated")
print("\nStarting production runs...")

with open("production_large_N.log", "w") as prod_log:
    for grid_item in cfg["grid"]:
        for phi in cfg["phi_list"]:
            for rep in range(1, cfg["replicas"] + 1):
                # Build filenames
                data_file = f"configs/{cfg['out_prefix']}_{grid_item['tag']}_phi{phi:.2f}_rep{rep}.data"
                input_file = f"run_{grid_item['tag']}_phi{phi:.2f}_rep{rep}.in"
                log_file = f"log_{grid_item['tag']}_phi{phi:.2f}_rep{rep}"
                
                # Check if data file exists
                if not pathlib.Path(data_file).exists():
                    msg = f"[skip] Missing data file: {data_file}"
                    print(msg)
                    prod_log.write(f"{datetime.datetime.now().isoformat()} {msg}\n")
                    continue
                
                # Generate input file from template
                cmd_sed = [
                    "sed",
                    "-e", f"s#__DATAFILE__#{data_file}#g",
                    "-e", f"s#__PHI__#{phi:.2f}#g", 
                    "-e", f"s#__N__#{grid_item['tag']}_rep{rep}#g",
                    str(TEMPLATE)
                ]
                
                with open(input_file, 'w') as f:
                    subprocess.run(cmd_sed, stdout=f, check=True)
                
                # Launch LAMMPS
                cmd_lammps = ["lmp", "-in", input_file]
                
                timestamp = datetime.datetime.now().isoformat()
                msg = f"Starting {grid_item['tag']} φ={phi:.2f} rep={rep}"
                print(f"{timestamp} {msg}")
                print(f"  Data: {data_file}")
                print(f"  Input: {input_file}")
                print(f"  Log: {log_file}")
                
                prod_log.write(f"{timestamp} {msg}\n")
                prod_log.write(f"  Data: {data_file}\n")
                prod_log.write(f"  Input: {input_file}\n")
                prod_log.write(f"  Log: {log_file}\n")
                prod_log.flush()
                
                try:
                    with open(log_file, 'w') as f:
                        subprocess.run(cmd_lammps, stdout=f, stderr=subprocess.STDOUT, check=True)
                    success_msg = "  ✓ Completed successfully"
                    print(success_msg)
                    prod_log.write(f"{success_msg}\n")
                except subprocess.CalledProcessError as e:
                    error_msg = f"  ✗ Failed with exit code {e.returncode}"
                    print(error_msg)
                    prod_log.write(f"{error_msg}\n")
                    continue
                except KeyboardInterrupt:
                    interrupt_msg = "  ⚠ Interrupted by user"
                    print(interrupt_msg)
                    prod_log.write(f"{interrupt_msg}\n")
                    sys.exit(1)
                
                prod_log.flush()

final_msg = f"\n{datetime.datetime.now().isoformat()} Large N production sweep completed!"
print(final_msg)
with open("production_large_N.log", "a") as prod_log:
    prod_log.write(final_msg + "\n")