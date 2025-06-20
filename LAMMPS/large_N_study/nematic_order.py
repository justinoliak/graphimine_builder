#!/usr/bin/env python3
"""
nematic_order.py  —  Compute the nematic order parameter S from a LAMMPS dump trajectory.

Features
--------
* **3‑D aware** – directors can point anywhere, no hard‑wired z‑component.
* **Geometry switch** – `--geometry rod` (long axis) or `--geometry disc` (sheet normal).
* **Stream processing** – reads one frame at a time; no huge memory footprint.
* **CLI flags** – skip equilibration frames, stride through trajectory, verbose progress.

Example
-------
    python nematic_order.py traj.lammpstrj \\
            --skip 1000       # ignore first 1 000 frames (equilibration) \\
            --every 10        # use every 10th frame \\
            --geometry disc   # treat each molecule as a disc/sheet

Outputs
-------
* Scalar S and the director vector (principal eigenvector of ⟨Q⟩).

Assumptions
-----------
The dump uses a line starting with::

    ITEM: ATOMS id mol type x y z …

Fields may be in any order but **must** include ``id``, ``mol``, ``x``, ``y``, ``z``.
"""
import argparse
import numpy as np
import sys

# -----------------------------------------------------------------------------#
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute nematic order parameter S "
                                            "from a LAMMPS dump trajectory")
    p.add_argument("dump", help="LAMMPS dump / .lammpstrj file")
    p.add_argument("--skip", type=int, default=0,
                   help="number of initial frames to skip (equilibration)")
    p.add_argument("--every", type=int, default=1,
                   help="use every n‑th frame (stride)")
    p.add_argument("--geometry", choices=["rod", "disc"], default="rod",
                   help="'rod' = use longest principal axis; "
                        "'disc' = use normal (smallest axis)")
    p.add_argument("--verbose", action="store_true",
                   help="print running progress to stderr")
    return p.parse_args()


# -----------------------------------------------------------------------------#
def read_frames(path):
    """Yield (timestep, ndarray[N,5]) with columns id, mol, x, y, z."""
    with open(path, "r") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                sys.exit("Error: unexpected format (missing TIMESTEP)")
            timestep = int(fh.readline())
            if not fh.readline().startswith("ITEM: NUMBER"):
                sys.exit("Error: unexpected format (missing NUMBER OF ATOMS)")
            n_atoms = int(fh.readline())
            # skip box bounds
            if not fh.readline().startswith("ITEM: BOX BOUNDS"):
                sys.exit("Error: unexpected format (missing BOX BOUNDS)")
            for _ in range(3):
                fh.readline()

            atom_hdr = fh.readline()
            if not atom_hdr.startswith("ITEM: ATOMS"):
                sys.exit("Error: unexpected format (missing ATOMS header)")
            fields = atom_hdr.strip().split()[2:]
            try:
                idx_id = fields.index("id")
                idx_mol = fields.index("mol")
                idx_x = fields.index("x")
                idx_y = fields.index("y")
                idx_z = fields.index("z")
            except ValueError:
                sys.exit("Error: dump must contain id, mol, x, y, z columns")

            arr = np.empty((n_atoms, 5))
            for i in range(n_atoms):
                parts = fh.readline().split()
                arr[i, 0] = int(parts[idx_id])
                arr[i, 1] = int(parts[idx_mol])
                arr[i, 2] = float(parts[idx_x])
                arr[i, 3] = float(parts[idx_y])
                arr[i, 4] = float(parts[idx_z])
            yield timestep, arr


# -----------------------------------------------------------------------------#
def accumulate_Q(dump_path, skip, every, geom, verbose=False):
    tot_Q = np.zeros((3, 3))
    tot_mol = 0
    frames_used = 0

    for fnum, (ts, atoms) in enumerate(read_frames(dump_path)):
        if fnum < skip or (fnum - skip) % every:
            continue
        frames_used += 1
        mol_ids = np.unique(atoms[:, 1].astype(int))
        for mol in mol_ids:
            coords = atoms[atoms[:, 1] == mol, 2:5]
            coords -= coords.mean(axis=0, keepdims=True)
            cov = np.cov(coords, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            director = (
                evecs[:, np.argmax(evals)]
                if geom == "rod"
                else evecs[:, np.argmin(evals)]
            )
            norm = np.linalg.norm(director)
            if norm == 0:
                continue
            u = director / norm
            tot_Q += 1.5 * np.outer(u, u) - 0.5 * np.identity(3)
            tot_mol += 1

        if verbose and frames_used % 10 == 0:
            print(f"[{frames_used:>6}] timestep {ts}", file=sys.stderr)

    if tot_mol == 0:
        sys.exit("No molecules processed; check mol IDs or skip/stride options.")

    return tot_Q / tot_mol, tot_mol, frames_used


# -----------------------------------------------------------------------------#
def main():
    args = parse_args()
    Q_avg, n_mol, n_frames = accumulate_Q(
        args.dump, args.skip, args.every, args.geometry, args.verbose
    )
    eigvals, eigvecs = np.linalg.eigh(Q_avg)
    S = eigvals[-1]
    director = eigvecs[:, -1]

    print(f"S  = {S:.6f}")
    print("Director =", director)
    if args.verbose:
        print(f"Frames analysed: {n_frames}")
        print(f"Molecules total: {n_mol}")


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()