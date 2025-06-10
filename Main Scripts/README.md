# Graphimine Builder

Simple tool for generating circular graphimine nanostructures on hexagonal lattices.

## Usage

```bash
python3 generate_hexagonal_graphimine.py N
```

Where `N` is the number of rings. Outputs `N=[number]_graphimine.xyz`.

## Example

```bash
python3 generate_hexagonal_graphimine.py 50
# Creates N=50_graphimine.xyz with ~109k atoms
```

## What it does

- Generates hexagonal lattice of monomers
- Filters to circular boundary  
- Adds proper edge termination (aldehydes, amines, imine hydrogens)
- Uses 90° hydrogen placement for realistic geometry

Requires numpy and scipy.