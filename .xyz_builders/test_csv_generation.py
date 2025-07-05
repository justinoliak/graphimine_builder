#!/usr/bin/env python3
"""
Test script for CSV generation - small range N=100-300 with 3 runs each
"""

import csv
from graphimine_count_only_csv import generate_single_structure

def main():
    # Generate test CSV data
    with open('test_graphimine_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['N_target', 'monomers', 'imines', 'aldehydes'])
        
        # Small test range: N from 100 to 300 in intervals of 100, with 3 runs each
        for N in range(100, 301, 100):
            print(f"Processing N={N}...")
            for run in range(3):
                result = generate_single_structure(N)
                if result:
                    monomers, imines, aldehydes = result
                    writer.writerow([N, monomers, imines, aldehydes])
                    print(f"  Run {run+1}: ({monomers}, {imines}, {aldehydes})")
    
    print("Test CSV file 'test_graphimine_data.csv' generated successfully!")

if __name__ == "__main__":
    main()