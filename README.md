# Optimized Neighborhood Correlation Implementation

This repository contains the code and data for my thesis project on optimizing Neighborhood Correlation (NC) for bioinformatics sequence analysis.

## Project Overview

This project implements and compares three versions of the Neighborhood Correlation algorithm:
1. Python implementation
2. Custom C++ implementation
3. Eigen C++ implementation

The goal is to analyze and improve the computational efficiency of NC for large-scale genomic data analysis.

## Repository Structure

- `/src`: Source code for all implementations
  - `/python`: Python implementation
  - `/cpp`: C++ implementations (both custom and Eigen)
- `/data`: datasets and result outputs
- `/scripts`: Python script to compare output

## Key Features

- Optimized C++ implementation of Neighborhood Correlation
- Comparative analysis of execution times across different implementations
- Scalability testing with datasets of varying sizes
- Parallelization in C++ implementations for improved performance

## Results Summary

Our custom C++ implementation achieved significant speedups compared to the Python version:
- 274.5x faster for the dataset 9M_450_200_50.
- Consistent accuracy with minimal output differences between implementations

## How to Compile and Run (This is how I run on Ubuntu)

### Prerequisites
- G++ compiler with C++17 support
- OpenMP library
- Eigen library (for Eigen implementation)

### Compilation

#### Custom C++ Implementation
g++ -std=c++17 -O3 -fopenmp main.cpp -o myexe

#### Eigen C++ Implementation
g++ -std=c++17 -O3 -fopenmp -march=native -I/usr/include/eigen3 main.cpp -o myexe

### Usage
./myexe <input_file>

To run with a specific number of threads (e.g., 1):
OMP_NUM_THREADS=1 ./myexe <input_file>

To run with all available processors:
OMP_NUM_THREADS=$(nproc) ./myexe <input_file>

The Custom C++ implementation automatically detects the input format, while for Eigen use -3 to specify 3 column input file:
OMP_NUM_THREADS=1 ./myexe 6M_300_200_50.tab -v -3

### Additional Options

- `-v`: Enable verbose output
- `-3`: Use 3-column input format (required for Eigen implementation, automatic for Custom C++)
- `-o <output_file>`: Specify output file (default: output.txt)
- `-x`: Enable cross-file mode
- `-c <threshold>`: Set consideration threshold (default: 30)
- `-t <transform>`: Apply score transform (options: sqrt, cubicroot, 2.5root, log10, ln)

For more detailed information on options, run:
./myexe --help

## Datasets

We used the following datasets in our experiments:
- 1000pieris.tab (1,000 sequences)
- 3M_150_200_50.tab (2,992,379 sequences)
- 6M_300_200_50.tab (5,984,959 sequences)
- 9M_450_200_50.tab (8,977,465 sequences)
- simil.m8 (2,344,775 sequences)

## Author

Theodor Lindberg
