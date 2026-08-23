# Solver Performance Analysis using PETSc

## Overview
This repository contains a comprehensive benchmarking study of various Krylov subspace solvers and preconditioners for solving diffusion problems. The project evaluates computational performance and convergence stability across increasing grid resolutions to establish best practices for large-scale linear systems within Computational Fluid Dynamics (CFD) solver stacks.

## Methodology
* **1D Diffusion Problem:** Formulated and implemented the 1D diffusion equation ($d^2u/dx^2 - 2u = f(x)$) using Finite Difference (FDM), Finite Element (FEM), and Finite Volume (FVM) methods. Validated against analytical solutions across Dirichlet-Dirichlet and Dirichlet-Neumann boundary conditions.
* **2D Steady-State Diffusion:** Discretized the 2D Laplacian ($\nabla^2\phi = 0$) using a 5-point FDM stencil with mixed boundary conditions.
* **PETSc Integration:** Automated grid setup, sparse matrix assembly, and solver execution leveraging the Portable, Extensible Toolkit for Scientific Computation (PETSc).

## Solvers & Preconditioners Evaluated
The following combinations were benchmarked across 7 different grid resolutions (from 5x5 up to 250x250):
* **Krylov Solvers:** CG, GMRES, MINRES, FCG, and Chebyshev.
* **Preconditioners:** None, Jacobi, Block Jacobi, ASM (Additive Schwarz Method), and GAMG (Geometric Algebraic Multigrid).

## Key Findings
* Achieved solver accuracy with residuals falling below $10^{-12}$ for all 1D validation cases.
* **Optimal Configuration:** The combination of **GMRES** paired with **ASM** or **GAMG** preconditioners delivered the most robust convergence rates.
* **Stability:** This optimal configuration successfully resolved the numerical instabilities observed in CG and Chebyshev solvers when applied to finer computational meshes.

## Tech Stack
* **Framework:** PETSc
* **Language:** C/C++ (or Python/mpi4py depending on your specific implementation)
* **Numerical Methods:** FDM, FEM, FVM