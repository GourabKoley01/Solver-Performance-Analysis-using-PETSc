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

## Visualizations & Performance Analysis

The following visualizations demonstrate the stability, accuracy, and scaling behavior of various solver-preconditioner pairs across increasing grid resolutions.

### 1. Steady-State Contour Plots (Grid Resolution Scaling)
These contour plots visualize the 2D steady-state temperature distribution. As the grid resolution increases from 10x10 up to 250x250, the differences in solver stability become apparent[cite: 1]. 

* **Robustness:** **GMRES** paired with **GAMG** or **ASM** preconditioners consistently maintained physical accuracy and robust convergence across all grid sizes[cite: 1].
* **Instability:** Solvers like **Chebyshev** and **CG** were only stable on coarse grids, diverging or producing highly inaccurate, non-physical results on finer meshes[cite: 1]. **MINRES** exhibited false convergence, displaying low residuals but physically incorrect solutions[cite: 1]. **FCG** showed poor convergence across all test cases[cite: 1].

<div align="center">
  <img src="assignment_5/new+test/testing/sub_plots/contour_allsolvers_10x10.jpg" width="45%" alt="10x10 Contour">
  <img src="assignment_5/new+test/testing/sub_plots/contour_allsolvers_20x20.jpg" width="45%" alt="20x20 Contour">
  <img src="assignment_5/new+test/testing/sub_plots/contour_allsolvers_40x40.jpg" width="45%" alt="40x40 Contour">
  <img src="assignment_5/new+test/testing/sub_plots/contour_allsolvers_80x80.jpg" width="45%" alt="80x80 Contour">
  <img src="assignment_5/new+test/testing/sub_plots/contour_allsolvers_250x250.jpg" width="90%" alt="250x250 Contour">
</div>

### 2. 3D Surface Plots (Coarse Grids)
The surface plots for the 5x5 and 10x10 grids provide a 3D perspective of the diffusion profile. They clearly illustrate the applied boundary conditions: $\phi=100$ at $x=0$, $\phi=0$ at $x=L_x$, and insulated boundaries ($\frac{\partial\phi}{\partial y}=0$) at $y=0$ and $y=L_y$.

<div align="center">
  <img src="assignment_5/new+test/testing/sub_plots/surface_allsolvers_5x5.jpg" width="45%" alt="5x5 Surface Plot">
  <img src="assignment_5/new+test/testing/sub_plots/surface_allsolvers_10x10.jpg" width="45%" alt="10x10 Surface Plot">
</div>

### 3. Computational Scaling: Iterations vs. Number of Nodes
This graph tracks the computational cost (iteration count) required for the solvers to converge as the global degrees of freedom (nodes) increase.

* Preconditioning drastically reduces the iteration count. 
* Un-preconditioned solvers experience exponential growth in iterations or fail entirely as the grid scales.

<div align="center">
  <img src="assignment_5/new+test/testing/sub_plots/iters_vs_nodes_labeled.jpg" width="70%" alt="Iterations vs Nodes">
</div>

## Tech Stack
* **Framework:** PETSc
* **Language:** C/C++ (or Python/mpi4py depending on your specific implementation)
* **Numerical Methods:** FDM, FEM, FVM
