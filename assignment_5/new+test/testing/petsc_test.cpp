#include <petscksp.h>
#include <vector>
#include <string>
#include <sys/stat.h>  // For mkdir
#include <cstdio>      // For FILE operations

inline PetscInt idx(PetscInt i, PetscInt j, PetscInt Nx) { return j * Nx + i; }

int main(int argc, char **args)
{
    PetscInitialize(&argc, &args, NULL,
                    "Laplace 2D multi-solver comparison with result saving\n");

    // --- Domain setup ---
    PetscInt Nx = 5, Ny = 5;
    PetscReal Lx = 1.0, Ly = 1.0;
    PetscOptionsGetInt(NULL, NULL, "-Nx", &Nx, NULL);
    PetscOptionsGetInt(NULL, NULL, "-Ny", &Ny, NULL);

    PetscReal hx = Lx / (Nx - 1), hy = Ly / (Ny - 1);
    const PetscReal offx = 1.0 / (hx * hx);
    const PetscReal offy = 1.0 / (hy * hy);
    const PetscReal diag = -2.0 * (offx + offy);
    PetscInt N = Nx * Ny;

    // --- Create A, b ---
    Mat A;
    Vec b;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
    MatSetFromOptions(A);
    MatSetUp(A);

    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, N);
    VecSetFromOptions(b);

    // We will need to know which nodes are Dirichlet and their values
    std::vector<int> is_dirichlet(N, 0);
    std::vector<PetscScalar> dirichlet_val(N, 0.0);

    // Mark left and right Dirichlet nodes
    for (PetscInt j = 0; j < Ny; ++j) {
        PetscInt left = idx(0, j, Nx);
        PetscInt right = idx(Nx - 1, j, Nx);
        is_dirichlet[left] = 1;       dirichlet_val[left] = 100.0; // u(0,y)=100
        is_dirichlet[right] = 1;      dirichlet_val[right] = 0.0;  // u(Lx,y)=0
    }

    // --- Assemble Laplace system ---
    // First, set Dirichlet rows as identity (so other processes/rows won't mistakenly couple to them)
    for (PetscInt j = 0; j < Ny; ++j) {
        for (PetscInt i = 0; i < Nx; ++i) {
            PetscInt row = idx(i, j, Nx);
            if (is_dirichlet[row]) {
                MatSetValue(A, row, row, 1.0, INSERT_VALUES);
                VecSetValue(b, row, dirichlet_val[row], INSERT_VALUES);
            }
        }
    }

    // Now fill the remaining rows (Neumann and interior) taking care to eliminate Dirichlet columns by moving known contributions to RHS
    for (PetscInt j = 0; j < Ny; ++j) {
        for (PetscInt i = 0; i < Nx; ++i) {
            PetscInt row = idx(i, j, Nx);
            if (is_dirichlet[row]) continue; // already set above

            // If this row is bottom (j==0) or top (j==Ny-1) -> apply Neumann by ghost elimination
            if (j == 0 || j == Ny - 1) {
                // Use ghost-point elimination: u_{i,-1} = u_{i,1}  (bottom)
                // or u_{i,Ny} = u_{i,Ny-2}  (top)
                // This results in coefficient for the adjacent interior y-node becoming 2*offy

                // Start with diagonal contribution
                MatSetValue(A, row, row, diag, INSERT_VALUES);

                // x-neighbors (left/right). If neighbor is Dirichlet, move contribution to RHS
                // Right neighbor (i+1)
                if (i + 1 < Nx) {
                    PetscInt colR = idx(i + 1, j, Nx);
                    if (is_dirichlet[colR]) {
                        // subtract offx * u_D from RHS (we move term to RHS: A_row,col * u_col -> -offx * value)
                        VecSetValue(b, row, -offx * dirichlet_val[colR], ADD_VALUES);
                    } else {
                        MatSetValue(A, row, colR, offx, INSERT_VALUES);
                    }
                }
                // Left neighbor (i-1)
                if (i - 1 >= 0) {
                    PetscInt colL = idx(i - 1, j, Nx);
                    if (is_dirichlet[colL]) {
                        VecSetValue(b, row, -offx * dirichlet_val[colL], ADD_VALUES);
                    } else {
                        MatSetValue(A, row, colL, offx, INSERT_VALUES);
                    }
                }

                // y-neighbor: the *adjacent* interior y-node gets 2*offy
                PetscInt adjacent_j = (j == 0) ? j + 1 : j - 1;
                PetscInt colAdj = idx(i, adjacent_j, Nx);
                // adjacent node cannot be Dirichlet in x-direction unless Nx==1 (not the case), but check anyway:
                if (is_dirichlet[colAdj]) {
                    // If it were Dirichlet (rare here), move it to RHS
                    VecSetValue(b, row, -2.0 * offy * dirichlet_val[colAdj], ADD_VALUES);
                } else {
                    MatSetValue(A, row, colAdj, 2.0 * offy, INSERT_VALUES);
                }

                // NOTE: We do NOT set a separate column for the ghost point. Ghost is eliminated.

            } else {
                // Interior row (1 <= j <= Ny-2), standard 5-point stencil with Dirichlet elimination for x-boundaries
                // diagonal
                MatSetValue(A, row, row, diag, INSERT_VALUES);

                // Right neighbor (i+1)
                if (i + 1 < Nx) {
                    PetscInt colR = idx(i + 1, j, Nx);
                    if (is_dirichlet[colR]) {
                        VecSetValue(b, row, -offx * dirichlet_val[colR], ADD_VALUES);
                    } else {
                        MatSetValue(A, row, colR, offx, INSERT_VALUES);
                    }
                }
                // Left neighbor (i-1)
                if (i - 1 >= 0) {
                    PetscInt colL = idx(i - 1, j, Nx);
                    if (is_dirichlet[colL]) {
                        VecSetValue(b, row, -offx * dirichlet_val[colL], ADD_VALUES);
                    } else {
                        MatSetValue(A, row, colL, offx, INSERT_VALUES);
                    }
                }

                // Top neighbor (j+1) -- cannot be Dirichlet (dirichlet only at left/right), but keep check
                PetscInt colUp = idx(i, j + 1, Nx);
                if (is_dirichlet[colUp]) {
                    VecSetValue(b, row, -offy * dirichlet_val[colUp], ADD_VALUES);
                } else {
                    MatSetValue(A, row, colUp, offy, INSERT_VALUES);
                }

                // Bottom neighbor (j-1)
                PetscInt colDown = idx(i, j - 1, Nx);
                if (is_dirichlet[colDown]) {
                    VecSetValue(b, row, -offy * dirichlet_val[colDown], ADD_VALUES);
                } else {
                    MatSetValue(A, row, colDown, offy, INSERT_VALUES);
                }
            }
        }
    }

    // Final assembly
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    // --- Solver configurations ---
    struct SolverConfig {
        std::string kspType;
        std::string pcType;
    };

    std::vector<SolverConfig> solvers = {
        {"cg", "none"},
        {"cg", "bjacobi"},
        {"cg", "asm"},
        {"cg", "gamg"},
        {"gmres", "none"},
        {"gmres", "bjacobi"},
        {"gmres", "asm"},
        {"gmres", "gamg"},
        {"minres", "none"},
        {"minres", "gamg"},
        {"fcg", "none"},
        {"fcg", "bjacobi"},
        {"chebyshev", "none"},
        {"chebyshev", "jacobi"},
    };

    // Create result directory
    std::string folder = "results_" + std::to_string(Nx) + "x" + std::to_string(Ny);
    mkdir(folder.c_str(), 0777);

    // Prepare CSV log file path
    std::string csv_file = folder + "/solver_log.csv";

    // Open CSV file once with header in rank 0
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
        FILE *csv = fopen(csv_file.c_str(), "w");
        if (csv != NULL) {
            fprintf(csv, "KSP_Type,PC_Type,Iter,Residual,Time(s)\n");
            fclose(csv);
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "Failed to open CSV log file for writing\n");
        }
    }

    // --- Loop over solvers ---
    for (const auto &cfg : solvers) {
        KSP ksp;
        Vec x;
        VecDuplicate(b, &x);

        PetscLogDouble t1, t2;
        PetscTime(&t1);

        KSPCreate(PETSC_COMM_WORLD, &ksp);
        KSPSetOperators(ksp, A, A);
        KSPSetType(ksp, cfg.kspType.c_str());
        PC pc;
        KSPGetPC(ksp, &pc);
        PCSetType(pc, cfg.pcType.c_str());
        KSPSetFromOptions(ksp);
        KSPSolve(ksp, b, x);

        PetscTime(&t2);
        PetscInt iters;
        KSPGetIterationNumber(ksp, &iters);

        PetscReal res_norm;
        KSPGetResidualNorm(ksp, &res_norm);

        if (rank == 0) {
            // Print to console
            PetscPrintf(PETSC_COMM_WORLD,
                        "%-10s + %-8s => iterations: %4d | residual: %10.4e | time: %10.6f s\n",
                        cfg.kspType.c_str(), cfg.pcType.c_str(), iters, (double)res_norm, t2 - t1);
            // Append to CSV
            FILE *csv = fopen(csv_file.c_str(), "a");
            if (csv != NULL) {
                fprintf(csv, "%s,%s,%d,%g,%f\n",
                        cfg.kspType.c_str(), cfg.pcType.c_str(), iters, (double)res_norm, t2 - t1);
                fclose(csv);
            }
        }

        // Save solution phi(x,y)
        Vec u_local;
        VecScatter scatter;
        VecScatterCreateToAll(x, &scatter, &u_local);
        VecScatterBegin(scatter, x, u_local, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(scatter, x, u_local, INSERT_VALUES, SCATTER_FORWARD);

        const PetscScalar *u_arr;
        VecGetArrayRead(u_local, &u_arr);

        std::string filename =
            folder + "/solution_" + cfg.kspType + "_" + cfg.pcType +
            "_Nx" + std::to_string(Nx) +
            "_Ny" + std::to_string(Ny) + ".txt";

        PetscViewer viewer;
        PetscViewerASCIIOpen(PETSC_COMM_SELF, filename.c_str(), &viewer);
        PetscViewerASCIIPrintf(viewer, "# Nx=%d Ny=%d\n", Nx, Ny);
        for (PetscInt jj = 0; jj < Ny; ++jj) {
            for (PetscInt ii = 0; ii < Nx; ++ii) {
                PetscInt iD = jj * Nx + ii;
                PetscViewerASCIIPrintf(viewer, "%g ", (double)u_arr[iD]);
            }
            PetscViewerASCIIPrintf(viewer, "\n");
        }
        PetscViewerDestroy(&viewer);

        VecRestoreArrayRead(u_local, &u_arr);
        VecScatterDestroy(&scatter);
        VecDestroy(&u_local);

        KSPDestroy(&ksp);
        VecDestroy(&x);
    }

    VecDestroy(&b);
    MatDestroy(&A);
    PetscFinalize();
    return 0;
}
