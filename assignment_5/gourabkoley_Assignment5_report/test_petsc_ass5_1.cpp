#include <petscksp.h>
#include <vector>
#include <string>
#include <sys/stat.h>  // For creating directories

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
    PetscReal diag = -2.0 * (1.0 / (hx * hx) + 1.0 / (hy * hy));
    PetscReal offx = 1.0 / (hx * hx), offy = 1.0 / (hy * hy);
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

    // --- Assemble Laplace system ---
    for (PetscInt j = 0; j < Ny; ++j) {
        for (PetscInt i = 0; i < Nx; ++i) {
            PetscInt row = idx(i, j, Nx);

            if (i == 0) { // Left Dirichlet u=100
                MatSetValue(A, row, row, 1.0, INSERT_VALUES);
                VecSetValue(b, row, 100.0, INSERT_VALUES);
            } else if (i == Nx - 1) { // Right Dirichlet u=0
                MatSetValue(A, row, row, 1.0, INSERT_VALUES);
                VecSetValue(b, row, 0.0, INSERT_VALUES);
            } else if (j == 0) { // Bottom Neumann
                MatSetValue(A, row, row, 1.0, INSERT_VALUES);
                MatSetValue(A, row, idx(i, j + 1, Nx), -1.0, INSERT_VALUES);
            } else if (j == Ny - 1) { // Top Neumann
                MatSetValue(A, row, row, 1.0, INSERT_VALUES);
                MatSetValue(A, row, idx(i, j - 1, Nx), -1.0, INSERT_VALUES);
            } else { // Interior
                MatSetValue(A, row, row, diag, INSERT_VALUES);
                MatSetValue(A, row, idx(i + 1, j, Nx), offx, INSERT_VALUES);
                MatSetValue(A, row, idx(i - 1, j, Nx), offx, INSERT_VALUES);
                MatSetValue(A, row, idx(i, j + 1, Nx), offy, INSERT_VALUES);
                MatSetValue(A, row, idx(i, j - 1, Nx), offy, INSERT_VALUES);
            }
        }
    }

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
        //{"cg", "jacobi"},
        {"cg", "bjacobi"},
        {"cg", "asm"},
        {"cg", "gamg"},
        {"gmres", "none"},
        //{"gmres", "jacobi"},
        {"gmres", "bjacobi"},
        {"gmres", "asm"},
        {"gmres", "gamg"},
        {"minres", "none"},
        //{"minres", "jacobi"},
        {"minres", "gamg"},
        {"fcg", "none"},
        //{"fcg", "jacobi"},
        {"fcg", "bjacobi"},
        {"chebyshev", "none"},
        {"chebyshev", "jacobi"}
    };

    // Create result directory
    std::string folder = "results_" + std::to_string(Nx) + "x" + std::to_string(Ny);
    mkdir(folder.c_str(), 0777);

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
        KSPSolve(ksp, b, x);

        PetscTime(&t2);
        PetscInt iters;
        KSPGetIterationNumber(ksp, &iters);
        PetscPrintf(PETSC_COMM_WORLD,
                    "%-8s + %-7s => iterations: %4d | time: %10.6f s\n",
                    cfg.kspType.c_str(), cfg.pcType.c_str(), iters, t2 - t1);

        // ========================
        // ?? Save solution phi(x,y)
        // ========================
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
        for (PetscInt j = 0; j < Ny; ++j) {
            for (PetscInt i = 0; i < Nx; ++i) {
                PetscInt idx1D = j * Nx + i;
                PetscViewerASCIIPrintf(viewer, "%g ", (double)u_arr[idx1D]);
            }
            PetscViewerASCIIPrintf(viewer, "\n");
        }
        PetscViewerDestroy(&viewer);

        VecRestoreArrayRead(u_local, &u_arr);
        VecScatterDestroy(&scatter);
        VecDestroy(&u_local);
        // ========================

        KSPDestroy(&ksp);
        VecDestroy(&x);
    }

    VecDestroy(&b);
    MatDestroy(&A);
    PetscFinalize();
    return 0;
}