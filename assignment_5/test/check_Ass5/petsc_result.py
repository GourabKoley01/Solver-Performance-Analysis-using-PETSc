import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Load benchmark log ===
log_path = "benchmark_results/solver_performance.log"
df = pd.read_csv(log_path, sep=r"\s+", comment="#",
                 names=["Nx", "Ny", "KSP", "PC", "Iterations", "Time"])

print(df.head())

# === Create plots folder ===
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# === Benchmark plot: time vs grid size ===
plt.figure(figsize=(8,6))
for (ksp, pc), grp in df.groupby(["KSP", "PC"]):
    plt.plot(grp["Nx"], grp["Time"], marker="o", label=f"{ksp}+{pc}")
plt.xlabel("Grid size (Nx = Ny)")
plt.ylabel("Solve Time (s)")
plt.title("Solver Time vs Grid Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "time_vs_grid.png"), dpi=300)
plt.close()

# === Benchmark plot: iterations vs grid size ===
plt.figure(figsize=(8,6))
for (ksp, pc), grp in df.groupby(["KSP", "PC"]):
    plt.plot(grp["Nx"], grp["Iterations"], marker="s", label=f"{ksp}+{pc}")
plt.xlabel("Grid size (Nx = Ny)")
plt.ylabel("Iteration Count")
plt.title("Solver Iterations vs Grid Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "iters_vs_grid.png"), dpi=300)
plt.close()

# === Contour Plots for each solver ===
for Nx in sorted(df["Nx"].unique()):
    Ny = Nx
    folder = f"results_{Nx}x{Ny}"
    if not os.path.isdir(folder):
        print(f"?? Missing folder: {folder}")
        continue

    for (ksp, pc) in df.loc[(df["Nx"] == Nx) & (df["Ny"] == Ny), ["KSP", "PC"]].drop_duplicates().itertuples(index=False):
        filename = f"solution_{ksp}_{pc}_Nx{Nx}_Ny{Ny}.txt"
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"?? File not found: {filepath}")
            continue

        # Load the numerical solution
        data = np.loadtxt(filepath, comments="#")
        x = np.linspace(0, 2.0, Nx)
        y = np.linspace(0, 1.0, Ny)
        X, Y = np.meshgrid(x, y)

        # Plot contour
        plt.figure(figsize=(6,4))
        cp = plt.contourf(X, Y, data, levels=20, cmap="turbo")
        plt.colorbar(cp, label="Temperature (u)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{ksp.upper()} + {pc.upper()} | Grid {Nx}x{Ny}")
        fname = f"contour_{Nx}x{Ny}_{ksp}_{pc}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, fname), dpi=300)
        plt.close()

print("? All plots generated successfully in 'plots/'")
