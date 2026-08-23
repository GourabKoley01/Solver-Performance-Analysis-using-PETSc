import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# === Load benchmark log ===
log_path = "benchmark_results/solver_performance.log"
df = pd.read_csv(log_path, sep=r"\s+", comment="#",
                 names=["Nx", "Ny", "KSP", "PC", "Iterations", "Time"])  # ? Added Residual

# Convert relevant columns to numeric and clean data
df["Time"] = pd.to_numeric(df["Time"], errors='coerce')
df["Iterations"] = pd.to_numeric(df["Iterations"], errors='coerce')
#df["Residual"] = pd.to_numeric(df["Residual"], errors='coerce')
df = df.dropna(subset=["Time", "Iterations"])

# Calculate total nodes
df["Nodes"] = df["Nx"] * df["Ny"]

# === Create plots folder ===
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# === Plot 1: Nodes vs Time ===
plt.figure(figsize=(10, 6))
for (ksp, pc), grp in df.groupby(["KSP", "PC"]):
    plt.plot(grp["Nodes"], grp["Time"], marker="o", label=f"{ksp}+{pc}")
plt.xlabel("Number of Nodes (Nx * Ny)")
plt.ylabel("Solve Time (s)")
plt.title("Solver Time vs Number of Nodes")
plt.legend(loc="best", fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "time_vs_nodes.png"), dpi=300)
plt.close()

# === Plot 2: Nodes vs Iterations ===
max_iter_threshold = 10000
df_conv = df[df["Iterations"] < max_iter_threshold]

plt.figure(figsize=(10, 6))
for (ksp, pc), grp in df_conv.groupby(["KSP", "PC"]):
    plt.plot(grp["Nodes"], grp["Iterations"], marker="s", label=f"{ksp}+{pc}")
plt.xlabel("Number of Nodes (Nx * Ny)")
plt.ylabel("Iteration Count")
plt.title("Solver Iterations vs Number of Nodes (Converged Runs)")
plt.legend(loc="best", fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "iters_vs_nodes.png"), dpi=300)
plt.close()
'''
# === Plot 3: Residual Norm (log scale) ===
plt.figure(figsize=(10, 6))
for (ksp, pc), grp in df.groupby(["KSP", "PC"]):
    plt.plot(grp["Nodes"], grp["Residual"], marker="^", label=f"{ksp}+{pc}")  # ? Fixed column name
plt.xlabel("Number of Nodes (Nx * Ny)")
plt.ylabel("Residual Norm")
plt.yscale('log')
plt.title("Residual Norm vs Number of Nodes")
plt.legend(loc="best", fontsize='small')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "residual_vs_nodes.png"), dpi=300)
plt.close()'''

# === Contour & Surface Subplots ===
ncols, nrows = 7, 2

for Nx in sorted(df["Nx"].unique()):
    Ny = Nx
    folder = f"results_{Nx}x{Ny}"
    if not os.path.isdir(folder):
        print(f"?? Missing folder: {folder}")
        continue

    runs = df[(df["Nx"] == Nx) & (df["Ny"] == Ny)][["KSP", "PC"]].drop_duplicates()
    num_runs = len(runs)

    fig_c, axs_c = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    fig_s, axs_s = plt.subplots(nrows, ncols, subplot_kw={"projection": "3d"}, figsize=(4*ncols, 3*nrows))

    axs_c = axs_c.flatten()
    axs_s = axs_s.flatten()

    x = np.linspace(0, 1.0, Nx)
    y = np.linspace(0, 1.0, Ny)
    X, Y = np.meshgrid(x, y)

    for idx, (_, (ksp, pc)) in enumerate(runs.iterrows()):
        filename = f"solution_{ksp}_{pc}_Nx{Nx}_Ny{Ny}.txt"
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"?? File not found: {filepath}")
            axs_c[idx].axis('off')
            axs_s[idx].axis('off')
            continue

        data = np.genfromtxt(filepath, comments="#", invalid_raise=False, filling_values=np.nan)

        if data.size == 0:
            axs_c[idx].axis('off')
            axs_s[idx].axis('off')
            continue

        if data.ndim == 1:
            try:
                data = data.reshape(Ny, Nx)
            except Exception as e:
                axs_c[idx].axis('off')
                axs_s[idx].axis('off')
                continue

        # Contour
        cp = axs_c[idx].contourf(X, Y, data, levels=100, cmap="turbo")
        fig_c.colorbar(cp, ax=axs_c[idx])
        axs_c[idx].set_xlabel("x")
        axs_c[idx].set_ylabel("y")
        axs_c[idx].set_title(f"{ksp.upper()} + {pc.upper()} (Contour)")

        # Surface
        surf = axs_s[idx].plot_surface(X, Y, data, cmap="turbo", edgecolor='none')
        fig_s.colorbar(surf, ax=axs_s[idx], shrink=0.6, aspect=10)
        axs_s[idx].set_xlabel("x")
        axs_s[idx].set_ylabel("y")
        axs_s[idx].set_zlabel("Temperature (u)")
        axs_s[idx].set_title(f"{ksp.upper()} + {pc.upper()} (Surface)")
        axs_s[idx].view_init(elev=30, azim=-60)

    # Turn off unused subplots
    for i in range(num_runs, nrows*ncols):
        axs_c[i].axis('off')
        axs_s[i].axis('off')

    fig_c.suptitle(f"Contour Plots for Grid {Nx}x{Ny}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_c.savefig(os.path.join(plot_dir, f"contour_allsolvers_{Nx}x{Ny}.png"), dpi=300)
    plt.close(fig_c)

    fig_s.suptitle(f"Surface Plots for Grid {Nx}x{Ny}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_s.savefig(os.path.join(plot_dir, f"surface_allsolvers_{Nx}x{Ny}.png"), dpi=300)
    plt.close(fig_s)

print(" All requested plots generated successfully in 'plots/'")
