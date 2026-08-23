#!/bin/bash
# ================================================================
# PETSc Multi-grid-size Benchmark Runner
# ================================================================

# --- Compilation ---
#echo "Compiling PETSc C++ solver..."
#mpicxx test_petsc_ass5_1.cpp -o test_petsc_ass5_1 $(pkg-config --cflags --libs PETSc) -lm

#if [ $? -ne 0 ]; then
#    echo "? Compilation failed. Check PETSc paths or file name."
#    exit 1
#fi

# --- Grid sizes ---
  grid_sizes=(5 10 20 40 80 100 250 500 )
  
 output_folder="benchmark_results"
  mkdir -p "$output_folder"
  log_file="$output_folder/solver_performance.log"
  echo "# Nx Ny KSP PC Iterations  Time(s)" > "$log_file"
  
# --- Run PETSc for each grid ---
  
  for Nx in "${grid_sizes[@]}"; do
      Ny=$Nx
      echo "--------------------------------------------------------"
      echo " Running for Nx=$Nx, Ny=$Ny"
      echo "--------------------------------------------------------"
  
      mpirun -n 20 ./petsc_test -Nx $Nx -Ny $Ny | tee tmp_output.txt
      sleep 1  # allow MPI to cleanup before next run
#      pkill -9 mpirun
  
      grep -E "=>" tmp_output.txt | \
          awk -v nx=$Nx -v ny=$Ny '{printf "%d %d %s %s %s %s\n", nx, ny, $1, $3, $6, $9}' \
          >> "$log_file"
  
      rm -f tmp_output.txt
done   
  
echo "Done! All results stored under $output_folder/"
 
 
#!/bin/bash
# PETSc Multi-grid-size Benchmark Runner

#grid_sizes=(5 10 20 40 80 100 250 500)

#output_folder="benchmark_results"
#mkdir -p "$output_folder"
#master_log="$output_folder/solver_performance.log"
#echo "# Nx Ny KSP PC Iterations Residual Time" > "$master_log"

# Loop over grids
#for Nx in "${grid_sizes[@]}"; do
#    Ny=$Nx
#    echo "--------------------------------------------------------"
#    echo " Running for Nx=$Nx, Ny=$Ny"
#    echo "--------------------------------------------------------"#

#    mpirun -n 20 ./test_petsc_ass5_1 -Nx $Nx -Ny $Ny

    # Wait a bit for cleanup
#    sleep 3

    # Append individual run log to master log
#    run_log="results_${Nx}x${Ny}/solver_performance.log"
#    if [ -f "$run_log" ]; then
#        # Skip header line when appending
#        tail -n +2 "$run_log" >> "$master_log"
#    else
#        echo "?? Missing log for Nx=$Nx, Ny=$Ny"
#    fi
#done

#echo "All runs done. Aggregated log at $master_log"


#!/bin/bash
#set -euo pipefail

# --- Grid sizes to test ---
#grid_sizes=(5 10 20 40 80 100 250 500)

#output_folder="benchmark_results"
#mkdir -p "$output_folder"
#log_file="$output_folder/solver_performance.log"

# --- Header for combined log ---
#echo "# Nx Ny KSP PC Iterations Residual Time" > "$log_file"

# --- Run loop for all grid sizes ---
#for Nx in "${grid_sizes[@]}"; do
#  Ny=$Nx
#  echo "--------------------------------------------------------"
#  echo " Running for Nx=$Nx, Ny=$Ny"
#  echo "--------------------------------------------------------"

  # Run PETSc executable with MPI
#  mpirun -n 20 ./test_petsc_ass5_1 -Nx $Nx -Ny $Ny | tee tmp_output.txt

  # Wait for MPI to fully clean up
#  sleep 2

  # Append all numeric lines from tmp output
#  grep -E '^[0-9]+' tmp_output.txt >> "$log_file" || true

#  rm -f tmp_output.txt
#done

#echo "? Done! All results saved in $log_file"
