import numpy as np 
import pandas as pd 

def f(x):
    return np.log10(x) - x + 3

def f_def(x):
    return (1/(x * np.log(10))) - 1 

def six_order_method(x0, tol, max_iteration):
    x = [x0]
    error = [np.nan]
    i = 0
    iteration = [i]
    while i < max_iteration:
        fxi = f(x[i])
        fdefxi = f_def(x[i])
        if fdefxi == 0:
            print(f"Stopped: Derivative zero at x = {x[i]}")
            break
        # Step 1: w_n
        w = x[i] - (fxi / fdefxi)
        fw = f(w)
        denominator_z = 2*fxi - 5 * fw
        if denominator_z == 0:
            print(f"Stopped: denominator zero at z step, iteration {i}")
            break
        # Step 2: z_n
        z = w - (fw / fdefxi) * ((2 * fxi - fw) / denominator_z)
        fz = f(z)
        denominator_xnp1 = fxi - 3 * fw
        if denominator_xnp1 == 0:
            print(f"Stopped: denominator zero at x[n+1] step, iteration {i}")
            break
        # Step 3: x[n+1]
        x_new = z - (fz / fdefxi) * ((fxi - fw) / denominator_xnp1)

        x.append(x_new)
        err = abs(x_new - x[i])
        error.append(err)
        i += 1
        iteration.append(i)
        if err < tol:
            break

    six_or = pd.DataFrame({
        "iteration": iteration,
        "x": x,
        "error": error
    })
    return six_or

initial_guesses = [0.25, 0.5, 1.0]
for x0 in initial_guesses:
    print(f"\nTrying initial guess x0 = {x0}:")
    result_table = six_order_method(x0, 1e-6, 100)
    print(result_table)