from SOS_variants_qvsc import *
from benchmark_fun import *
import pandas as pd
import os
from tqdm import tqdm
from quantum_sampler_real_backend import *
import numpy as np

def cantilever_beam(x):
    x1, x2, x3, x4, x5 = x

    # objective (weight)
    f = 0.0624 * (x1 + x2 + x3 + x4 + x5)

    # constraint SOS paper
    g = (
        62 / x1**3 +
        37 / x2**3 +
        19 / x3**3 +
        7  / x4**3 +
        1  / x5**3
    )

    # violation (only if > 1)
    violation = max(0, g - 1)

    penalty = 1e3 * violation

    return f + penalty

def benchmark_problems():
    return {
        "CantileverBeam": (
            cantilever_beam,    
            5, [(0.01, 100)] * 5
        )
    }
    

def run_benchmark():

    max_iter = 50
    pop_size = 20
    N_RUNS = 25

    os.makedirs("results_cantilever_noisyPHQSOS_final", exist_ok=True)

    problems = benchmark_problems()

    for name, (func, dim, bounds) in tqdm(problems.items(), desc="Problems",  position=0):

        results = []
        all_histories = []

        bounds = np.array(bounds)  # 🔥 FIX QUI

        for r in tqdm(range(N_RUNS), desc=f"{name} runs", position=1):

            seed = 42 * (r + 1)

            algo = PHSOS( #SOS
                func=func,
                bounds=bounds,   # 🔥 NO * dim
                dim=dim,
                pop_size=pop_size,
                max_iter=max_iter,
                seed=seed
            )

            best_x, best_f, history, _ = algo.optimize()

            best_x = np.array(best_x)

            results.append({
                "run": r,
                "seed": seed,
                "best_f": best_f,
                **{f"x{i+1}": best_x[i] for i in range(dim)},
                "evals": max_iter * pop_size
            })

            all_histories.append(history)

        # =========================
        # TABLE PER PAPER (IMPORTANT)
        # =========================
        df = pd.DataFrame(results)
        df.to_csv(f"results_cantilever_noisyPHQSOS_final/{name}_FULL_TABLE.csv", index=False)

        summary = pd.DataFrame([{
            "mean": df["best_f"].mean(),
            "std": df["best_f"].std(),
            "best": df["best_f"].min()
        }])

        summary.to_csv(f"results_cantilever_noisyPHQSOS_final/{name}_SUMMARY.csv", index=False)

        # =========================
        # CURVE
        # =========================
        H = np.array(all_histories)
        np.save(f"results_cantilever_noisyPHQSOS_final/{name}_mean_curve.npy", H.mean(axis=0))
        np.save(f"results_cantilever_noisyPHQSOS_final/{name}_std_curve.npy", H.std(axis=0))

        print(name)
        print(summary)



def run_benchmark_realdevice():

    max_iter = 40
    pop_size = 15
    N_RUNS = 20

    os.makedirs("results_cantilever_PHQSOS_real_ibm_marrakesh", exist_ok=True)

    problems = benchmark_problems()
    qs = QuantumSamplerIBM(
        backend_name="ibm_marrakesh",
        shots=1048,
        batch_size=10
    )

    for name, (func, dim, bounds) in tqdm(problems.items(), desc="Problems", position=0):

        results = []
        all_histories = []
        bounds = np.array(bounds)

        for r in tqdm(range(N_RUNS), desc=f"{name} runs", position=1):

            seed = 42 * (r + 1)

            # =========================
            # 🔥 REAL PHSOS WITH IBM
            # =========================
            algo = REAL_PHSOS(
                func=func,
                bounds=bounds,
                dim=dim,
                pop_size=pop_size,
                max_iter=max_iter,
                seed=seed,
                qs=qs   # 🔥 PASS IBM SAMPLER
            )

            best_x, best_f, history, _ = algo.optimize()

            best_x = np.array(best_x)

            results.append({
                "run": r,
                "seed": seed,
                "best_f": best_f,
                **{f"x{i+1}": best_x[i] for i in range(dim)},
                "evals": max_iter * pop_size
            })

            all_histories.append(history)

        # =========================
        # TABLE PER PAPER
        # =========================
        df = pd.DataFrame(results)
        df.to_csv(f"results_cantilever_PHQSOS_real_ibm_marrakesh/{name}_FULL_TABLE.csv", index=False)

        summary = pd.DataFrame([{
            "mean": df["best_f"].mean(),
            "std": df["best_f"].std(),
            "best": df["best_f"].min()
        }])

        summary.to_csv(f"results_cantilever_PHQSOS_real_ibm_marrakesh/{name}_SUMMARY.csv", index=False)

        # =========================
        # CURVE
        # =========================
        H = np.array(all_histories)
        np.save(f"results_cantilever_PHQSOS_real_ibm_marrakesh/{name}_mean_curve.npy", H.mean(axis=0))
        np.save(f"results_cantilever_PHQSOS_real_ibm_marrakesh/{name}_std_curve.npy", H.std(axis=0))

        print(name)
        print(summary)

    # 🔥 CHIUSURA SESSIONE IBM
    qs.close()

run_benchmark()
#run_benchmark_realdevice()