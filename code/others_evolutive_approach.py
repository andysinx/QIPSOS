import numpy as np
import pandas as pd
import os

from benchmark_fun import *


# =========================================================
# MEALPY IMPORTS (CORRECT 3.0.3)
# =========================================================

from mealpy import GA, DE, PSO, MA,  FloatVar
from mealpy.bio_based.VCS import OriginalVCS
from mealpy.bio_based.BBOA import OriginalBBOA
from mealpy.bio_based.BMO import OriginalBMO
from mealpy.bio_based.SBO import OriginalSBO
from mealpy.bio_based.SOA import OriginalSOA
from mealpy.human_based.SPBO import DevSPBO
from mealpy.math_based.HC import OriginalHC
from mealpy.math_based.HC import SwarmHC
from mealpy.math_based.TS import OriginalTS
from mealpy.swarm_based.ABC import OriginalABC
from mealpy.swarm_based.BA import OriginalBA
from mealpy.swarm_based.COA import OriginalCOA
from mealpy.swarm_based.CSA import OriginalCSA
from mealpy.swarm_based.FOA import OriginalFOA
from mealpy.swarm_based.PSO import C_PSO
from mealpy.swarm_based.TDO import OriginalTDO
from mealpy.swarm_based.ZOA import OriginalZOA
# =========================================================
# PROBLEM BUILDER (Mealpy format)
# =========================================================

def build_problem(func, bounds):
    return {
        "obj_func": func,
        "bounds": FloatVar(
            lb=[b[0] for b in bounds],
            ub=[b[1] for b in bounds]
        ),
        "minmax": "min"
    }


# =========================================================
# RESULT CONTAINER
# =========================================================

class Result:
    def __init__(self, best, fitness, history):
        self.best = best
        self.fitness = fitness
        self.history = history


# =========================================================
# RUNNER (NO CALLBACK, USE INTERNAL HISTORY)
# =========================================================

def run_algorithm(model_class, func, bounds, seed, pop_size, epoch):

    problem = build_problem(func, bounds)

    model = model_class(epoch=epoch, pop_size=pop_size)

    g_best = model.solve(
        problem,
        mode="single",
        seed=seed
    )

    # Mealpy history (correct way)
    history = model.history.list_global_best_fit

    return Result(
        g_best.solution,
        g_best.target.fitness,
        history
    )


# =========================================================
# WRAPPERS
# =========================================================

def GA_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(GA.BaseGA, func, bounds, seed, pop_size, epoch)

def MA_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(MA.OriginalMA, func, bounds, seed, pop_size, epoch)

def DE_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(DE.OriginalDE, func, bounds, seed, pop_size, epoch)

def PSO_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(PSO.OriginalPSO, func, bounds, seed, pop_size, epoch)

def VCS_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalVCS, func, bounds, seed, pop_size, epoch)

def BBOA_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalBBOA, func, bounds, seed, pop_size, epoch)

def BMO_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalBMO, func, bounds, seed, pop_size, epoch)

def SBO_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalSBO, func, bounds, seed, pop_size, epoch)

def SOA_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalSOA, func, bounds, seed, pop_size, epoch)  

def DEVSPBO_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(DevSPBO, func, bounds, seed, pop_size, epoch)

def HC_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalHC, func, bounds, seed, pop_size, epoch)

def SwarmHC_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(SwarmHC, func, bounds, seed, pop_size, epoch)

def TS_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalTS, func, bounds, seed, pop_size, epoch)

def ABC_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalABC, func, bounds, seed, pop_size, epoch)

def  BA_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalBA, func, bounds, seed, pop_size, epoch)

def COA_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalCOA, func, bounds, seed, pop_size, epoch)

def CSA_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalCSA, func, bounds, seed, pop_size, epoch)

def FOA_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalFOA, func, bounds, seed, pop_size, epoch)

def C_PSO_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(C_PSO, func, bounds, seed, pop_size, epoch)

def TDO_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalTDO, func, bounds, seed, pop_size, epoch)

def ZOA_ALG(func, bounds, dim, seed, pop_size, epoch):
    return run_algorithm(OriginalZOA, func, bounds, seed, pop_size, epoch)

'''"GA": GA_ALG,
"DE": DE_ALG,
"PSO": PSO_ALG,
"VCS": VCS_ALG,
"BBOA": BBOA_ALG,
"BMO": BMO_ALG,
"SBO": SBO_ALG,
"SOA": SOA_ALG'''
'''"DEVSPBO_ALG": DEVSPBO_ALG,
"MA": MA_ALG,
"HC": HC_ALG,
"SwarmHC": SwarmHC_ALG,
"TS": TS_ALG,
"ABC": ABC_ALG,
"BA": BA_ALG,
"COA": COA_ALG,
"CSA": CSA_ALG,
"FOA": FOA_ALG,
"C_PSO": C_PSO_ALG,
"TDO": TDO_ALG,
"ZOA": ZOA_ALG'''

def run_benchmark():
    algorithms = {
        "VCS": VCS_ALG,
        "BBOA": BBOA_ALG,
        "BMO": BMO_ALG,
        "COA": COA_ALG,
        "TDO": TDO_ALG,
        "ZOA": ZOA_ALG
    }

    POP_SIZE = 15#15
    EPOCH = 15#20
    N_RUNS = 4#20

    problems = benchmark_problems1()

    os.makedirs("results_othersapproach2_4runs", exist_ok=True)

    for name, (func, dim, bounds) in problems.items():

        print("\n===================================")
        print("PROBLEM:", name)
        print("===================================")

        all_results = []
        all_histories = {k: [] for k in algorithms}

        for r in range(N_RUNS):

            seed = 42 * (r + 1)

            for alg_name, alg_fn in algorithms.items():

                res = alg_fn(
                    func,
                    bounds,
                    dim,
                    seed,
                    POP_SIZE,
                    EPOCH
                )

                all_results.append({
                    "problem": name,
                    "algorithm": alg_name,
                    "run": r,
                    "best": res.fitness
                })

                all_histories[alg_name].append(res.history)

                print(f"{name} | {alg_name} | run {r} | best {res.fitness:.6f}")

        # SAVE
        df = pd.DataFrame(all_results)
        df.to_csv(f"results_othersapproach2_4runs/{name}_results.csv", index=False)

        summary = df.groupby("algorithm")["best"].agg(["mean", "std", "min"])
        summary.to_csv(f"results_othersapproach2_4runs/{name}_summary.csv")

        for alg, hist in all_histories.items():
            np.save(
                f"results_othersapproach2_4runs/{name}_{alg}_history.npy",
                np.array(hist)
            )

        print("\nSUMMARY:")
        print(summary)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    run_benchmark()