
import os
import time
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm
from mealpy import GA, DE, PSO, MA, FloatVar

from benchmark_fun import benchmark_problems


# =========================================================
# EXPERIMENT CONFIGURATION
# =========================================================

POP_SIZE = 16
N_RUNS = 20
MAX_NFE = 2415
MAX_EPOCH = 5000

REFERENCE_POP_SIZE = 15
REFERENCE_GENERATIONS = 40

CHECKPOINTS = (
    REFERENCE_POP_SIZE
    + 4 * REFERENCE_POP_SIZE
    * np.arange(REFERENCE_GENERATIONS + 1)
)

OUTPUT_DIR = "results_classical_equalNFE_40iter"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Disable Mealpy per-epoch INFO messages
logging.getLogger("mealpy").setLevel(logging.WARNING)


# =========================================================
# ALGORITHMS
# =========================================================

ALGORITHMS = {
    "GA": GA.BaseGA,
    "DE": DE.OriginalDE,
    "PSO": PSO.OriginalPSO,
    "MA": MA.OriginalMA,
}


# =========================================================
# FITNESS EVALUATION BUDGET
# =========================================================

class NFEBudgetReached(Exception):
    pass


class BudgetedObjective:

    def __init__(self, func, max_nfe=MAX_NFE):

        self.func = func
        self.max_nfe = max_nfe

        self.nfe = 0

        self.best_fitness = np.inf
        self.best_position = None

        self.checkpoints = CHECKPOINTS.copy()

        self.history = []
        self.history_nfe = []

    def __call__(self, x):

        if self.nfe >= self.max_nfe:
            raise NFEBudgetReached()

        x = np.asarray(x, dtype=float)

        fitness = float(self.func(x))

        self.nfe += 1

        if np.isfinite(fitness):
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_position = x.copy()

        # Convergence at quantum SOS NFE checkpoints
        next_idx = len(self.history_nfe)

        if next_idx < len(self.checkpoints):

            if self.nfe == self.checkpoints[next_idx]:

                self.history_nfe.append(self.nfe)

                self.history.append(
                    self.best_fitness
                )

        return fitness


# =========================================================
# PROBLEM BUILDER
# =========================================================

def build_problem(objective, bounds):

    bounds = np.asarray(bounds, dtype=float)

    return {
        "obj_func": objective,
        "bounds": FloatVar(
            lb=bounds[:, 0].tolist(),
            ub=bounds[:, 1].tolist()
        ),
        "minmax": "min"
    }


# =========================================================
# RUN SINGLE ALGORITHM
# =========================================================

def run_algorithm(
    model_class,
    func,
    bounds,
    seed,
    pop_size=POP_SIZE,
    max_nfe=MAX_NFE
):

    objective = BudgetedObjective(
        func=func,
        max_nfe=max_nfe
    )

    problem = build_problem(
        objective,
        bounds
    )

    model_kwargs = {
        "epoch": MAX_EPOCH,
        "pop_size": pop_size
    }

    if model_class is MA.OriginalMA:
        model_kwargs["max_local_gens"] = 5

    model = model_class(**model_kwargs)

    start_time = time.perf_counter()

    try:

        model.solve(
            problem,
            mode="single",
            seed=seed
        )

    except NFEBudgetReached:
        # Normal termination at the evaluation limit
        pass

    runtime = time.perf_counter() - start_time

    # =====================================
    # VALIDATION
    # =====================================

    if objective.nfe != max_nfe:
        raise RuntimeError(
            f"Incorrect NFE: {objective.nfe}, "
            f"expected {max_nfe}"
        )

    if objective.best_position is None:
        raise RuntimeError(
            "No finite fitness value was found"
        )

    if len(objective.history) != len(CHECKPOINTS):
        raise RuntimeError(
            "Incomplete convergence history: "
            f"{len(objective.history)} / "
            f"{len(CHECKPOINTS)}"
        )

    if not np.array_equal(
        objective.history_nfe,
        CHECKPOINTS
    ):
        raise RuntimeError(
            "Incorrect convergence checkpoints"
        )

    return {
        "best": objective.best_fitness,
        "best_position": objective.best_position,
        "nfe": objective.nfe,
        "history": np.asarray(
            objective.history,
            dtype=float
        ),
        "history_nfe": np.asarray(
            objective.history_nfe,
            dtype=int
        ),
        "runtime_seconds": runtime
    }


# =========================================================
# SAVE RESULTS FOR ONE PROBLEM
# =========================================================

def save_problem_results(
    name,
    all_results,
    all_histories,
    all_positions
):

    df = pd.DataFrame(all_results)

    # Raw results
    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{name}_classical_equalNFE_results.csv"
        ),
        index=False
    )

    # Summary, including only algorithms with results
    summary = df.groupby("algorithm")["best"].agg(
        ["mean", "std", "min"]
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{name}_classical_equalNFE_summary.csv"
        )
    )

    # Convergence curves
    for alg_name in ALGORITHMS:

        if not all_histories[alg_name]:
            continue

        H = np.asarray(
            all_histories[alg_name],
            dtype=float
        )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"{name}_{alg_name}_mean_curve.npy"
            ),
            np.mean(H, axis=0)
        )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"{name}_{alg_name}_std_curve.npy"
            ),
            np.std(H, axis=0)
        )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"{name}_{alg_name}_nfe_axis.npy"
            ),
            CHECKPOINTS
        )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"{name}_{alg_name}_best_positions.npy"
            ),
            np.asarray(all_positions[alg_name])
        )

    return summary


# =========================================================
# MAIN BENCHMARK
# =========================================================

def run_benchmark():

    problems = benchmark_problems()

    print("====================================")
    print("CLASSICAL BASELINES - EQUAL NFE")
    print("====================================")
    print(f"Algorithms: {list(ALGORITHMS)}")
    print(f"Population: {POP_SIZE}")
    print(f"Runs: {N_RUNS}")
    print(f"NFE/run: {MAX_NFE}")
    print(f"Checkpoints: {len(CHECKPOINTS)}")
    print(f"Output: {OUTPUT_DIR}")
    print("====================================")

    for name, (func, dim, bounds) in tqdm(
        problems.items(),
        desc="Benchmark problems"
    ):

        bounds = np.asarray(bounds, dtype=float)

        if bounds.shape != (dim, 2):
            raise ValueError(
                f"{name}: bounds shape {bounds.shape}, "
                f"expected ({dim}, 2)"
            )

        all_results = []

        all_histories = {
            alg: [] for alg in ALGORITHMS
        }

        all_positions = {
            alg: [] for alg in ALGORITHMS
        }

        for r in tqdm(
            range(N_RUNS),
            desc=f"{name} runs",
            leave=False
        ):

            seed = 42 * (r + 1)

            for alg_name, model_class in ALGORITHMS.items():

                result = run_algorithm(
                    model_class=model_class,
                    func=func,
                    bounds=bounds,
                    seed=seed
                )

                all_results.append({
                    "problem": name,
                    "algorithm": alg_name,
                    "run": r,
                    "seed": seed,
                    "best": result["best"],
                    "nfe": result["nfe"],
                    "runtime_seconds":
                        result["runtime_seconds"]
                })

                all_histories[alg_name].append(
                    result["history"]
                )

                all_positions[alg_name].append(
                    result["best_position"]
                )

                # Individual run history
                np.save(
                    os.path.join(
                        OUTPUT_DIR,
                        f"{name}_{alg_name}_run{r}_history.npy"
                    ),
                    result["history"]
                )

                # Individual best position
                np.save(
                    os.path.join(
                        OUTPUT_DIR,
                        f"{name}_{alg_name}_run{r}_best_position.npy"
                    ),
                    result["best_position"]
                )

                print(
                    f"{name} | {alg_name} | "
                    f"run {r+1}/{N_RUNS} | "
                    f"best={result['best']:.8g} | "
                    f"NFE={result['nfe']} | "
                    f"time={result['runtime_seconds']:.2f}s",
                    flush=True
                )

                # Save completed results immediately
                save_problem_results(
                    name,
                    all_results,
                    all_histories,
                    all_positions
                )

        summary = save_problem_results(
            name,
            all_results,
            all_histories,
            all_positions
        )

        print(f"\n{name}")
        print(summary)


if __name__ == "__main__":
    run_benchmark()