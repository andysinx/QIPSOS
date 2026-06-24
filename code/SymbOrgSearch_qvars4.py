from quantum_sampler import *
from benchmark_fun import *
import pandas as pd
import os
from tqdm import tqdm
import numpy as np


class Hybrid_SOS:
    def __init__(self, func, bounds, dim, pop_size=15, max_iter=80, seed=None):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter

        if seed is not None:
            np.random.seed(seed)

        self.pop = self._init_population()
        self.fitness = np.array([self.func(x) for x in self.pop])

        self.best_idx = np.argmin(self.fitness)
        self.best = self.pop[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]

        # HISTORY
        self.history_best = []
        self.history_mean = []
        self.history_position = []

    def _init_population(self):
        return np.array([
            np.array([
                np.random.uniform(self.bounds[i][0], self.bounds[i][1])
                for i in range(self.dim)
            ])
            for _ in range(self.pop_size)
        ])

    def _clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _update_best(self):
        idx = np.argmin(self.fitness)
        if self.fitness[idx] < self.best_fitness:
            self.best_fitness = self.fitness[idx]
            self.best = self.pop[idx].copy()

    def optimize(self):

        for _ in range(self.max_iter):

            for i in range(self.pop_size):

                xi = self.pop[i].copy()

                # =====================
                # MUTUALISM (QUANTUM)
                # =====================
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                xj = self.pop[j].copy()

                bf1, bf2 = np.random.randint(1, 3, size=2)
                mv = (xi + xj) / 2

                xi_new = xi.copy()
                xj_new = xj.copy()

                # --- subset selection ---
                if self.dim > 20:
                    max_change = self.dim // 3
                elif self.dim <=2:
                    max_change = self.dim *3
                else:
                    max_change = self.dim

                n_change = np.random.randint(1, max_change + 1)
                idx = np.random.choice(self.dim, n_change, replace=False)

                # --- quantum sampling ---
                theta = mv * np.pi#np.random.uniform(0, 2*np.pi, n_change)
                q = quantum_sample(theta)   # {-1, +1}

                # mapping to (0,1)
                r = (q + 1) / 2
                r = np.clip(r, 0.0, 1.0)

                # stabilization (important)
                r = 0.8 * r + 0.2 * np.random.rand(n_change)

                # --- update rule ---
                for k, d in enumerate(idx):
                    xi_new[d] = xi[d] + r[k] * (self.best[d] - mv[d] * bf1)
                    xj_new[d] = xj[d] + r[k] * (self.best[d] - mv[d] * bf2)

                xi_new = self._clip(xi_new)
                xj_new = self._clip(xj_new)

                fi = self.func(xi_new)
                fj = self.func(xj_new)

                if fi < self.fitness[i]:
                    self.pop[i] = xi_new
                    self.fitness[i] = fi

                if fj < self.fitness[j]:
                    self.pop[j] = xj_new
                    self.fitness[j] = fj

                # =====================
                # COMMENSALISM (QUANTUM)
                # =====================
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                xj = self.pop[j].copy()

                if self.dim > 20:
                    max_change = self.dim // 3
                elif self.dim <=2:
                    max_change = self.dim *3
                else:
                    max_change = self.dim

                n_change = np.random.randint(1, max_change + 1)
                idx = np.random.choice(self.dim, n_change, replace=False)

                theta = np.random.uniform(0, 2*np.pi, n_change)
                u_q = quantum_sample(theta)

                alpha = 0.5

                xi_new = xi.copy()

                for k, d in enumerate(idx):
                    xi_new[d] = xi[d] + alpha * u_q[k] * (self.best[d] - xj[d])

                xi_new = self._clip(xi_new)

                fi = self.func(xi_new)

                if fi < self.fitness[i]:
                    self.pop[i] = xi_new
                    self.fitness[i] = fi

                # =====================
                # PARASITISM (QUANTUM)
                # =====================
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                xi = self.pop[i].copy()

                if self.dim > 20:
                    max_change = self.dim // 3
                elif self.dim <=2:
                    max_change = self.dim *3
                else:
                    max_change = self.dim

                n_change = np.random.randint(1, max_change + 1)
                idx = np.random.choice(self.dim, n_change, replace=False)

                theta = xi[idx] * np.pi
                z = quantum_sample(theta)

                perturb = decode_quantum(z, self.bounds[idx])

                xq = xi.copy()
                xq[idx] = xi[idx] + 0.1 * (
                    perturb - 0.5 * (self.bounds[idx, 1] + self.bounds[idx, 0])
                )

                xq = self._clip(xq)

                fp = self.func(xq)

                if fp < self.fitness[j]:
                    self.pop[j] = xq
                    self.fitness[j] = fp

            # =====================
            # UPDATE BEST
            # =====================
            self._update_best()

            self.history_best.append(self.best_fitness)
            self.history_mean.append(np.mean(self.fitness))
            self.history_position.append(self.best.copy())

        return self.best, self.best_fitness, self.history_best, self.history_position


# =====================
# BENCHMARK
# =====================
def run_benchmark():

    max_iter = 15#80
    pop_size = 10#15
    N_RUNS =4#20

    os.makedirs("results_HEA5_qvars4_noisy", exist_ok=True)
    problems = benchmark_problems()

    for name, (func, dim, bounds) in tqdm(problems.items(), desc="Problems",  position=0):

        results = []
        all_histories = []
        all_positions = []

        for r in tqdm(range(N_RUNS), desc=f"{name} runs", position=1):

            seed = 42 * (r + 1)

            algo = Hybrid_SOS(
                func=func,
                bounds=np.array(bounds * dim),
                dim=dim,
                pop_size=pop_size,
                max_iter=max_iter,
                seed=seed
            )

            best_x, best_f, history, position_history = algo.optimize()

            results.append({
                "problem": name,
                "run": r,
                "seed": seed,
                "best": best_f
            })

            all_histories.append(history)
            all_positions.append(position_history)

            np.save(f"results_HEA5_qvars4_noisy/{name}_run{r}_history.npy", np.array(history))
            np.save(f"results_HEA5_qvars4_noisy/{name}_run{r}_position.npy", np.array(position_history))

        df = pd.DataFrame(results)
        df.to_csv(f"results_HEA5_qvars4_noisy/{name}_results.csv", index=False)

        summary = df["best"].agg(["mean", "std", "min"])
        summary.to_csv(f"results_HEA5_qvars4_noisy/{name}_summary.csv")

        H = np.array(all_histories)
        np.save(f"results_HEA5_qvars4_noisy/{name}_mean_curve.npy", np.mean(H, axis=0))
        np.save(f"results_HEA5_qvars4_noisy/{name}_std_curve.npy", np.std(H, axis=0))

        print(name)
        print(summary)


if __name__ == "__main__":
    run_benchmark()