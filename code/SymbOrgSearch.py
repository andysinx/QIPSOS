from benchmark_fun import *
import pandas as pd
import os

class SOS:
    def __init__(self, func, bounds, dim, pop_size=30, max_iter=50, seed=None):
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

        # 🔥 HISTORY PER PLOTTING
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
                # MUTUALISM
                # =====================
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                xj = self.pop[j].copy()

                bf1, bf2 = np.random.randint(1, 3, size=2)
                mv = (xi + xj) / 2

                xi_new = xi + np.random.rand(self.dim) * (self.best - mv * bf1)
                xj_new = xj + np.random.rand(self.dim) * (self.best - mv * bf2)

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
                # COMMENSALISM
                # =====================
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                xj = self.pop[j].copy()

                xi_new = xi + (np.random.rand(self.dim) * 2 - 1) * (self.best - xj)
                xi_new = self._clip(xi_new)

                fi = self.func(xi_new)

                if fi < self.fitness[i]:
                    self.pop[i] = xi_new
                    self.fitness[i] = fi

                # =====================
                # PARASITISM
                # =====================
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                parasite = xi.copy()

                n_change = np.random.randint(1, self.dim + 1)
                idx = np.random.choice(self.dim, n_change, replace=False)

                for k in idx:
                    parasite[k] = np.random.uniform(
                        self.bounds[k][0], self.bounds[k][1]
                    )

                parasite = self._clip(parasite)

                fp = self.func(parasite)

                if fp < self.fitness[j]:
                    self.pop[j] = parasite
                    self.fitness[j] = fp

            # =====================
            # UPDATE BEST
            # =====================
            self._update_best()

            # =====================
            # HISTORY (IMPORTANT FIX)
            # =====================
            self.history_best.append(self.best_fitness)
            self.history_mean.append(np.mean(self.fitness))
            self.history_position.append(self.best.copy())

        return self.best, self.best_fitness, self.history_best, self.history_position
    

def run_benchmark():

    max_iter = 15#20
    pop_size = 10#15
    N_RUNS = 4#20

    results = []
    all_histories = []

    os.makedirs("results_4run", exist_ok=True)
    problems = benchmark_problems()
    for name, (func, dim, bounds) in problems.items():

        results = []
        all_histories = []
        all_positions = []

        for r in range(N_RUNS):

            seed = 42 * (r + 1)

            algo = SOS(
                func=func,
                bounds=np.array(bounds * dim),
                dim=dim,
                pop_size=pop_size,
                max_iter=max_iter,
                seed=seed
            )

            _, best_f, history, position_history = algo.optimize()

            results.append({
                "problem": name,
                "run": r,
                "seed": seed,
                "best": best_f
            })

            all_histories.append(history)
            all_positions.append(position_history)

             # salva trajectory singola seed
            np.save(
                f"results_4run/{name}_run{r}_history.npy",
                np.array(history)
            )
            np.save(
                f"results_4run/{name}_run{r}_position_history.npy",
                np.array(position_history)
            )

        # 🔥 QUI CSV PER PROBLEMA
        df = pd.DataFrame(results)
        df.to_csv(f"results_4run/{name}_results.csv", index=False)

        summary = df["best"].agg(["mean", "std", "min"])
        summary.to_csv(f"results_4run/{name}_summary.csv")

        # 🔥 QUI AGGREGAZIONE CORRETTA
        H = np.array(all_histories)

        mean_curve = np.mean(H, axis=0)
        std_curve = np.std(H, axis=0)

        np.save(f"results_4run/{name}_mean_curve.npy", mean_curve)
        np.save(f"results_4run/{name}_std_curve.npy", std_curve)

        print(name)
        print(summary)

run_benchmark()