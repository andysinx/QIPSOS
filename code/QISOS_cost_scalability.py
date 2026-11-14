#!/usr/bin/env python3
"""QIP-SOS ONLY: matched settings for previously completed SOS / QE-SOS studies.

Requires benchmark_fun.py for --study cost. No Qiskit needed.
The QISOS core is derived from the user-supplied implementation; default mode
p_only retains classical mutualism/commensalism and analytic parasitism.
Timing includes population initialization and optimization; separate sampler
and objective evaluation times are instrumented.
"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np

def probability_one(theta):
    """Born probability for |0> -> RY(t),RZ(t),RY(t),RZ(t) -> measurement.

    No initial Hadamard and no entangling gates.
    """
    theta = np.asarray(theta, dtype=float)
    return np.clip(np.sin(theta)**2 * np.cos(theta / 2)**2, 0.0, 1.0)


def analytic_sample(theta, rng):
    """Sample independent signs, with measurement bit 1 mapped to +1."""
    p = probability_one(theta)
    return np.where(rng.random(p.shape) < p, 1, -1)


class QISOS:
    MODES = ('full', 'p_only', 'c_only')

    def __init__(self, func, bounds, dim, pop_size=15, max_iter=40,
                 seed=None, mode='full', max_change=6,
                 commensalism_alpha=0.5, parasitism_alpha=0.1):
        if mode not in self.MODES:
            raise ValueError(f'mode must be one of {self.MODES}')
        self.func = func
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = int(dim)
        self.pop_size = int(pop_size)
        self.max_iter = int(max_iter)
        self.mode = mode
        self.max_change = min(int(max_change), self.dim)
        self.commensalism_alpha = float(commensalism_alpha)
        self.parasitism_alpha = float(parasitism_alpha)
        if self.bounds.shape != (self.dim, 2) or np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError('Invalid bounds')
        if self.pop_size < 2 or self.dim < 1 or self.max_change < 1 or self.max_iter < 0:
            raise ValueError('Invalid population, dimension, iterations or max_change')
        # Legacy RandomState reproduces the SOS/QE-SOS numpy.random draw semantics.
        # Local RNG avoids changing the global NumPy random state.
        self.rng = np.random.RandomState(seed)
        self.nfe = 0
        self.objective_seconds = 0.0
        self.sampler_seconds = 0.0
        self.sampler_calls = 0  # Classical sampling events, NOT quantum backend calls
        self.runtime_seconds = 0.0
        self.pop = self._init_population()
        self.fitness = np.array([self._evaluate(x) for x in self.pop])
        self.best_idx = int(np.argmin(self.fitness))
        self.best = self.pop[self.best_idx].copy()
        self.best_fitness = float(self.fitness[self.best_idx])
        self.history_best = []
        self.history_mean = []
        self.history_position = []
        self.history_nfe = []

    def _evaluate(self, x):
        self.nfe += 1
        t0 = time.perf_counter()
        result = float(self.func(x))
        self.objective_seconds += time.perf_counter() - t0
        return result

    def _init_population(self):
        return np.array([[self.rng.uniform(*self.bounds[k]) for k in range(self.dim)]
                         for _ in range(self.pop_size)])

    def _clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _other(self, i):
        j = self.rng.randint(self.pop_size)
        while j == i:
            j = self.rng.randint(self.pop_size)
        return j

    def _selected_indices(self):
        n = self.rng.randint(1, self.max_change + 1)
        return self.rng.choice(self.dim, n, replace=False)

    def _update_best(self):
        idx = int(np.argmin(self.fitness))
        if self.fitness[idx] < self.best_fitness:
            self.best_fitness = float(self.fitness[idx])
            self.best = self.pop[idx].copy()

    def optimize(self):
        start = time.perf_counter()
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                xi = self.pop[i].copy()
                # MUTUALISM -- same as supplied SOS and QE-SOS
                j = self._other(i)
                xj = self.pop[j].copy()
                bf1, bf2 = self.rng.randint(1, 3, size=2)
                mv = (xi + xj) / 2
                xi_new = self._clip(xi + self.rng.rand(self.dim) * (self.best - mv * bf1))
                xj_new = self._clip(xj + self.rng.rand(self.dim) * (self.best - mv * bf2))
                fi, fj = self._evaluate(xi_new), self._evaluate(xj_new)
                if fi < self.fitness[i]:
                    self.pop[i], self.fitness[i] = xi_new, fi
                if fj < self.fitness[j]:
                    self.pop[j], self.fitness[j] = xj_new, fj

                # COMMENSALISM -- preserve original QE-SOS alpha=0.5 and dimension selection
                j = self._other(i)
                xj = self.pop[j].copy()
                if self.mode in ('full', 'c_only'):
                    idx = self._selected_indices()
                    # Integrating theta~U(0,2pi) yields independent Bernoulli(0.25).
                    # Equivalent in distribution, not an identical RNG trajectory.
                    t0 = time.perf_counter()
                    signs = np.where(self.rng.rand(len(idx)) < 0.25, 1, -1)
                    self.sampler_seconds += time.perf_counter() - t0
                    self.sampler_calls += 1
                    xi_new = xi.copy()
                    xi_new[idx] = xi[idx] + self.commensalism_alpha * signs * (self.best[idx] - xj[idx])
                else:
                    xi_new = xi + (self.rng.rand(self.dim) * 2 - 1) * (self.best - xj)
                xi_new = self._clip(xi_new)
                fi = self._evaluate(xi_new)
                if fi < self.fitness[i]:
                    self.pop[i], self.fitness[i] = xi_new, fi

                # PARASITISM -- QE-SOS's state-dependent angle and boundary decoding
                j = self._other(i)
                xi = self.pop[i].copy() if self.mode in ('full', 'p_only') else xi  # QE-SOS refreshes; classical SOS retains snapshot
                if self.mode in ('full', 'p_only'):
                    idx = self._selected_indices()
                    lower, upper = self.bounds[idx, 0], self.bounds[idx, 1]
                    theta = np.pi * (xi[idx] - lower) / (upper - lower)
                    t0 = time.perf_counter()
                    signs = analytic_sample(theta, self.rng)
                    self.sampler_seconds += time.perf_counter() - t0
                    self.sampler_calls += 1
                    # decode(+1)=upper; decode(-1)=lower, then subtract midpoint
                    # This is exactly 0.5 * (upper-lower) * sign.
                    shift = self.parasitism_alpha * 0.5 * (upper - lower) * signs
                    parasite = xi.copy()
                    parasite[idx] = xi[idx] + shift
                else:
                    parasite = xi.copy()  # Original SOS uses stale xi snapshot
                    n_change = self.rng.randint(1, self.dim + 1)
                    idx = self.rng.choice(self.dim, n_change, replace=False)
                    for k in idx:
                        parasite[k] = self.rng.uniform(*self.bounds[k])
                parasite = self._clip(parasite)
                fp = self._evaluate(parasite)
                if fp < self.fitness[j]:
                    self.pop[j], self.fitness[j] = parasite, fp
            self._update_best()
            self.history_best.append(self.best_fitness)
            self.history_mean.append(float(np.mean(self.fitness)))
            self.history_position.append(self.best.copy())
            self.history_nfe.append(self.nfe)
        self.runtime_seconds = time.perf_counter() - start
        return self.best, self.best_fitness, self.history_best, self.history_position




def experiments(args):
    if args.study == 'cost':
        from benchmark_fun import benchmark_problems
        cases = [(name, fun, dim, bounds) for name, (fun, dim, bounds) in benchmark_problems().items()]
    else:
        def sphere(x): return float(np.sum(x*x))
        def rastrigin(x): return float(np.sum(x*x - 10*np.cos(2*np.pi*x) + 10))
        cases = [(name, fun, d, [bounds]*d)
                 for name, fun, bounds in [('Sphere', sphere, (-100, 100)),
                                            ('Rastrigin', rastrigin, (-5.12, 5.12))]
                 for d in args.dimensions]
    out = Path(args.output_root) / args.study
    out.mkdir(parents=True, exist_ok=True)
    expected_nfe = args.population * (1 + 4 * args.generations)
    rows = []
    for name, fun, dim, bounds in cases:
        for r in range(args.runs):
            seed = 42 * (r + 1)
            t0 = time.perf_counter()
            algo = QISOS(fun, bounds, dim, pop_size=args.population,
                         max_iter=args.generations, seed=seed, mode='p_only',
                         max_change=args.max_change,
                         parasitism_alpha=args.parasitism_alpha)
            init_seconds = time.perf_counter() - t0
            _, best, _, _ = algo.optimize()
            total_seconds = time.perf_counter() - t0
            assert algo.nfe == expected_nfe, (name, dim, seed, algo.nfe, expected_nfe)
            assert algo.sampler_calls == args.population * args.generations
            row = dict(function=name, dimension=dim, algorithm='QIP-SOS', seed=seed,
                       population=args.population, generations=args.generations,
                       nfe=algo.nfe, quantum_calls=0, shots=0,
                       runtime_seconds=total_seconds, quantum_seconds=0.0,
                       nonquantum_seconds=total_seconds,
                       initialization_seconds=init_seconds, best_fitness=float(best),
                       analytic_sampler_calls=algo.sampler_calls,
                       analytic_sampler_seconds=algo.sampler_seconds,
                       objective_seconds=algo.objective_seconds,
                       overhead_seconds=max(0., total_seconds - algo.sampler_seconds - algo.objective_seconds))
            rows.append(row)
            print(f'{args.study} {name:15s} D={dim:3d} run={r+1:2d}/{args.runs} '
                  f'best={best:.6g} time={total_seconds:.5f}s sampler={algo.sampler_seconds:.5f}s', flush=True)
            # Save progress after every run, for safe interruption/restart inspection.
            with (out / 'all_runs.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['function'], row['dimension']), []).append(row)
    with (out / 'summary.csv').open('w', newline='') as f:
        fields = ['function', 'dimension', 'algorithm', 'runs', 'nfe',
                  'runtime_mean', 'runtime_std', 'sampler_mean', 'sampler_std',
                  'best_mean', 'best_std']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for (name, dim), items in groups.items():
            def stats(key):
                a = np.array([x[key] for x in items]); return float(a.mean()), float(a.std(ddof=1)) if len(a)>1 else 0.0
            rt, rts = stats('runtime_seconds')
            st, sts = stats('analytic_sampler_seconds')
            bm, bms = stats('best_fitness')
            w.writerow(dict(function=name, dimension=dim, algorithm='QIP-SOS',
                            runs=len(items), nfe=expected_nfe, runtime_mean=rt,
                            runtime_std=rts, sampler_mean=st, sampler_std=sts,
                            best_mean=bm, best_std=bms))
    metadata = dict(study=args.study, algorithm='QIP-SOS p_only',
                    population=args.population, generations=args.generations,
                    runs=args.runs, dimensions=args.dimensions if args.study=='scale' else None,
                    nfe=expected_nfe, max_change=args.max_change,
                    parasitism_alpha=args.parasitism_alpha,
                    python=platform.python_version(), numpy=np.__version__,
                    platform=platform.platform(),
                    timing='perf_counter; includes initialization and optimization',
                    note='quantum_calls=0; analytic_sampler_calls is classical, not quantum')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'\nSaved {len(rows)} runs to {out.resolve()}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='QIP-SOS only computational cost and scalability')
    ap.add_argument('--study', choices=['cost','scale'], required=True)
    ap.add_argument('--runs', type=int, default=None)
    ap.add_argument('--population', type=int, default=15)
    ap.add_argument('--generations', type=int, default=None)
    ap.add_argument('--dimensions', type=int, nargs='+', default=[10,30,50,100,200])
    ap.add_argument('--max-change', type=int, default=6)
    ap.add_argument('--parasitism-alpha', type=float, default=0.1)
    ap.add_argument('--output-root', default='results_QIP_cost_scalability')
    args = ap.parse_args()
    if args.runs is None: args.runs = 20 if args.study=='cost' else 3
    if args.generations is None: args.generations = 40 if args.study=='cost' else 10
    if args.runs < 1 or args.population < 2 or args.generations < 1 or min(args.dimensions)<1:
        ap.error('Invalid experimental parameters')
    experiments(args)