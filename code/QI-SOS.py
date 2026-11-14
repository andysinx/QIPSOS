"""QI-SOS benchmark and optional sampler validation (No-CNOT analytical model)."""
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from benchmark_fun import benchmark_problems
import time


def matrix_probability(t):
    ry = np.array([[np.cos(t/2), -np.sin(t/2)],
                   [np.sin(t/2), np.cos(t/2)]], dtype=complex)
    rz = np.diag([np.exp(-0.5j*t), np.exp(0.5j*t)])
    state = rz @ ry @ rz @ ry @ np.array([1, 0], dtype=complex)
    return abs(state[1])**2


def validate():
    theta = np.linspace(-3*np.pi, 3*np.pi, 501)
    expected = np.array([matrix_probability(t) for t in theta])
    err = np.max(np.abs(probability_one(theta) - expected))
    assert err < 1e-12, err
    rng = np.random.RandomState(123)
    for t in [0.0, np.pi/3, np.pi/2, np.pi, 1.7*np.pi]:
        samples = analytic_sample(np.full(100000, t), rng)
        observed = np.mean(samples == 1)
        target = float(probability_one(t))
        assert abs(observed-target) < 0.012, (t, observed, target)
    # Numerical quadrature of uniform angle -> Bernoulli(1/4)
    assert abs(np.mean(probability_one(np.linspace(0, 2*np.pi, 200001))) - 0.25) < 1e-5
    for mode, expected_calls in [('full', 2), ('p_only', 1), ('c_only', 1)]:
        algo = QISOS(lambda x: float(np.sum(x*x)), [(-5, 5)]*3, 3,
                     pop_size=5, max_iter=2, seed=42, mode=mode)
        algo.optimize()
        assert algo.nfe == 5 + 4*5*2
        assert algo.sampler_calls == expected_calls * 5 * 2
    print(f'PASS: matrix agreement max error={err:.3e}, Monte Carlo, 1/4 marginal, NFE, modes')
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ImportError:
        print('Qiskit not installed: optional independent check skipped')
        return
    for t in [0.1, 0.7, 1.2, 2.6]:
        qc = QuantumCircuit(1)
        for _ in range(2):
            qc.ry(t, 0)
            qc.rz(t, 0)
        p1 = Statevector.from_instruction(qc).probabilities()[1]
        assert abs(p1 - probability_one(t)) < 1e-12
    print('PASS: Qiskit Statevector cross-check')


"""QI-SOS operator ablation. Only NumPy is required for optimization.

full: analytic C + analytic P; p_only: original SOS C + analytic P;
c_only: analytic C + original SOS P.
"""

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
        return float(self.func(x))

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
                    signs = np.where(self.rng.rand(len(idx)) < 0.25, 1, -1)
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
                    signs = analytic_sample(theta, self.rng)
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



"""Run: python run_benchmark.py --mode full --runs 20
Requires benchmark_fun.py from the original project in this directory or PYTHONPATH.
"""


def run(mode='full', runs=20, pop_size=15, max_iter=40, max_change=6,
        output_root='results_QI_SOS', only=None):
    output = Path(output_root) / mode
    output.mkdir(parents=True, exist_ok=True)
    if runs < 1 or pop_size < 2 or max_iter < 1:
        raise ValueError('runs>=1, pop_size>=2 and max_iter>=1 are required')
    problems = benchmark_problems()
    if only:
        problems = {k: v for k, v in problems.items() if k in only}
        missing = set(only) - set(problems)
        if missing:
            raise ValueError(f'Unknown problems: {sorted(missing)}')
    expected_nfe = pop_size + 4 * pop_size * max_iter
    for name, (func, dim, bounds) in tqdm(problems.items(), desc=f'Problems ({mode})'):
        results, histories, nfe_histories = [], [], []
        for r in range(runs):
            seed = 42 * (r + 1)
            algo = QISOS(func, bounds, dim, pop_size, max_iter, seed,
                         mode=mode, max_change=max_change)
            best_x, best_f, history, positions = algo.optimize()
            assert algo.nfe == expected_nfe, (name, r, algo.nfe, expected_nfe)
            results.append(dict(problem=name, run=r, seed=seed, best=best_f,
                                nfe=algo.nfe, sampler_calls=algo.sampler_calls,
                                runtime_seconds=algo.runtime_seconds))
            histories.append(history)
            nfe_histories.append(algo.history_nfe)
            np.save(output / f'{name}_run{r}_history.npy', np.asarray(history))
            np.save(output / f'{name}_run{r}_position_history.npy', np.asarray(positions))
            np.save(output / f'{name}_run{r}_nfe_history.npy', np.asarray(algo.history_nfe))
        df = pd.DataFrame(results)
        df.to_csv(output / f'{name}_{mode}_results.csv', index=False)
        df['best'].agg(['mean', 'std', 'min']).to_csv(output / f'{name}_{mode}_summary.csv')
        H = np.asarray(histories)
        N = np.asarray(nfe_histories)
        assert np.all(N == N[0])
        np.save(output / f'{name}_{mode}_mean_curve.npy', H.mean(axis=0))
        np.save(output / f'{name}_{mode}_std_curve.npy', H.std(axis=0))
        np.save(output / f'{name}_{mode}_nfe_axis.npy', N[0])
        print(f'{name} {mode}: mean={df.best.mean():.6g}, std={df.best.std():.6g}, NFE={expected_nfe}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=QISOS.MODES + ('all',), default='full')
    p.add_argument('--validate', action='store_true', help='Run sampler and NFE validation before benchmarks')
    p.add_argument('--runs', type=int, default=20)
    p.add_argument('--pop-size', type=int, default=15)
    p.add_argument('--max-iter', type=int, default=40)
    p.add_argument('--max-change', type=int, default=6)
    p.add_argument('--output-root', default='results_QI_SOS')
    p.add_argument('--only', nargs='*', default=None)
    args = p.parse_args()
    if args.validate:
        validate()
    for mode in (QISOS.MODES if args.mode == 'all' else (args.mode,)):
        run(mode, args.runs, args.pop_size, args.max_iter,
            args.max_change, args.output_root, args.only)
