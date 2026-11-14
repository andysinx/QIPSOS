"""QE-SOS circuit-level ablation: identical QE-SOS operators in all variants.
Requires benchmark_fun.py with benchmark_problems().
Dependencies: numpy pandas tqdm qiskit qiskit-aer.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from benchmark_fun import benchmark_problems

VARIANTS = ('full', 'no_cnot', 'one_layer', 'no_rz')


def build_circuit(theta, variant='full'):
    """Same theta encoding, with CNOT, depth, and RZ ablations."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    n = len(theta)
    qc = QuantumCircuit(n, n)
    layers = 1 if variant == 'one_layer' else 2
    for _ in range(layers):
        for i in range(n):
            qc.ry(float(theta[i]), i)
            if variant != 'no_rz':
                qc.rz(float(theta[i]), i)
        if variant != 'no_cnot':
            for i in range(n - 1):
                qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


class BitSampler:
    def __init__(self, shots=512, device='GPU', seed=42, variant='full'):
        self.shots = shots
        self.variant = variant
        self.device = device
        self.seed = seed
        self.simulator = AerSimulator(method='statevector', device=device, precision='single')
        self.calls = 0

    def sample(self, theta, rng):
        qc = build_circuit(theta, self.variant)
        # Transpile to Aer-supported gates. Independent seeds per circuit call.
        compiled = transpile(qc, self.simulator, optimization_level=0)
        sim_seed = int(rng.integers(1, 2**31 - 1))
        counts = self.simulator.run(compiled, shots=self.shots, seed_simulator=sim_seed).result().get_counts()
        bitstrings = list(counts)
        probabilities = np.array(list(counts.values()), dtype=float)
        probabilities /= probabilities.sum()
        bits = bitstrings[int(rng.choice(len(bitstrings), p=probabilities))]
        self.calls += 1
        return np.array([1 if bit == '1' else -1 for bit in bits[::-1]], dtype=float)


def decode_quantum(z, bounds):
    return bounds[:, 0] + (z + 1) * 0.5 * (bounds[:, 1] - bounds[:, 0])


class AblationSOS:
    def __init__(self, func, bounds, dim, variant, pop_size=15, max_iter=40,
                 seed=42, shots=512, device='CPU'):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        self.func, self.bounds, self.dim = func, np.asarray(bounds, dtype=float), dim
        if self.bounds.shape != (dim, 2) or np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError('Invalid bounds')
        if pop_size < 2:
            raise ValueError('pop_size must be >= 2')
        self.variant, self.pop_size, self.max_iter = variant, pop_size, max_iter
        self.rng = np.random.default_rng(seed)
        self.quantum = BitSampler(shots=shots, device=device, seed=seed, variant=variant)
        self.nfe = 0
        self.pop = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(pop_size, dim))
        self.fitness = np.array([self.evaluate(x) for x in self.pop])
        idx = np.argmin(self.fitness)
        self.best, self.best_fitness = self.pop[idx].copy(), float(self.fitness[idx])
        self.history_best, self.history_nfe, self.history_position = [], [], []
        self.runtime_seconds = 0.0

    def evaluate(self, x):
        self.nfe += 1
        value = float(self.func(x))
        if not np.isfinite(value):
            raise ValueError('Non-finite objective')
        return value

    def other(self, i):
        j = int(self.rng.integers(self.pop_size - 1))
        return j + (j >= i)

    def clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def signs(self, theta):
        return self.quantum.sample(theta, self.rng)

    def optimize(self):
        t0 = time.perf_counter()
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                xi = self.pop[i].copy()
                # Mutualism: same equations as both supplied original implementations.
                j = self.other(i)
                xj = self.pop[j].copy()
                bf1, bf2 = self.rng.integers(1, 3, size=2)
                mv = (xi + xj) / 2
                xi_new = self.clip(xi + self.rng.random(self.dim) * (self.best - mv * bf1))
                xj_new = self.clip(xj + self.rng.random(self.dim) * (self.best - mv * bf2))
                fi, fj = self.evaluate(xi_new), self.evaluate(xj_new)
                if fi < self.fitness[i]:
                    self.pop[i], self.fitness[i] = xi_new, fi
                if fj < self.fitness[j]:
                    self.pop[j], self.fitness[j] = xj_new, fj

                # Quantum Commensalism: unchanged across circuit variants.
                j = self.other(i)
                xj = self.pop[j].copy()
                max_change = min(self.dim, 6)
                n_change = int(self.rng.integers(1, max_change + 1))
                idx = self.rng.choice(self.dim, n_change, replace=False)
                theta = self.rng.uniform(0, 2 * np.pi, size=n_change)
                signs = self.signs(theta)
                xi_new = xi.copy()
                xi_new[idx] += 0.5 * signs * (self.best[idx] - xj[idx])
                xi_new = self.clip(xi_new)
                fi = self.evaluate(xi_new)
                if fi < self.fitness[i]:
                    self.pop[i], self.fitness[i] = xi_new, fi

                # Quantum Parasitism: unchanged across circuit variants.
                j = self.other(i)
                xi = self.pop[i].copy()
                max_change = min(self.dim, 6)
                n_change = int(self.rng.integers(1, max_change + 1))
                idx = self.rng.choice(self.dim, n_change, replace=False)
                theta = np.pi * ((xi[idx] - self.bounds[idx, 0]) /
                                 (self.bounds[idx, 1] - self.bounds[idx, 0]))
                z = self.signs(theta)
                perturb = decode_quantum(z, self.bounds[idx])
                candidate = xi.copy()
                candidate[idx] += 0.1 * (perturb - 0.5 * (self.bounds[idx, 1] + self.bounds[idx, 0]))
                candidate = self.clip(candidate)
                fp = self.evaluate(candidate)
                if fp < self.fitness[j]:
                    self.pop[j], self.fitness[j] = candidate, fp
            idx = int(np.argmin(self.fitness))
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = float(self.fitness[idx])
                self.best = self.pop[idx].copy()
            self.history_best.append(self.best_fitness)
            self.history_nfe.append(self.nfe)
            self.history_position.append(self.best.copy())
        self.runtime_seconds = time.perf_counter() - t0
        expected = self.pop_size + 4 * self.pop_size * self.max_iter
        assert self.nfe == expected, (self.nfe, expected)
        return self.best.copy(), self.best_fitness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument('--benchmarks', nargs='+', default=['Rastrigin', 'Ackley', 'Rosenbrock', 'Griewank', 'Schwefel12', 'Michalewicz10'])
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--pop-size', type=int, default=15)
    ap.add_argument('--generations', type=int, default=40)
    ap.add_argument('--shots', type=int, default=512, choices=[512], help='Fixed at 512 shots for this study')
    ap.add_argument('--device', choices=['CPU', 'GPU'], default='CPU')
    ap.add_argument('--output', default='results_QESOS_circuit_ablation')
    args = ap.parse_args()
    problems = benchmark_problems()
    missing = [name for name in args.benchmarks if name not in problems]
    if missing:
        raise ValueError(f'Benchmark names not found: {missing}. Available: {list(problems)}')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    all_rows = []
    for name in tqdm(args.benchmarks, desc='Benchmarks'):
        func, dim, bounds = problems[name]
        for variant in tqdm(args.variants, desc=name, leave=False):
            target = output / name / variant
            target.mkdir(parents=True, exist_ok=True)
            rows = []
            for r in tqdm(range(args.runs), desc=variant, leave=False):
                seed = 42 * (r + 1)
                algo = AblationSOS(func, bounds, dim, variant, args.pop_size, args.generations,
                                   seed, args.shots, args.device)
                best_x, best_f = algo.optimize()
                assert algo.quantum.calls == 2 * args.pop_size * args.generations
                row = dict(problem=name, variant=variant, run=r, seed=seed, best=best_f,
                           nfe=algo.nfe, quantum_calls=algo.quantum.calls,
                           runtime_seconds=algo.runtime_seconds)
                rows.append(row)
                all_rows.append(row)
                np.save(target / f'run{r}_history.npy', np.asarray(algo.history_best))
                np.save(target / f'run{r}_nfe_history.npy', np.asarray(algo.history_nfe))
                np.save(target / f'run{r}_position_history.npy', np.asarray(algo.history_position))
            pd.DataFrame(rows).to_csv(target / 'results.csv', index=False)
            print(f'{name} {variant}: mean={np.mean([x["best"] for x in rows]):.8g}')
    df = pd.DataFrame(all_rows)
    df.to_csv(output / 'all_runs.csv', index=False)
    df.groupby(['problem', 'variant'])['best'].agg(['mean', 'std', 'median', 'min', 'max']).to_csv(output / 'summary.csv')

if __name__ == '__main__':
    main()
