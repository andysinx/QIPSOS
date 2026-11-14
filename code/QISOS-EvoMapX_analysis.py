"""Standalone EvoMapX analysis for QI-SOS. Only local benchmark_fun.py is needed.
Requires numpy, pandas, matplotlib, networkx, tqdm.
"""
"""QI-SOS benchmark and optional sampler validation (No-CNOT analytical model)."""
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
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



OPS = ('Mutualism', 'Commensalism', 'Parasitism')

class Tracer:
    """Records all proposals and the provenance of accepted solutions."""
    def __init__(self, pop, fitness, bounds):
        self.bounds = np.asarray(bounds, dtype=float)
        self.records = []
        self.graph = nx.DiGraph()
        self.ids = []
        self.next_id = 0
        self.diversity = []
        for i, (x, f) in enumerate(zip(pop, fitness)):
            self.ids.append(self._node(-1, 'Initialization', i, f))

    def _node(self, generation, operator, slot, fitness):
        node = self.next_id
        self.next_id += 1
        self.graph.add_node(node, generation=int(generation), operator=operator,
                            slot=int(slot), fitness=float(fitness))
        return node

    def log(self, generation, operator, slot, old_fit, new_fit, parents, accepted):
        gain = max(0.0, float(old_fit) - float(new_fit)) if accepted else 0.0
        self.records.append(dict(generation=generation, operator=operator, slot=slot,
                                 old_fitness=float(old_fit), candidate_fitness=float(new_fit),
                                 accepted=int(accepted), improvement=gain, node_id=self.next_id if accepted else -1))
        if accepted:
            child = self._node(generation, operator, slot, new_fit)
            for parent in set(parents):
                self.graph.add_edge(int(parent), child)
            self.ids[slot] = child

    def record_diversity(self, pop):
        norm = (np.asarray(pop) - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        dist = np.linalg.norm(norm[:, None, :] - norm[None, :, :], axis=2)
        self.diversity.append(float(dist[np.triu_indices(len(norm), 1)].mean()))

    def frame(self):
        return pd.DataFrame(self.records)

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
        self.tracer = Tracer(self.pop, self.fitness, self.bounds)

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
        for gen in range(self.max_iter):
            for i in range(self.pop_size):
                xi = self.pop[i].copy()
                xi_id = self.tracer.ids[i]
                # MUTUALISM -- same as supplied SOS and QE-SOS
                j = self._other(i)
                xj = self.pop[j].copy()
                xj_id = self.tracer.ids[j]
                bf1, bf2 = self.rng.randint(1, 3, size=2)
                mv = (xi + xj) / 2
                xi_new = self._clip(xi + self.rng.rand(self.dim) * (self.best - mv * bf1))
                xj_new = self._clip(xj + self.rng.rand(self.dim) * (self.best - mv * bf2))
                fi, fj = self._evaluate(xi_new), self._evaluate(xj_new)
                accept_i = bool(fi < self.fitness[i])
                accept_j = bool(fj < self.fitness[j])
                self.tracer.log(gen, 'Mutualism', i, self.fitness[i], fi, (xi_id, xj_id), accept_i)
                self.tracer.log(gen, 'Mutualism', j, self.fitness[j], fj, (xi_id, xj_id), accept_j)
                if accept_i:
                    self.pop[i], self.fitness[i] = xi_new, fi
                if accept_j:
                    self.pop[j], self.fitness[j] = xj_new, fj

                # COMMENSALISM -- preserve original QE-SOS alpha=0.5 and dimension selection
                j = self._other(i)
                xj = self.pop[j].copy()
                xj_id = self.tracer.ids[j]
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
                accepted = bool(fi < self.fitness[i])
                self.tracer.log(gen, 'Commensalism', i, self.fitness[i], fi, (xi_id, xj_id), accepted)
                if accepted:
                    self.pop[i], self.fitness[i] = xi_new, fi

                # PARASITISM -- QE-SOS's state-dependent angle and boundary decoding
                j = self._other(i)
                if self.mode in ('full', 'p_only'):
                    xi = self.pop[i].copy()
                    xi_id = self.tracer.ids[i]
                # c_only retains the earlier xi snapshot and its original ancestor
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
                accepted = bool(fp < self.fitness[j])
                self.tracer.log(gen, 'Parasitism', j, self.fitness[j], fp, (xi_id,), accepted)
                if accepted:
                    self.pop[j], self.fitness[j] = parasite, fp
            self._update_best()
            self.history_best.append(self.best_fitness)
            self.history_mean.append(float(np.mean(self.fitness)))
            self.history_position.append(self.best.copy())
            self.history_nfe.append(self.nfe)
            self.tracer.record_diversity(self.pop)
        self.runtime_seconds = time.perf_counter() - start
        return self.best, self.best_fitness, self.history_best, self.history_position





def lineage_metrics(graph):
    """Descriptive PEG metrics; descendant counts are NOT causal effects."""
    rows = []
    for node, attrs in graph.nodes(data=True):
        if attrs['operator'] == 'Initialization':
            continue
        desc = nx.descendants(graph, node)
        rows.append(dict(node=node, generation=attrs['generation'],
                         operator=attrs['operator'], descendants=len(desc),
                         terminal_descendants=sum(graph.out_degree(d) == 0 for d in desc)))
    return pd.DataFrame(rows)


def lineage_cds(events, graph, decay=0.9):
    """Exploratory lineage-weighted score using each accepted node's OWN direct gain.

    CDS[g,op] = sum_{accepted events at t<=g} gain(event)*decay**(g-t)
    for which an accepted descendant from that event is created at g.
    Direct gain is not reallocated to other operators. This is descriptive,
    not a counterfactual attribution of delayed improvement.
    """
    rows = []
    for node, attr in graph.nodes(data=True):
        if attr['operator'] == 'Initialization':
            continue
        g0 = attr['generation']
        # Direct improvement of this node against replaced occupant, recorded by event node_id.
        matched = events.loc[events.node_id == node]
        if len(matched) != 1:
            raise AssertionError('PEG node does not map to one accepted event')
        gain = float(matched.iloc[0].improvement)
        rows.append(dict(generation=g0, operator=attr['operator'], score=gain, kind='direct'))
        for descendant in nx.descendants(graph, node):
            gd = graph.nodes[descendant]['generation']
            if gd > g0:
                rows.append(dict(generation=gd, operator=attr['operator'],
                                 score=gain * decay**(gd-g0), kind='lineage_proxy'))
    return pd.DataFrame(rows, columns=['generation','operator','score','kind'])


def summarize_events(events):
    stats = events.groupby(['generation', 'operator']).agg(
        total_gain=('improvement', 'sum'), attempts=('accepted', 'size'),
        successes=('accepted', 'sum')).reset_index()
    stats['cds'] = stats.total_gain / stats.attempts
    stats['success_rate'] = stats.successes / stats.attempts
    return stats


def make_plots(events, diversity, graph, out, label, final_ids):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stats = summarize_events(events)
    stats.to_csv(out / 'oam_cds_by_generation.csv', index=False)
    events.to_csv(out / 'operator_events.csv', index=False)
    nx.write_graphml(graph, out / 'peg.graphml')
    lineage_metrics(graph).to_csv(out / 'PEG_lineage_metrics.csv', index=False)
    weighted = lineage_cds(events, graph)
    weighted.to_csv(out / 'CDS_lineage_components.csv', index=False)
    if len(weighted):
        wide = weighted.groupby(['generation','operator','kind'], as_index=False).score.sum()
        fig, ax = plt.subplots(figsize=(10, 4))
        for op in OPS:
            d = wide[(wide.operator == op) & (wide.kind == 'lineage_proxy')]
            ax.plot(d.generation, d.score, label=op)
        ax.set(xlabel='Generation', ylabel='Decayed lineage proxy (not causal)',
               title=f'{label}: lineage-weighted CDS proxy')
        ax.legend(); fig.tight_layout(); fig.savefig(out / 'CDS_lineage_proxy.png', dpi=220); plt.close(fig)
    pivot = stats.pivot(index='operator', columns='generation', values='cds').reindex(OPS)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(pivot.to_numpy(), aspect='auto', interpolation='nearest')
    ax.set_yticks(range(3), OPS)
    ax.set(xlabel='Generation (0-indexed)', title=f'{label}: OAM (accepted improvement / attempt)')
    fig.colorbar(im, ax=ax, label='Fitness improvement')
    fig.tight_layout(); fig.savefig(out / 'OAM.png', dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for op in OPS:
        d = stats[stats.operator == op]
        ax.plot(d.generation, d.cds, label=op)
    ax.set(xlabel='Generation', ylabel='Accepted improvement / attempt', title=f'{label}: direct CDS')
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'CDS_time.png', dpi=220); plt.close(fig)
    vals = stats.groupby('operator').agg(total=('total_gain', 'sum'), attempts=('attempts', 'sum'))
    vals = (vals.total / vals.attempts).reindex(OPS)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(vals.index, vals.values)
    ax.set(ylabel='Total accepted improvement / total attempts', title=f'{label}: overall CDS')
    fig.tight_layout(); fig.savefig(out / 'CDS_bar.png', dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(diversity)), diversity)
    ax.set(xlabel='Generation', ylabel='Mean normalized pairwise distance', title=f'{label}: population diversity')
    fig.tight_layout(); fig.savefig(out / 'diversity.png', dpi=220); plt.close(fig)
    best = min(final_ids, key=lambda node: graph.nodes[node]['fitness'])
    sub = graph.subgraph(nx.ancestors(graph, best) | {best}).copy()
    positions = {n: (graph.nodes[n]['generation'], -graph.nodes[n]['slot']) for n in sub}
    colors = {'Initialization': '#888888', 'Mutualism': '#3478a6',
              'Commensalism': '#e89b30', 'Parasitism': '#6eae72'}
    fig, ax = plt.subplots(figsize=(12, 6))
    nx.draw_networkx_edges(sub, positions, ax=ax, arrows=False, alpha=.35, width=.8)
    nx.draw_networkx_nodes(sub, positions, ax=ax, node_size=24,
                           node_color=[colors[graph.nodes[n]['operator']] for n in sub])
    ax.set(title=f'{label}: PEG ancestry of best surviving organism',
           xlabel='Generation', ylabel='Negative population slot')
    fig.tight_layout(); fig.savefig(out / 'PEG.png', dpi=220); plt.close(fig)


def run_campaign(problems, output_dir, modes, runs=20, pop_size=15, generations=40):
    root = Path(output_dir)
    for name, (func, dim, bounds) in tqdm(problems.items(), desc='Benchmark problems'):
        for mode in modes:
            label = {'full': 'QI-full', 'p_only': 'QI-p_only', 'c_only': 'QI-c_only'}[mode]
            directory = root / name / label
            directory.mkdir(parents=True, exist_ok=True)
            rows, events, divs = [], [], []
            for r in range(runs):
                seed = 42 * (r + 1)
                algo = QISOS(func, bounds, dim, pop_size, generations, seed=seed, mode=mode)
                algo.optimize()
                assert algo.nfe == pop_size + 4 * pop_size * generations
                rows.append(dict(run=r, seed=seed, best=algo.best_fitness, nfe=algo.nfe,
                                 sampler_calls=algo.sampler_calls, runtime_seconds=algo.runtime_seconds))
                df = algo.tracer.frame()
                assert len(df) == 4 * pop_size * generations
                df['run'] = r
                events.append(df)
                divs.append(algo.tracer.diversity)
                if r == 0:
                    make_plots(df, algo.tracer.diversity, algo.tracer.graph,
                               directory / 'representative_run0', f'{name} / {label} / run 0', algo.tracer.ids)
            pd.DataFrame(rows).to_csv(directory / 'results.csv', index=False)
            all_events = pd.concat(events, ignore_index=True)
            all_events.to_csv(directory / 'all_operator_events.csv', index=False)
            all_events.groupby(['run', 'generation', 'operator']).agg(
                gain=('improvement', 'sum'), attempts=('accepted', 'size'),
                accepted=('accepted', 'sum')).reset_index().to_csv(
                    directory / 'per_run_operator_metrics.csv', index=False)
            agg = summarize_events(all_events)
            agg.to_csv(directory / 'aggregate_oam_cds.csv', index=False)
            fig, ax = plt.subplots(figsize=(10, 4))
            for op in OPS:
                d = agg[agg.operator == op]
                ax.plot(d.generation, d.cds, label=op)
            ax.set(xlabel='Generation', ylabel='Mean accepted improvement / attempt',
                   title=f'{name} / {label}: pooled direct CDS ({runs} runs)')
            ax.legend(); fig.tight_layout(); fig.savefig(directory / 'CDS_aggregate.png', dpi=220); plt.close(fig)
            a = np.asarray(divs)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(np.arange(generations), a.mean(axis=0))
            ax.fill_between(np.arange(generations), a.mean(axis=0)-a.std(axis=0),
                            a.mean(axis=0)+a.std(axis=0), alpha=.2)
            ax.set(xlabel='Generation', ylabel='Normalized pairwise distance',
                   title=f'{name} / {label}: diversity mean ± SD')
            fig.tight_layout(); fig.savefig(directory / 'diversity_aggregate.png', dpi=220); plt.close(fig)
            print(f'{name} / {label}: completed {runs} runs', flush=True)



def self_test():
    """Compare against original algorithm shipped inside this same file's reference
    companion when present; otherwise check internal invariants. No external imports."""
    for mode in QISOS.MODES:
        for dim in (2, 5, 10):
            f = lambda x: float(np.dot(x, x))
            a = QISOS(f, [(-5, 5)]*dim, dim, pop_size=5, max_iter=3,
                      seed=42, mode=mode)
            a.optimize()
            assert a.nfe == 65
            assert len(a.tracer.records) == 60
            assert sum(r['accepted'] for r in a.tracer.records) == a.tracer.graph.number_of_nodes()-5
            assert nx.is_directed_acyclic_graph(a.tracer.graph)
            assert all(len(set(a.tracer.ids)) == 5 for _ in [0])
            assert all(np.isfinite(r['improvement']) for r in a.tracer.records)
            assert len(a.tracer.diversity) == 3
            assert a.sampler_calls == (2 if mode == 'full' else 1)*5*3
    print('PASS: 9 deterministic internal invariants (NFE, PEG, OAM, CDS, sampler)')

def main():
    parser = argparse.ArgumentParser(description='Standalone EvoMapX-inspired QI-SOS instrumentation')
    parser.add_argument('--mode', choices=('full', 'p_only', 'c_only', 'all'), default='all')
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--population', type=int, default=15)
    parser.add_argument('--problem', default=None, help='Exact benchmark name')
    parser.add_argument('--output', default='results_evomapx_QI_SOS')
    parser.add_argument('--self-test', action='store_true', help='Check deterministic equivalence with original QI-SOS')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.runs < 1 or args.generations < 1 or args.population < 2:
        parser.error('runs>=1, generations>=1, population>=2 required')
    problems = benchmark_problems()
    if args.problem:
        if args.problem not in problems:
            parser.error(f'Unknown benchmark: {args.problem}; available: {list(problems)}')
        problems = {args.problem: problems[args.problem]}
    modes = QISOS.MODES if args.mode == 'all' else (args.mode,)
    run_campaign(problems, args.output, modes, args.runs, args.population, args.generations)


if __name__ == '__main__':
    main()