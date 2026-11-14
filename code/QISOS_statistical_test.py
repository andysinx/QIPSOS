#!/usr/bin/env python3
"""Unified 9-algorithm statistics. Requires numpy pandas scipy statsmodels.
Place next to results_QI_SOS/ and results_classical_equalNFE_40iter/.
Run: python QISOS_statistical_9_algorithms.py
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, mannwhitneyu, binomtest, f as f_distribution
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parent
QI_FILE = BASE / 'results_QI_SOS' / 'QISOS_comparison_allruns.csv'
CLASSICAL_DIR = BASE / 'results_classical_equalNFE_40iter'
OUT = BASE / 'QIP_SOS_statistical_9_algorithms'
ALGS = ['QI-full', 'QI-p_only', 'QE-SOS', 'SOS', 'QI-c_only', 'GA', 'DE', 'PSO', 'MA']
CLASSICAL = {'GA','DE','PSO','MA'}
N_BENCHMARKS, N_RUNS, EXPECTED_NFE, ALPHA = 26, 20, 2415, 0.05


def load():
    if not QI_FILE.is_file():
        raise FileNotFoundError(f'Missing QI-SOS results: {QI_FILE}')
    qi = pd.read_csv(QI_FILE)
    required = {'Benchmark','Algorithm','Run','Seed','Best','NFE'}
    if required - set(qi.columns):
        raise ValueError(f'QI-SOS columns missing: {required-set(qi.columns)}')
    qi = qi.copy()
    if set(qi.Algorithm.unique()) != set(ALGS)-CLASSICAL:
        raise ValueError(f'Unexpected QI-SOS algorithms: {sorted(qi.Algorithm.unique())}')
    if not CLASSICAL_DIR.is_dir():
        raise FileNotFoundError(f'Missing classical results directory: {CLASSICAL_DIR}\nCopy *_classical_equalNFE_results.csv files here. Code alone is not results.')
    files = sorted(CLASSICAL_DIR.glob('*_classical_equalNFE_results.csv'))
    if not files:
        raise FileNotFoundError(f'No per-benchmark classical results CSVs in {CLASSICAL_DIR}')
    classical_frames = []
    for path in files:
        c = pd.read_csv(path)
        needed = {'problem','algorithm','run','seed','best','nfe'}
        if needed-set(c.columns):
            raise ValueError(f'{path.name}: missing {needed-set(c.columns)}')
        c = c.rename(columns={'problem':'Benchmark','algorithm':'Algorithm','run':'Run','seed':'Seed','best':'Best','nfe':'NFE','runtime_seconds':'Runtime_s'})
        classical_frames.append(c)
    classic = pd.concat(classical_frames, ignore_index=True)
    if not set(classic.Algorithm.unique()) <= CLASSICAL:
        raise ValueError(f'Unexpected classical algorithms: {sorted(set(classic.Algorithm)-CLASSICAL)}')
    if 'D' not in classic.columns:
        dimensions = (qi[['Benchmark','D']].dropna(subset=['D']).drop_duplicates()
                      if 'D' in qi else None)
        if dimensions is not None and dimensions['Benchmark'].duplicated().any():
            raise ValueError('Dimensioni D discordanti per lo stesso benchmark nel CSV QI-SOS')
        if dimensions is not None:
            classic = classic.merge(dimensions, on='Benchmark', how='left', validate='many_to_one')
    if 'D' in qi.columns:
        dim_map = qi[['Benchmark','D']].dropna(subset=['D']).drop_duplicates().set_index('Benchmark')['D']
        qi['D'] = qi['D'].fillna(qi['Benchmark'].map(dim_map))
    combined = pd.concat([qi, classic], ignore_index=True)
    validate(combined)
    return combined


def validate(df):
    expected = N_BENCHMARKS * N_RUNS * len(ALGS)
    if set(df.Algorithm.unique()) != set(ALGS):
        raise ValueError(f'Algorithm mismatch: {sorted(df.Algorithm.unique())}')
    if df.Benchmark.nunique() != N_BENCHMARKS or len(df) != expected:
        raise ValueError(f'Expected {N_BENCHMARKS} benchmarks and {expected} rows, got {df.Benchmark.nunique()} benchmarks and {len(df)} rows')
    if df.duplicated(['Benchmark','Algorithm','Run']).any():
        raise ValueError('Duplicate benchmark-algorithm-run rows')
    counts = df.groupby(['Benchmark','Algorithm']).size()
    if counts.size != N_BENCHMARKS * len(ALGS) or not counts.eq(N_RUNS).all():
        raise ValueError('Missing/incomplete benchmark-algorithm groups')
    for field in ['Best','NFE','Run','Seed']:
        if not np.isfinite(pd.to_numeric(df[field], errors='coerce')).all():
            raise ValueError(f'Invalid {field} values')
    if not pd.to_numeric(df.NFE).eq(EXPECTED_NFE).all():
        raise ValueError('Not all runs have NFE=2415')
    # Ensure identical run labels and seed assignment across all methods for each benchmark.
    for benchmark, group in df.groupby('Benchmark'):
        reference = None
        for alg in ALGS:
            g = group[group.Algorithm == alg].sort_values('Run')
            signature = tuple(zip(g.Run.astype(int), g.Seed.astype(int)))
            if reference is None: reference = signature
            elif signature != reference:
                raise ValueError(f'Run/seed mismatch in {benchmark} for {alg}')
    if 'D' in df and df.D.notna().any():
        if df.groupby('Benchmark').D.nunique(dropna=True).gt(1).any():
            raise ValueError('Dimension mismatch for at least one benchmark')


def holm(frame):
    frame = frame.copy()
    frame['p_holm'] = multipletests(frame.p_raw.to_numpy(), method='holm')[1]
    frame['significant_holm_0_05'] = frame.p_holm < ALPHA
    return frame.sort_values('p_holm')


def signed_rank(x,y):
    d = np.asarray(x,dtype=float)-np.asarray(y,dtype=float)
    if np.all(d==0): return 0.0,1.0
    test = wilcoxon(d, alternative='two-sided',zero_method='wilcox',method='auto')
    return float(test.statistic),float(test.pvalue)


def cliffs_delta(x,y):
    x,y=np.asarray(x),np.asarray(y)
    return float(((x[:,None]>y[None,:]).sum()-(x[:,None]<y[None,:]).sum())/(len(x)*len(y)))


def main():
    df=load()
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT/'combined_allruns_9_algorithms.csv',index=False)
    aggregations = dict(n_runs=('Best','size'),mean=('Best','mean'),std=('Best','std'),median=('Best','median'),best=('Best','min'),worst=('Best','max'))
    if 'D' in df: aggregations={'D':('D','first'),**aggregations}
    desc=df.groupby(['Benchmark','Algorithm'],as_index=False).agg(**aggregations)
    desc.to_csv(OUT/'descriptive_statistics.csv',index=False)
    means=desc.pivot(index='Benchmark',columns='Algorithm',values='mean')[ALGS]
    medians=desc.pivot(index='Benchmark',columns='Algorithm',values='median')[ALGS]
    ranks=means.rank(axis=1,ascending=True,method='average')
    for name,matrix in [('benchmark_means',means),('benchmark_medians',medians),('benchmark_ranks',ranks)]:
        matrix.to_csv(OUT/f'{name}.csv')
    summary=pd.DataFrame({'algorithm':ALGS,'mean_rank':[ranks[a].mean() for a in ALGS],
                          'wins_including_ties':[(means[a]==means.min(axis=1)).sum() for a in ALGS]}).sort_values('mean_rank')
    summary.to_csv(OUT/'algorithm_ranks.csv',index=False)
    fr=friedmanchisquare(*(means[a].to_numpy() for a in ALGS))
    n,k=len(means),len(ALGS)
    w=float(fr.statistic/(n*(k-1)))
    denominator=n*(k-1)-float(fr.statistic)
    fstat=((n-1)*float(fr.statistic)/denominator) if denominator>0 else float('inf')
    fp=float(f_distribution.sf(fstat,k-1,(k-1)*(n-1)))
    global_tests=pd.DataFrame([{'test':'Friedman','statistic':float(fr.statistic),'p_value':float(fr.pvalue),'effect_size_Kendall_W':w},
                               {'test':'Iman-Davenport','statistic':fstat,'p_value':fp,'effect_size_Kendall_W':w}])
    global_tests.to_csv(OUT/'global_tests.csv',index=False)
    pairs=list(itertools.combinations(ALGS,2))
    pair_tables={}
    for matrix,label in [(means,'means'),(ranks,'ranks')]:
        records=[]
        for a,b in pairs:
            x,y=matrix[a].to_numpy(),matrix[b].to_numpy()
            d=x-y
            stat,p=signed_rank(x,y)
            records.append(dict(algorithm_a=a,algorithm_b=b,statistic=stat,p_raw=p,a_better=int((d<0).sum()),b_better=int((d>0).sum()),ties=int((d==0).sum()),median_difference=float(np.median(d))))
        pair_tables[label]=holm(pd.DataFrame(records))
        pair_tables[label].to_csv(OUT/f'pairwise_wilcoxon_{label}_holm.csv',index=False)
    signs=[]
    for a,b in pairs:
        d=means[a].to_numpy()-means[b].to_numpy()
        wins,losses,ties=int((d<0).sum()),int((d>0).sum()),int((d==0).sum())
        p=float(binomtest(wins,wins+losses,0.5).pvalue) if wins+losses else 1.0
        signs.append(dict(algorithm_a=a,algorithm_b=b,a_better=wins,b_better=losses,ties=ties,p_raw=p))
    sign_table=holm(pd.DataFrame(signs))
    sign_table.to_csv(OUT/'pairwise_sign_test_holm.csv',index=False)
    records=[]
    for benchmark,group in df.groupby('Benchmark'):
        for a,b in pairs:
            x=group.loc[group.Algorithm==a,'Best'].to_numpy(dtype=float)
            y=group.loc[group.Algorithm==b,'Best'].to_numpy(dtype=float)
            if np.all(x==x[0]) and np.all(y==y[0]) and x[0]==y[0]:
                u,p=len(x)*len(y)/2,1.0
            else:
                test=mannwhitneyu(x,y,alternative='two-sided',method='auto')
                u,p=float(test.statistic),float(test.pvalue)
            records.append(dict(Benchmark=benchmark,algorithm_a=a,algorithm_b=b,n_a=len(x),n_b=len(y),mean_a=float(x.mean()),mean_b=float(y.mean()),median_a=float(np.median(x)),median_b=float(np.median(y)),mannwhitney_U=u,p_raw=p,cliffs_delta=cliffs_delta(x,y)))
    per=pd.DataFrame(records)
    total=len(per)
    per[f'p_holm_all_{total}']=multipletests(per.p_raw,method='holm')[1]
    per[f'significant_holm_all_{total}']=per[f'p_holm_all_{total}']<ALPHA
    per['p_holm_within_pair_26']=np.nan
    for _,idx in per.groupby(['algorithm_a','algorithm_b']).groups.items():
        per.loc[idx,'p_holm_within_pair_26']=multipletests(per.loc[idx,'p_raw'],method='holm')[1]
    per['significant_holm_within_pair_26']=per.p_holm_within_pair_26<ALPHA
    per.sort_values(['algorithm_a','algorithm_b','Benchmark']).to_csv(OUT/'per_benchmark_mannwhitney_holm.csv',index=False)
    report=[f'QIP-SOS | {n} benchmarks x {N_RUNS} runs x {k} algorithms | NFE={EXPECTED_NFE}',
            'Minimization: lower Best is better.',f'Pairwise comparisons: {len(pairs)}; Mann-Whitney tests: {total}',
            '', 'GLOBAL TESTS',global_tests.to_string(index=False),'','MEAN RANKS',summary.to_string(index=False)]
    for label,table in [('WILCOXON ON BENCHMARK MEANS',pair_tables['means']),('WILCOXON ON BENCHMARK RANKS',pair_tables['ranks']),('SIGN TEST',sign_table)]:
        report.extend(['',f'{label} (Holm across {len(pairs)} pairs)',table.to_string(index=False)])
    report.extend(['','MANN-WHITNEY U PER BENCHMARK',f'Significant Holm across all {total}: {int(per[f"significant_holm_all_{total}"].sum())}',f'Significant Holm within each pair (26): {int(per.significant_holm_within_pair_26.sum())}',
                   '','NOTES:','- Paired Wilcoxon unit is benchmark, not run/seed.','- Raw objective differences vary in scale between benchmarks.','- Mann-Whitney uses independent runs within each benchmark.','- Holm families are explicitly separated.','- Non-significance does not prove equivalence.','- The quantum entanglement/no-entanglement comparison is NOT included unless separately supplied.'])
    (OUT/'statistical_report.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report))
    print(f'\nSaved in: {OUT}')

if __name__=='__main__': main()