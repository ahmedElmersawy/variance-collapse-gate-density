import pandas as pd
import numpy as np
from scipy.stats import spearmanr, binomtest

df = pd.read_csv('gradient_gate_outputs/csv/mechanism_logging.csv')

print('=== Within-activation correlation (pooled 3 seeds x 5 epochs = 15 points) ===')
print('bn_mean_abs_gamma vs gate_mean, pre_activation_var vs gate_mean, hessian_trace vs active_frac')
for act in ['relu', 'gelu', 'silu', 'mish']:
    sub = df[df.activation == act]
    r1, p1 = spearmanr(sub.bn_mean_abs_gamma, sub.gate_mean)
    r2, p2 = spearmanr(sub.pre_activation_var, sub.gate_mean)
    r3, p3 = spearmanr(sub.hessian_trace_estimate, sub.active_frac)
    r4, p4 = spearmanr(sub.epoch, sub.pre_activation_var)
    r5, p5 = spearmanr(sub.epoch, sub.hessian_trace_estimate)
    print(f'{act:6s}: gamma~gate_mean rho={r1:+.3f} p={p1:.3f} | '
          f'preactvar~gate_mean rho={r2:+.3f} p={p2:.3f} | '
          f'trace~active_frac rho={r3:+.3f} p={p3:.3f} | '
          f'epoch~preactvar rho={r4:+.3f} p={p4:.3f} | '
          f'epoch~trace rho={r5:+.3f} p={p5:.3f}')

print()
print('=== Sign consistency across the 12 independent seed-trajectories (3 seeds x 4 activations) ===')
rows = []
for (act, seed), g in df.groupby(['activation', 'seed']):
    g = g.sort_values('epoch')
    r_gamma, _ = spearmanr(g.epoch, g.bn_mean_abs_gamma)
    r_preact, _ = spearmanr(g.epoch, g.pre_activation_var)
    r_trace, _ = spearmanr(g.epoch, g.hessian_trace_estimate)
    rows.append(dict(activation=act, seed=seed, r_gamma=r_gamma, r_preact=r_preact, r_trace=r_trace))
seed_df = pd.DataFrame(rows)
seed_df.to_csv('gradient_gate_outputs/csv/mechanism_seedlevel.csv', index=False)

n = len(seed_df)
k_gamma_decl = int((seed_df.r_gamma < 0).sum())
k_preact_decl = int((seed_df.r_preact < 0).sum())
k_trace_rise = int((seed_df.r_trace > 0).sum())
print(f'BN gamma declining with epoch: {k_gamma_decl}/{n} (sign-test p={binomtest(k_gamma_decl,n,0.5,alternative="greater").pvalue:.2e})')
print(f'pre-activation variance declining with epoch: {k_preact_decl}/{n} (sign-test p={binomtest(k_preact_decl,n,0.5,alternative="greater").pvalue:.2e})')
print(f'Hessian trace estimate rising with epoch: {k_trace_rise}/{n} (sign-test p={binomtest(k_trace_rise,n,0.5,alternative="greater").pvalue:.2e})')

print()
print('=== Relative magnitude of BN-gamma shrinkage across activations (start->end, % change) ===')
for act in ['relu', 'gelu', 'silu', 'mish']:
    sub = df[df.activation == act]
    start = sub[sub.epoch == 0].bn_mean_abs_gamma.mean()
    end = sub[sub.epoch == 24].bn_mean_abs_gamma.mean()
    print(f'{act:6s}: {start:.4f} -> {end:.4f}  ({(end-start)/start*100:+.1f}%)')
