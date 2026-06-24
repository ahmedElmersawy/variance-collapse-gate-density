import pandas as pd
import numpy as np
from scipy.stats import spearmanr, binomtest

df = pd.read_csv('gradient_gate_outputs/csv/bn_vs_gn_gate_dynamics.csv')

rows = []
for (norm, act, seed), g in df.groupby(['norm', 'activation', 'seed']):
    g = g.sort_values('epoch')
    r, p = spearmanr(g.epoch, g.active_frac)
    rows.append(dict(norm=norm, activation=act, seed=seed,
                      rho=r, start=g.active_frac.iloc[0], end=g.active_frac.iloc[-1],
                      final_test_acc=g.test_acc.iloc[-1]))
seed_df = pd.DataFrame(rows)
seed_df.to_csv('gradient_gate_outputs/csv/bn_vs_gn_statistics.csv', index=False)

pd.set_option('display.width', 160)
print('=== Direction + sign test, per norm x activation (n=3 seeds each) ===')
for norm in ['batchnorm', 'groupnorm']:
    print(f'--- {norm} ---')
    for act in ['relu', 'gelu', 'silu', 'mish']:
        sub = seed_df[(seed_df.norm == norm) & (seed_df.activation == act)]
        n_decl = (sub.rho < 0).sum()
        print(f'  {act:6s}: rho={[round(x,3) for x in sub.rho]}  start->end mean: '
              f'{sub.start.mean():.4f} -> {sub.end.mean():.4f}  declining={n_decl}/3  '
              f'final_test_acc={sub.final_test_acc.mean():.3f}')

print()
print('=== Sign test per norm/activation (n=3 seeds, the correct unit) ===')
for norm in ['batchnorm', 'groupnorm']:
    for act in ['relu']:
        s = seed_df[(seed_df.norm == norm) & (seed_df.activation == act)]
        k = int((s.rho < 0).sum())
        bt = binomtest(k, 3, p=0.5, alternative='greater')
        print(f'{norm}/{act}: {k}/3 declining, sign-test p={bt.pvalue:.3f}')
    for act in ['gelu', 'silu', 'mish']:
        s = seed_df[(seed_df.norm == norm) & (seed_df.activation == act)]
        k = int((s.rho > 0).sum())
        bt = binomtest(k, 3, p=0.5, alternative='greater')
        print(f'{norm}/{act}: {k}/3 rising, sign-test p={bt.pvalue:.3f}')

print()
print('=== Pooled across activations within each norm: does direction split survive? ===')
for norm in ['batchnorm', 'groupnorm']:
    sub = seed_df[seed_df.norm == norm]
    relu_decl = int((sub[sub.activation == 'relu'].rho < 0).sum())
    smooth_rise = int((sub[sub.activation.isin(['gelu', 'silu', 'mish'])].rho > 0).sum())
    print(f'{norm}: ReLU declining {relu_decl}/3, smooth-activation rising {smooth_rise}/9')
