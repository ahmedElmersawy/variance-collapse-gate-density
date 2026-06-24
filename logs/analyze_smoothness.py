import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

df = pd.read_csv('gradient_gate_outputs/csv/smoothness_sweep_results.csv')

GROUP_ORDER = {
    'relu': ('A_hard_gated', 0),
    'leaky_relu_0.001': ('B_near_relu', 1),
    'leaky_relu_0.01': ('B_near_relu', 2),
    'leaky_relu_0.05': ('B_near_relu', 3),
    'leaky_relu_0.10': ('B_near_relu', 4),
    'prelu': ('C_learnable', 5),
    'softplus_beta50': ('D_softplus', 6),
    'softplus_beta20': ('D_softplus', 7),
    'softplus_beta10': ('D_softplus', 8),
    'softplus_beta5': ('D_softplus', 9),
    'gelu': ('E_fully_smooth', 10),
    'silu': ('E_fully_smooth', 11),
    'mish': ('E_fully_smooth', 12),
}

# prelu seed=0 is a genuine training failure (test_acc=0.1 throughout, gate
# stats all NaN from epoch 0) -- disclosed explicitly, excluded from
# quantitative correlation analysis (cannot compute a trend from NaN/constant
# chance-level data), not silently dropped from the report.
FAILED_RUNS = {('prelu', 0)}

METRIC = 'active_frac_t0.1'  # primary metric: 0.01 floor-saturates for
# leaky_relu_0.05/0.10 and prelu (their negative-branch slope already
# exceeds 0.01, so active_frac@0.01 is constant/uninformative for them by
# construction, not because the phenomenon is absent -- see report).

rows = []
for (act, seed), g in df.groupby(['activation', 'seed']):
    g = g.sort_values('epoch')
    failed = (act, seed) in FAILED_RUNS
    if failed or g[METRIC].isna().any():
        rows.append(dict(activation=act, seed=seed, group=GROUP_ORDER[act][0], order=GROUP_ORDER[act][1],
                          rho=np.nan, p_value=np.nan, active_frac_start=np.nan, active_frac_end=np.nan,
                          delta_active_frac=np.nan, relative_change=np.nan,
                          active_frac_001_start=g['active_frac_t0.01'].iloc[0],
                          active_frac_001_end=g['active_frac_t0.01'].iloc[-1],
                          smoothness_index_var=np.nan, final_test_acc=g.test_acc.iloc[-1],
                          excluded_reason='training_failure' if failed else 'nan_metric'))
        continue
    rho, p = spearmanr(g.epoch, g[METRIC])
    start = g[METRIC].iloc[0]
    end = g[METRIC].iloc[-1]
    delta = end - start
    rel_change = delta / start if start != 0 else np.nan
    smoothness_index = g['gate_var'].iloc[0]
    rows.append(dict(activation=act, seed=seed, group=GROUP_ORDER[act][0], order=GROUP_ORDER[act][1],
                      rho=rho, p_value=p, active_frac_start=start, active_frac_end=end,
                      delta_active_frac=delta, relative_change=rel_change,
                      active_frac_001_start=g['active_frac_t0.01'].iloc[0],
                      active_frac_001_end=g['active_frac_t0.01'].iloc[-1],
                      smoothness_index_var=smoothness_index, final_test_acc=g.test_acc.iloc[-1],
                      excluded_reason=None))
seed_df = pd.DataFrame(rows).sort_values(['order', 'seed'])
seed_df.to_csv('gradient_gate_outputs/csv/smoothness_sweep_primary_stats.csv', index=False)

pd.set_option('display.width', 220)
print('=== Primary analysis: per-trajectory direction/magnitude using active_frac@0.10 (13 activations x 3 seeds = 39) ===')
print(seed_df[['activation', 'group', 'seed', 'rho', 'delta_active_frac', 'relative_change',
               'smoothness_index_var', 'final_test_acc', 'excluded_reason']].to_string(index=False))

valid = seed_df[seed_df.excluded_reason.isna()]
print(f'\n{len(valid)}/39 trajectories usable for quantitative correlation; '
      f'{39-len(valid)} excluded: {seed_df[seed_df.excluded_reason.notna()][["activation","seed","excluded_reason"]].values.tolist()}')

cond = valid.groupby(['activation', 'group', 'order']).agg(
    mean_rho=('rho', 'mean'), std_rho=('rho', 'std'), n_seeds=('rho', 'count'),
    mean_delta=('delta_active_frac', 'mean'), std_delta=('delta_active_frac', 'std'),
    mean_smoothness=('smoothness_index_var', 'mean'), std_smoothness=('smoothness_index_var', 'std'),
    mean_final_acc=('final_test_acc', 'mean'),
).reset_index().sort_values('order')
cond.to_csv('gradient_gate_outputs/csv/smoothness_sweep_condition_summary.csv', index=False)
print()
print('=== Condition-level summary (sorted by hypothesized smoothness order, active_frac@0.10) ===')
print(cond.to_string(index=False))

print()
print('=== Mechanistic test: smoothness_index (Var[f\'(x)] at init) vs gate-density trend ===')
print('--- Condition-level (n=12 valid conditions; prelu has n_seeds=2 due to the excluded failed run) ---')
r_pear_rho, p_pear_rho = pearsonr(cond.mean_smoothness, cond.mean_rho)
r_spear_rho, p_spear_rho = spearmanr(cond.mean_smoothness, cond.mean_rho)
r_pear_delta, p_pear_delta = pearsonr(cond.mean_smoothness, cond.mean_delta)
r_spear_delta, p_spear_delta = spearmanr(cond.mean_smoothness, cond.mean_delta)
print(f'smoothness_index vs mean_rho:   Pearson r={r_pear_rho:+.3f} p={p_pear_rho:.4f}   '
      f'Spearman rho={r_spear_rho:+.3f} p={p_spear_rho:.4f}')
print(f'smoothness_index vs mean_delta: Pearson r={r_pear_delta:+.3f} p={p_pear_delta:.4f}   '
      f'Spearman rho={r_spear_delta:+.3f} p={p_spear_delta:.4f}')

print()
print('--- Seed-level (n=37 valid independent trajectories) ---')
r_pear_rho_s, p_pear_rho_s = pearsonr(valid.smoothness_index_var, valid.rho)
r_spear_rho_s, p_spear_rho_s = spearmanr(valid.smoothness_index_var, valid.rho)
r_pear_delta_s, p_pear_delta_s = pearsonr(valid.smoothness_index_var, valid.delta_active_frac)
r_spear_delta_s, p_spear_delta_s = spearmanr(valid.smoothness_index_var, valid.delta_active_frac)
print(f'smoothness_index vs rho:   Pearson r={r_pear_rho_s:+.3f} p={p_pear_rho_s:.2e}   '
      f'Spearman rho={r_spear_rho_s:+.3f} p={p_spear_rho_s:.2e}')
print(f'smoothness_index vs delta: Pearson r={r_pear_delta_s:+.3f} p={p_pear_delta_s:.2e}   '
      f'Spearman rho={r_spear_delta_s:+.3f} p={p_spear_delta_s:.2e}')

print()
print('--- Bootstrap 95% CI for condition-level Spearman(smoothness_index, mean_delta), resampling seeds within condition ---')
rng = np.random.default_rng(0)
boots = []
acts = cond.activation.tolist()
for _ in range(2000):
    boot_means_smooth, boot_means_delta = [], []
    for act in acts:
        sub = valid[valid.activation == act]
        idx = rng.integers(0, len(sub), size=len(sub))
        boot_means_smooth.append(sub.smoothness_index_var.values[idx].mean())
        boot_means_delta.append(sub.delta_active_frac.values[idx].mean())
    r, _ = spearmanr(boot_means_smooth, boot_means_delta)
    boots.append(r)
boots = np.array(boots)
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f'point estimate rho={r_spear_delta:+.3f}, bootstrap 95% CI=[{lo:+.3f}, {hi:+.3f}]')

print()
print('=== Monotonicity check (mean_delta, in hypothesized smoothness order) ===')
ordered = cond.sort_values('order')
deltas = ordered.mean_delta.values
n_violations = sum(1 for i in range(len(deltas) - 1) if deltas[i + 1] < deltas[i] - 1e-9)
print(f'{n_violations} non-monotonic adjacent-pair violations out of {len(deltas)-1} transitions')
for _, row in ordered.iterrows():
    print(f"  {row['order']:2d} {row['activation']:18s} n_seeds={row['n_seeds']:.0f}  "
          f"smoothness_var={row['mean_smoothness']:.4f}  mean_delta_af@0.10={row['mean_delta']:+.4f}  "
          f"mean_rho={row['mean_rho']:+.3f}")
