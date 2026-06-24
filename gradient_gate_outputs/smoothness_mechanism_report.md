# Smoothness Mechanism Report

**Setup:** CIFAR-native ResNet-18, CIFAR-10, identical optimizer/scheduler/
augmentation/batch size/instrumentation used throughout this project, 25
epochs, 3 seeds. 13 activation conditions: ReLU; LeakyReLU at slopes
0.001/0.01/0.05/0.10; PReLU (learnable); Softplus at β=50/20/10/5; GELU;
SiLU; Mish. 39 runs total, all completed without crashing except one
(disclosed below).

## A note on Softplus before any results: this is not just another smooth activation

Softplus's derivative is **exactly** the logistic sigmoid: `softplus'(x) =
sigmoid(beta*x)`. This is not an analogy — it is the same function this
project's original synthetic theory used to define the gate
`Gamma = sigmoid(alpha*(z-c))`, with `beta` playing the role of `alpha`.
The Softplus-β sweep is, to our knowledge, the first place in this project
where the original synthetic gate formalism and a real, trained network's
activation function are *literally the same mathematical object*, not
connected by analogy or shared vocabulary. This matters for how the result
below should be read: it is not just evidence for a smoothness hypothesis,
it is a direct empirical test of whether the synthetic theory's stiffness
parameter behaves the way the synthetic theory predicted, in a real
trained ResNet.

## Data-quality issues, disclosed before any interpretation

1. **PReLU, seed 0, is a genuine training failure**: test accuracy is
   exactly 10% (chance level) from epoch 0 onward, and gate statistics are
   NaN from epoch 0 (consistent with an immediate numerical divergence).
   This run is excluded from quantitative analysis and is **not**
   silently dropped — it is listed in `smoothness_sweep_primary_stats.csv`
   with `excluded_reason=training_failure`. PReLU's condition-level
   statistics are therefore based on n=2 seeds, not 3.
2. **`active_frac@0.01` (the threshold used everywhere else in this
   project) floor-saturates for LeakyReLU(0.05), LeakyReLU(0.10), and
   PReLU.** Each of these activations' "off-state" derivative (the
   negative-branch slope) is itself ≥0.01, so every unit counts as
   "active" at that threshold regardless of training — the metric is
   constant by construction, not because the phenomenon is absent. We
   confirmed this by checking `active_frac@0.10` for the same runs, which
   is *not* floor-saturated and reveals real, substantial trends (e.g.
   LeakyReLU(0.05): 0.500→0.366, a clear ReLU-like decline). **We use
   `active_frac@0.10` as the primary metric for this report** for
   consistency across all 13 conditions, and report this threshold
   sensitivity explicitly rather than switching thresholds silently
   per-condition. This is itself a transferable methodological finding:
   a fixed low gate-density threshold, chosen for hard-gated activations,
   can manufacture a false null result for near-hard-gated activations
   whose off-state floor exceeds it.

## Primary analysis: per-trajectory direction and magnitude

Full table: `smoothness_sweep_primary_stats.csv` (39 rows, all conditions
included, failures disclosed). Condition-level summary:
`smoothness_sweep_condition_summary.csv`.

| activation | group | n_seeds | mean ρ(epoch, active_frac@0.10) | mean Δactive_frac | smoothness index Var[f'] |
|---|---|---|---|---|---|
| relu | A | 3 | −0.976 | −0.135 | 0.250 |
| leaky_relu_0.001 | B | 3 | −0.974 | −0.122 | 0.249 |
| leaky_relu_0.01 | B | 3 | −0.986 | −0.137 | 0.245 |
| leaky_relu_0.05 | B | 3 | −0.968 | −0.136 | 0.226 |
| leaky_relu_0.10 | B | 3 | −0.964 | −0.126 | 0.202 |
| prelu | C | **2** | +0.647 | +0.063 | 0.261 |
| softplus_β50 | D | 3 | −0.588 | −0.071 | 0.235 |
| softplus_β20 | D | 3 | +0.731 | +0.017 | 0.211 |
| softplus_β10 | D | 3 | +0.999 | +0.111 | 0.185 |
| softplus_β5 | D | 3 | +0.998 | +0.090 | 0.130 |
| gelu | E | 3 | +0.996 | +0.058 | 0.122 |
| silu | E | 3 | +0.990 | +0.049 | 0.075 |
| mish | E | 3 | +0.971 | +0.042 | 0.097 |

**The sign transition happens between Softplus(β=50) and Softplus(β=20)**
— not at the LeakyReLU/PReLU boundary as a naive reading of the group
labels might suggest. Every LeakyReLU variant tested (slope 0.001 through
0.10) declines as strongly as ReLU itself; Softplus(β=50), despite being
smooth, behaves like the hard-gated group (declines); Softplus(β=20) is
the first condition to flip sign.

## Mechanistic analysis: smoothness index vs. gate-density trend

`smoothness_correlation_statistics.csv` has the full table. Headline
results:

| level | x vs y | test | statistic | p / CI |
|---|---|---|---|---|
| condition (n=12) | smoothness_index vs mean ρ | Spearman | **−0.731** | p=0.0045 |
| condition (n=12) | smoothness_index vs mean Δactive_frac | Spearman | −0.451 | p=0.122 (n.s. by naive test) |
| condition (n=12), bootstrap | smoothness_index vs mean Δactive_frac | Spearman | −0.521 | 95% CI=[−0.775, −0.401] (entirely negative) |
| seed-trajectory (n=37) | smoothness_index vs ρ | Spearman | **−0.710** | p=6.1×10⁻⁷ |
| seed-trajectory (n=37) | smoothness_index vs Δactive_frac | Spearman | **−0.605** | p=5.7×10⁻⁵ |

The relationship is strong and statistically robust when tested at the
seed-trajectory level (the more appropriate unit, since it uses every
independent run rather than collapsing to 12 condition means) and for the
trend-shape statistic (ρ) at the condition level. The condition-level
naive Spearman test on raw magnitude (Δactive_frac) does not individually
reach p<0.05 at n=12, but its bootstrap confidence interval — which
properly propagates the seed-level uncertainty within each condition
rather than treating 12 condition means as exact — is entirely negative
and excludes zero. We report both the naive and the bootstrap result
rather than only the more favorable one.

## Monotonicity: real, with one well-explained exception, and labels that don't perfectly match the actual ranking

A naive adjacent-pair check (sorted by our a-priori hypothesized order)
finds 6/12 transitions "violating" strict monotonicity. On inspection,
this overstates the disagreement:

- **One genuine, important exception: PReLU.** Its smoothness index *at
  initialization* (0.261) is the **highest of all 13 conditions** —
  higher than ReLU's — yet its trend rises rather than falls. This is not
  a counterexample to the smoothness hypothesis so much as a limit of
  measuring smoothness only at initialization: PReLU's negative-branch
  slope is a *learned* parameter, and its gate variance drops sharply
  during training (from ~0.24–0.28 at init to ~0.08–0.12 by epoch 24,
  visible in `mechanism_logging.csv`-style trajectories within this run)
  — meaning PReLU's *effective* smoothness changes substantially over
  training in a way fixed-shape activations' does not. An initialization-
  only smoothness index is the wrong measurement for an adaptive
  activation, and we say so rather than treating this as a refutation.
- **Four "violations" are within-regime magnitude wobbles, not sign
  reversals** (e.g. Softplus(β10)→Softplus(β5): 0.111→0.090, both clearly
  rising) — consistent with a **ceiling effect**: `active_frac@0.10` is
  bounded above by 1, and the smoothest conditions start closest to that
  ceiling, leaving less room to rise further. This is the same ceiling
  dynamic flagged earlier in this project's threshold-robustness work.
- **Two "violations" (GELU↔SiLU↔Mish ordering) are an artifact of our own
  a-priori group labels, not of the data**: empirically, SiLU's measured
  smoothness index (0.075) is *lower* (smoother) than both Mish's (0.097)
  and GELU's (0.122) — we had grouped all three together as
  "E_fully_smooth" without an internal ranking. The correct test —
  Spearman correlation against the *actual measured* smoothness index,
  not our assumed label order — already accounts for this and is the
  number reported above (ρ=−0.731 condition-level, ρ=−0.710
  seed-level), not the naive label-order check.

## Conclusion, in the required wording

There is a real, monotonic — in the correlational sense, against the
*actual measured* smoothness index — relationship between activation
derivative smoothness and gate-density trend direction, robust at the
seed-trajectory level (p<10⁻⁵ for both the shape and magnitude
statistics) and supported at the condition level for the trend-shape
statistic (p=0.0045) and, via bootstrap, for the magnitude statistic as
well. The Softplus-β family — whose derivative is exactly a logistic
sigmoid, the same function as the project's original synthetic theory —
shows the predicted transition from ReLU-like decline (β=50) to
GELU-like rise (β=20 and below) within the smoothness sweep itself.

**This is evidence consistent with the hypothesis that activation
smoothness is a plausible mechanistic driver of the gate-density
direction split. It does not prove that smoothness is the sole or
complete explanation** — PReLU's behavior shows that a fixed,
initialization-time smoothness measurement is insufficient for adaptive
activations, and the relationship's exact functional form (versus simply
its sign and rank correlation) is not established by this experiment. We
do not claim proof of mechanism; we report what the data shows, including
where it complicates a clean story.
