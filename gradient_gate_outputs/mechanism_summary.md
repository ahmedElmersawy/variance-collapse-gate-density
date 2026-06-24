# Experiment 3: Mechanism Logging — Summary

**This is exploratory. No causal claims are made — only observed
relationships with statistical confidence, exactly as instructed.**

**Setup:** Bundled into Experiment 2's training runs (CIFAR-native
ResNet-18, CIFAR-10, standard BatchNorm, 4 activations, 3 seeds). At
epochs {0, 6, 12, 18, 24}, on the fixed instrumentation batch: BatchNorm
γ/β statistics (pooled and per-layer), a Hutchinson trace estimate and
power-iteration top-eigenvalue estimate of the loss Hessian (eval mode,
to avoid BN running-stat contamination from the extra forward passes),
pre-/post-activation variance, gate_mean, active_frac.

## Finding 1 — three trends are universal across all four activations, all statistically robust

Sign test across the 12 independent (activation × seed) trajectories:

| quantity | direction | sign-test result |
|---|---|---|
| BN mean \|γ\| | declining | 12/12, p=2.44×10⁻⁴ |
| pre-activation variance | declining | 12/12, p=2.44×10⁻⁴ |
| Hessian trace estimate | rising | 12/12, p=2.44×10⁻⁴ |

Relative BN-γ shrinkage is similar in magnitude across activations
(ReLU −81.0%, GELU −73.3%, SiLU −72.4%, Mish −73.4%).

**Because these three trends are essentially universal — present and
similar in magnitude across ReLU and all three smooth activations — none
of them, individually, explains why `active_frac` diverges by activation
class.** The narrowing/sharpening happens to everyone; what differs is how
each activation's gate responds to it (see Experiment 1/threshold-robustness
findings: quantile compression for smooth activations vs. bimodal
mass-shift for ReLU).

## Finding 2 — correlation with gate_mean (pooled 3 seeds × 5 epochs = 15 points per activation)

| activation | γ ~ gate_mean | pre-act. var ~ gate_mean |
|---|---|---|
| relu | ρ=+0.918, p<0.001 | ρ=+0.829, p<0.001 |
| gelu | ρ=+0.918, p<0.001 | ρ=+0.714, p=0.003 |
| silu | ρ=+0.825, p<0.001 | ρ=+0.743, p=0.002 |
| mish | ρ=+0.729, p=0.002 | ρ=+0.714, p=0.003 |

Shrinking BatchNorm scale correlates strongly and significantly with the
(universal) decline in raw gate magnitude (`gate_mean`), for every
activation. This addresses the question "does shrinking BatchNorm scale
correlate with gate compression" — yes, observationally, for `gate_mean`.

## Finding 3 — sharpness correlates with `active_frac`, and the *sign* of that correlation mirrors the activation-class split

| activation | Hessian trace ~ active_frac |
|---|---|
| relu | ρ=−0.861, p<0.001 |
| gelu | ρ=+0.721, p=0.002 |
| silu | ρ=+0.732, p=0.002 |
| mish | ρ=+0.775, p=0.001 |

Sharpness rises for everyone, but its correlation with `active_frac` is
**negative for ReLU and positive for all three smooth activations** —
mirroring the direction split itself. This is a coherent observation, not
a new independent finding: since active_frac falls for ReLU and rises for
smooth activations while sharpness rises for both, the sign of this
correlation is close to a restatement of the main result, not new evidence
for it. We report it because it is a real, internally consistent
relationship in the data, not as an additional independent confirmation.

## What this does and does not establish

- **Does NOT establish causality** between BN-γ shrinkage, sharpness
  growth, or pre-activation-variance narrowing and the activation-class
  gate-density split. All three observed processes are common to every
  activation tested; they cannot, by themselves, explain a phenomenon that
  differs by activation.
- **Is consistent with** the quantile-compression account from the
  threshold-robustness study: a similarly-narrowing pre-activation
  distribution, passed through different activation derivative shapes
  (ReLU's step function vs. the smooth activations' hump-shaped
  derivative), plausibly produces the divergent `active_frac` outcome —
  but this script does not test that mechanistic link directly (it would
  require relating the per-channel pre-activation distribution shape to
  the per-channel gate value, not done here).
- The Hutchinson trace estimator is noisy at n=5 samples (one GELU
  seed/epoch combination produced a negative trace estimate, epoch 6 seed
  1, consistent with estimator variance rather than a real negative-trace
  region) — treat individual point estimates as approximate, the
  aggregate 12/12 sign-test direction as the reliable takeaway.
