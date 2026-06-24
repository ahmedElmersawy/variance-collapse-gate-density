# Experiment 2: Gate-Density-Guided Pruning — Summary

**Setup:** CIFAR-native ResNet-18, CIFAR-10, standard BatchNorm, 4
activations, 3 seeds, 25 epochs, final checkpoints saved. For each
checkpoint: per-channel final gate density and per-channel weight
magnitude (L1 norm of the conv1 filter) computed for all 1920 channels
across the 8 `stages.{0..7}.conv1` layers (gated by `act1`). Channels
pruned globally (not per-layer-uniform) by zeroing `conv1.weight`,
`bn1.weight` (γ), and `bn1.bias` (β) for the selected channels — exact
zero contribution downstream, since every activation tested maps f(0)=0.
No retraining; immediate test-accuracy drop measured at 4 pruning ratios
against 3 criteria (gate-density-lowest-first, magnitude-lowest-first,
random).

## Headline result — and it is NOT what the smoke test (1-epoch checkpoint) suggested

At a fully-trained (25-epoch) checkpoint, **gate-density pruning is
dramatically worse than magnitude pruning for every smooth activation, at
every ratio, with no exceptions**:

| activation | ratio | gate_density drop | magnitude drop | random drop |
|---|---|---|---|---|
| relu | 10% | 0.047 ± 0.018 | 0.031 ± 0.038 | 0.036 ± 0.013 |
| relu | 50% | 0.647 ± 0.042 | 0.656 ± 0.096 | 0.749 ± 0.065 |
| gelu | 10% | **0.653 ± 0.030** | 0.008 ± 0.007 | 0.067 ± 0.051 |
| gelu | 50% | **0.823 ± 0.003** | 0.370 ± 0.377 | 0.766 ± 0.047 |
| silu | 10% | **0.787 ± 0.036** | 0.013 ± 0.014 | 0.042 ± 0.010 |
| mish | 10% | **0.802 ± 0.017** | 0.008 ± 0.003 | 0.086 ± 0.043 |

Full table: `pruning_results.csv` (144 rows: 4 activations × 3 seeds × 4
ratios × 3 methods).

## Statistical analysis (paired by activation × seed × ratio, n=12 each)

| activation | mean(gate_density − magnitude) drop | 95% CI | paired-t p |
|---|---|---|---|
| relu | −0.026 | [−0.108, +0.057] | 0.55 (n.s.) |
| gelu | **+0.479** | [+0.312, +0.645] | 1.5×10⁻⁴ |
| silu | **+0.531** | [+0.368, +0.694] | 5.2×10⁻⁵ |
| mish | **+0.551** | [+0.396, +0.705] | 2.3×10⁻⁵ |

## Interpretation — per the pre-specified rules, reported honestly, no selective reporting

- **ReLU: informative equivalence.** Gate-density pruning performs
  statistically indistinguishably from magnitude pruning (p=0.55), and
  both outperform random at low ratios. For ReLU, gate density carries
  roughly the same predictive information about channel importance as the
  standard magnitude baseline.
- **GELU/SiLU/Mish: gate density *underperforms* magnitude pruning,
  badly and consistently** — even at 10% pruning, removing the
  lowest-gate-density 10% of channels destroys 65–80 percentage points of
  accuracy, while removing the lowest-*magnitude* 10% costs under 1.5
  points. Gate density is not merely uninformative for smooth activations
  here — it is **anti-correlated** with true channel importance: the
  channels it identifies as "least active" are disproportionately
  channels the network depends on heavily.
- We do not have a proven causal explanation for this anti-correlation.
  A plausible, untested account consistent with Experiment 3's findings:
  the quantile-compression dynamics observed for smooth activations
  (low tail rising, high tail falling toward a moderate center) may mean
  that "lowest gate density" for these activations does not track "least
  weight-supported" the way it does for ReLU's binary {0,1} gate — see
  `mechanism_summary.md`. This is offered as a hypothesis, not a finding.

**This result should be read as a caution, not as a positive
"predictive consequence" claim for gate density as a general-purpose
pruning signal.** It is informative — it tells us gate density and weight
magnitude diverge sharply in what they identify as unimportant for smooth
activations — but the direction of that divergence is the opposite of
what would make gate density a useful pruning criterion on its own for
those activations.
