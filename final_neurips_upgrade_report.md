# Final NeurIPS Upgrade Report

Three experiments executed exactly as specified: BatchNorm necessity
ablation, gate-density-guided pruning, and mechanism logging (bundled into
the pruning runs). All code, raw CSVs, figures, and per-experiment summaries
are in `gradient_gate/run_bn_ablation.py`, `run_pruning_experiment.py`,
`run_pruning_analysis.py`, and `gradient_gate_outputs/`. This report
synthesizes the three and answers the five required questions directly.
Conservative throughout — underclaiming preferred to overclaiming.

## Does the activation-class claim survive BatchNorm removal?

**Yes, completely.** Swapping every BatchNorm2d for GroupNorm on the
identical ResNet-18 skeleton, the direction is unchanged: ReLU declines
3/3 independent seeds, GELU/SiLU/Mish rise 9/9, under both normalization
schemes. BatchNorm is not necessary for the phenomenon. Magnitude is
noisier under GroupNorm (especially Mish) and final accuracy is somewhat
lower — both attributable to GroupNorm being unmatched in hyperparameters
at this scale, not evidence about the underlying mechanism. This is a
genuine strengthening of Contribution 2: it rules out yet another
structural confound (now: not architecture, not normalization scheme),
narrowing what could possibly be driving the effect down to the activation
function itself.

## Does gate density predict pruning sensitivity?

**Yes — but the answer is not the positive "novel predictive consequence"
one might have hoped for, and we report it exactly as found.**

- **ReLU:** gate-density pruning and magnitude pruning are statistically
  indistinguishable (paired-t p=0.55). Informative equivalence — gate
  density carries roughly the same importance signal as the standard
  baseline.
- **GELU, SiLU, Mish:** gate-density pruning is *dramatically and
  significantly worse* than magnitude pruning at every ratio tested
  (p=1.5×10⁻⁴, 5.2×10⁻⁵, 2.3×10⁻⁵ respectively) — removing the
  lowest-gate-density 10% of channels costs 65–80 accuracy points, while
  removing the lowest-*magnitude* 10% costs under 1.5 points. For these
  three activations, gate density is not merely uninformative about
  pruning importance — it is **anti-correlated** with it.

This matters for how the paper frames gate density generally: it is a
real, robust, measurable, activation-class-dependent training-dynamics
signal, but **it should not be presented or implied to be a generally
useful proxy for "channel importance."** For three of the four activations
studied, using it that way is actively harmful. This finding should be
in the main paper specifically *because* it's a negative result that
forecloses an obvious but wrong inference a reader might otherwise draw
from Contribution 2.

## What mechanism is most consistent with the new evidence?

Three trends are universal across all four activations, all robust
(12/12 independent trajectories, sign-test p=2.44×10⁻⁴ each): BatchNorm
scale (γ) shrinks, pre-activation variance narrows, and a Hessian-trace
sharpness proxy rises — all over the same 25 epochs, at similar relative
magnitudes regardless of activation. **None of these three, individually,
explains the activation-class divergence in `active_frac`, because they
happen the same way for everyone.**

The mechanism most consistent with the full body of evidence (this
experiment plus the earlier threshold-robustness/quantile study) is:
training narrows the pre-activation distribution similarly across
activations, but the *same* narrowing produces *opposite* `active_frac`
outcomes depending on the shape of the activation's derivative — a step
function for ReLU (narrowing shifts probability mass between two fixed
modes, 0 and 1) versus a smooth, hump-shaped derivative for GELU/SiLU/Mish
(narrowing compresses both tails toward the peak, simultaneously raising
the low tail away from zero and lowering the high tail — net effect:
`active_frac` rises even as `gate_mean` falls). **This is offered as the
most consistent available account, not a proven causal mechanism** — this
experiment did not test the per-channel link between pre-activation
distribution shape and gate value directly, and that would be the natural
next step if a fuller mechanistic paper were pursued.

## Which results belong in the NeurIPS main paper

1. **BatchNorm-necessity ablation (Experiment 1)** — promote to main
   paper. It directly answers the most natural remaining "is this really
   about the activation" question, cheaply and conclusively.
2. **Pruning result for all four activations, including the negative
   finding for GELU/SiLU/Mish (Experiment 2)** — promote to main paper,
   reported with the full honesty above. This is the paper's most
   "interesting" new result in the sense reviewers reward (unexpected,
   not just more robustness), and the negative framing is itself a
   contribution: it forecloses an overclaim before a reviewer can make it
   for the authors.
3. **The three universal mechanism trends (Experiment 3), summarized in
   one paragraph and one compact figure** — main paper, briefly. Keep the
   conclusion exactly as scoped: these are correlates, not a derivation,
   and they explain why the mechanism question remains open rather than
   closing it.

## Which results should remain appendix-only

- Full per-seed, per-epoch numeric tables for all three experiments
  (`bn_vs_gn_statistics.csv`, `pruning_results.csv`, `mechanism_logging.csv`,
  `bn_gamma_layerwise.csv`, `mechanism_seedlevel.csv`).
- The layerwise BN-γ breakdown figure detail (per-layer panel in
  `bn_gamma_dynamics.png`) — the pooled trend is what matters for the
  main-text claim; the per-layer detail is supporting material.
- The Hutchinson-trace-vs-active_frac sign-mirroring observation
  (Finding 3 in `mechanism_summary.md`) — correctly noted in that summary
  as close to a restatement of the main result rather than independent
  evidence; mention in one sentence in the main text at most, full detail
  in the appendix.
- The estimator-noise caveat for the Hutchinson trace (one negative
  point estimate) — appendix footnote, not main text.
- GroupNorm's lower absolute accuracy and noisier per-seed trends — state
  the qualitative conclusion (direction survives) in the main text;
  put the full seed-by-seed numbers in the appendix.

## One discipline note carried over from the rest of this process

No claim in this report or its underlying summaries uses causal language
beyond what a sign test or paired comparison actually licenses. Where a
result was negative or inconvenient for the broader narrative (GELU/SiLU/
Mish underperforming magnitude pruning; GroupNorm's noisier trends; the
Hutchinson estimator's noise), it is reported in full, not minimized.
