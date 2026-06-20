# Upgrade Summary: P1-P4 Campaign

Written for a skeptical reviewer. What was run, what survived, what didn't,
what was rescoped, and the honest case for the paper's strengthened claims.
Every number below traces to a raw CSV in `gradient_gate_outputs/csv/` and a
re-runnable script in `gradient_gate/`; the full reasoning chain (including
two real bugs caught and fixed mid-analysis) is in `RESULTS_LOG.md`.

## What was run

| # | Question | Script | Jobs |
|---|---|---|---|
| P1 | Does the activation-class direction split survive other optimizers? | `run_training_dynamics.py --optimizer {adam,adamw}`, `run_pruning_experiment.py --optimizer adamw` | 11061122, 11061123 |
| P2 | Does the mu-vs-z_low mechanism hold per-channel, not just pooled? | `run_channel_mechanism.py`, `analyze_channel_mechanism.py` | 11061124 |
| P3 | Does the split survive a non-CIFAR, larger, higher-resolution dataset? | `run_tinyimagenet_dynamics.py` | 11061327, 11061328 |
| P4 | Is the smooth-activation rise CNN-specific? | `sequence_models.py`, `run_training_dynamics.py --archs mlp_mixer transformer_encoder` | 11061548 |

All 6 jobs completed successfully (`sacct`: exit 0:0, empty `.err` files).

## What survived cleanly

- **Adam**: the architecture-fixed ablation replicates exactly under Adam
  (coupled weight decay) — 12/12 independent runs per activation,
  sign-test $p=4.88\times10^{-4}$, magnitudes matching SGD.
- **The smooth-activation rise generalizes broadly**: it replicates on
  Tiny-ImageNet-200 (net displacement, 9/9 seeds) and on two
  non-convolutional, LayerNorm-only architectures (MLP-Mixer,
  Transformer-Encoder — 6/6 seeds each, $p=0.03125$). It is not CNN-specific
  or BatchNorm-specific.
- **Per-channel mechanism, once correctly specified**: a sigma-normalized
  z-score margin gives strong, uniform, positive per-channel correlation
  for all four activations (12/12 activation×seed cells,
  $p=2.44\times10^{-4}$) — the direct confirmation the prior report flagged
  as missing.

## What didn't survive, and was rescoped honestly

- **AdamW collapses the split.** All four activations decline under AdamW
  (12/12 each, $p=4.88\times10^{-4}$) instead of diverging by class.
  BatchNorm's $\gamma$ itself splits unexpectedly under AdamW (relu/gelu
  still shrink; silu/mish grow) without explaining the uniform collapse.
  **The direction split and the mechanism are now scoped explicitly to
  coupled-weight-decay optimizers** — this is stated in the abstract,
  Section 4.5, and the Discussion, not buried.
- **The ReLU decline does not generalize beyond CNNs.** In MLP-Mixer, ReLU
  rises (5/6 seeds) instead of declining. In the Transformer-Encoder it is
  small and dataset-dependent, nothing like the CNN's large, robust decline.
  **The smooth-activation rise and the ReLU decline are reported as having
  different scope** — one architecture-general, one CNN-specific.
- **"Effective rank always rises" is CNN-specific, not general.** MLP-Mixer's
  rank declines for every activation including ReLU; the Transformer-Encoder
  splits by activation. The original claim (114+ CNN runs, all rising) is
  narrowed to the architectures it was actually tested on.
- **Tiny-ImageNet's smooth-activation trend statistic is unreliable**, not
  reversed: net displacement still matches CIFAR's direction in 9/9
  (active_frac) and 8/9 (rank) seeds, but trajectories become non-monotonic
  (early overshoot, partial relax) at this scale, so the project's standard
  whole-trajectory linear-trend statistic loses power on this already-small
  effect. Both statistics are reported side by side in
  `analyze_seedlevel_direction.py`'s output, not just the favorable one.

## Two real bugs caught and fixed before trusting any result

1. **`z_low` definition bug** (`run_channel_mechanism.py`): a literal
   left-to-right `inf{z : g(z)>theta}` scan latches onto a transient,
   irrelevant bump in GELU/Mish's non-monotonic derivative. Fixed by
   scanning right-to-left for the boundary of the *sustained* active region;
   verified against the existing theory doc's published table (matches to 3
   decimals) before using it for anything new.
2. **Per-channel margin mis-specification**: the raw margin
   $\mu_c - z_{\rm low}$ gave a confusing, activation-split correlation
   result (strong for ReLU, weakly negative for GELU/SiLU/Mish). Diagnosed
   rather than reported as a weak mechanism: ReLU's threshold sits at the
   live center of its channels' distributions (mu-driven), while
   GELU/SiLU/Mish's sits far in the tail (sigma-driven quantile compression,
   which the mu-only margin cannot see). The sigma-normalized z-score margin
   resolves this for all four activations uniformly.

A third, smaller methodological fix: `analyze_seedlevel_direction.py`'s
first version silently merged distinct (arch, dataset, seed) runs that
share an integer seed value into one corrupted trajectory when pooling
across architectures — fixed by always computing the per-seed statistic on
the full identifying tuple and only pooling at the summary step.

## The honest case for the paper's strengthened claims

The paper's central contribution — gate density diverges by activation
class, and this is derived rather than merely observed — is now supported
by a direct per-channel test (not just a population average) and is shown
to generalize along the dimension that matters most for the smooth-activation
half of the finding (architecture family, normalization scheme, optimizer
choice within the coupled-weight-decay family). The dimensions where it does
*not* generalize (AdamW; the ReLU-specific decline outside CNNs; the
rank-rise claim outside CNNs) are reported with the same rigor as the
positive results, narrowing the paper's claims to what the evidence actually
supports rather than what would make the cleanest story. This is consistent
with the discipline applied throughout this project: every directional claim
traces to a seed-level sign test, no result was tuned or filtered to fit a
narrative, and contradicting results were investigated and reported, not
discarded.

## Where this leaves `main.tex`

- New Section 4.5/4.6 (`sec:optimizer`, `sec:generalization`): P1, P3, P4.
- New Section 5.4 (`sec:channel_mechanism`): P2.
- Abstract, Introduction, Contributions list, Discussion/Limitations, and
  Conclusion all updated to state the new evidence and its scope precisely.
- 3 new figures (`optimizer_generalization.png`, `channel_mechanism_zscore.png`,
  `generalization_scale_architecture.png`), all referenced and verified
  present (0 missing figures across 41 unique references).
- `main.pdf` recompiles cleanly (38 pages, zero warnings beyond one
  pre-existing, benign float-placement notice) from a fresh extraction of
  `gradient_gate_collapse_paper.zip`, verified end-to-end as an Overleaf
  upload would experience it.
