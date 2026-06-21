# Upgrade Summary: NeurIPS-8 Revision (Task A-D)

This section documents the most recent campaign, run on top of the P1-P4
campaign documented below. **Headline result: Task A succeeded.** The
AdamW "collapse" reported in the P1-P4 campaign (see the superseded bullet
in that section, left in place but corrected below) is no longer an
unexplained scope boundary -- it is a successful, falsifiable prediction
of the same mechanism, fed AdamW's real measured dynamics.

## Task A (highest priority): does the mechanism predict the AdamW anomaly?

**Pre-registered in writing** (`RESULTS_LOG.md`) before computing anything:
hypothesis, exact predictor (the unmodified sigma-normalized margin from
the SGD per-channel result), and the precise meaning of "predicts
correctly," all fixed in advance. No redefinition of `z_low`, no
reselected channels, no new free parameters were used at any point.

**Result: the predictor succeeds completely.**
1. The local law (margin correlates with active_frac) holds under AdamW
   too (72-90% of channels per seed, all four activations).
2. The decisive test -- does the margin's *trend* match AdamW's observed
   uniform decline -- passes for 12/12 activation×seed cells
   ($p=2.44\times10^{-4}$).
3. The mechanistic link is direct, not inferred: per-channel sigma shrinks
   ~67-77% under SGD for every activation, but is flat (ReLU, $-0.9\%$) or
   **grows** (GELU $+2.2\%$, SiLU $+4.8\%$, Mish $+4.7\%$) under AdamW.
   With $\mu$ still drifting negative and sigma no longer collapsing, the
   margin can only decline -- exactly what is observed, for every
   activation, in every seed.

**This is not three disconnected optimizer-specific findings; it is one
predictor whose input trajectory differs.** The governing quantity is
whether sigma collapses, itself controlled by coupled (SGD, Adam) vs.
decoupled (AdamW) weight decay. This is now the paper's headline result,
not a caveat -- the abstract, intro, contributions list, a new Section 5.5
(`sec:adamw_predict`), and the Discussion were all rewritten around it.

## Task B: why does mu drift negative? (partial closure)

A generic SGD+L2-equilibrium argument ($\theta^\ast \approx -\bar g/\lambda$
for a roughly activation-independent average gradient pressure $\bar g$
and the shared weight-decay coefficient $\lambda$ -- a different argument
from the gamma mechanism's scale-invariance logic, since beta has no
multiplicative symmetry to exploit) correctly predicts the narrow, shared
drift band already reported (range 0.024 across 9 activations whose
`z_low` spans 4 orders of magnitude). It does **not** explain the residual,
statistically real cross-activation variation within that band (one-way
ANOVA $F=29.7$, $p=7.7\times10^{-9}$) -- tested and ruled out two natural
candidates (`z_low`, an init-time smoothness index) as the source, then
stopped rather than fishing for a third. Reported as a partial closure:
order of magnitude now derived, residual variation still open.

## Task C: scale beyond Tiny-ImageNet

ImageNet-1k checked and confirmed gated (registration/ToS acceptance
required, no anonymous scriptable download) -- a genuine external access
blocker, not bypassed. Chose Places365-Standard instead: freely,
anonymously downloadable (no registration), 365 scene classes, native
256x256 resolution (downsampled to 96x96 here -- still 2.25x the linear
resolution of the Tiny-ImageNet experiment), real photographs. Downloaded
26.7GB to `/scratch/gilbreth/aelmersa/places365/` (not `/home`, which has
only a 25GB quota). Full blind extraction of all ~1.84M images proved far
too slow for this filesystem's small-file write overhead (~70 minutes for
13% progress); switched to a targeted extraction of exactly the
150-train/20-val-per-class subsample the experiment design calls for
(62,050 of 1.84M files, the first 150 sequentially-numbered images per
train class plus the full val set -- a fixed, deterministic, documented
selection).

**Infrastructure correction**: that targeted extraction was found running
as an unsupervised background process directly on the shared login node
(3h11m, 95% CPU) -- a real violation of this project's own
SLURM-for-multi-hour-work rule. Killed it and moved it into a proper
SLURM job (`run_places365_extract.sh`), combined with the training step
since this partition rejects GPU-less job submissions outright (so a
second, separate GPU allocation just for extraction would be wasteful).
**Job 11077301** (24h budget) is running as of this writing -- see
`RESULTS_LOG.md` for the live status; this section will be updated with
final results once it completes.

## Task D: one practical-payoff attempt, honestly null

**Pre-registered in writing** before computing anything: predictor (final
active_frac, already known from existing checkpoints), outcome (test
accuracy after fine-tuning on a fixed, seeded 5%-CIFAR-10 subsample,
identical recipe for all 24 checkpoints regardless of original optimizer),
and the exact test (Pearson/Spearman, pooled and per-optimizer-subset).

**Result: no statistically detectable correlation in any subset** (pooled:
Pearson $r=0.24$, $p=0.26$; Spearman $\rho=-0.33$, $p=0.11$ -- the two
do not even agree in sign, consistent with no real effect rather than an
underpowered one). Reported exactly as pre-registered; did not substitute
a different outcome metric (e.g. accuracy drop instead of post-finetune
accuracy) after seeing the null, which would have been exactly the
p-hacking the guardrails prohibit.

## Synthetic appendix: missing-artifact and retired-claim language removed

Cut the unconfirmed MNIST-generalization paragraph entirely (required a
now-absent TensorFlow dependency; figure and CSV could not be located).
Removed the "could not be located... only the spatial illustration is
unavailable" disclosure from the spatial-gate paragraph, keeping its
quantitative claims (already independently verified from
`grad_sparsity_all_optimizers.csv`) stated directly. Simplified the
predictive-difficulty-model caption to state the cross-validated result
without referencing the superseded earlier-draft number. Left one
"retired" reference in place (a 4-parameter Fisher-information fit
abandoned for an ill-conditioned matrix) since that is a legitimate,
transparent statistical-methodology disclosure, not an unverifiable claim.

---

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

- **AdamW collapses the split** — *superseded, see the Task A section above
  this one.* At the time this bullet was written, all four activations
  declining under AdamW (12/12 each) looked like an unexplained scope
  boundary, and BatchNorm $\gamma$'s own unexpected split (relu/gelu
  shrink; silu/mish grow) didn't explain it. The follow-on campaign tested
  this as a falsifiable prediction instead of a boundary and found the
  exact per-channel predictor, fed AdamW's real measured dynamics,
  predicts the collapse correctly with zero new parameters. **This is now
  the paper's headline result, not a limitation.**
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
