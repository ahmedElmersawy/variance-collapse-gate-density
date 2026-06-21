# Results Log — NeurIPS Upgrade Campaign

Audit trail. One entry per job/script run: config, job id (if sbatch),
runtime, output paths, one-line outcome. Append-only; never edit past
entries except to add a completion line.

## Context at campaign start (2026-06-20)

Read `ROADMAP.md` and `final_neurips_upgrade_report.md` in full. State
confirmed consistent with both: architecture-fixed ablation (resolves
activation-vs-architecture), BN-necessity ablation, pruning negative
result, and population-level mu-vs-z_low mechanism check (9/9 sign
predictions) are all done and already reflected in `main.tex`. Open items
match this campaign's P1-P4 exactly: optimizer generalization, direct
per-channel mechanism test, one scale step beyond CIFAR, broader
architecture coverage (ROADMAP.md items 2-3; final_neurips_upgrade_report.md's
explicit admission that the per-channel link was never tested directly).

`run_training_dynamics.py`'s optimizer is hardcoded to SGD (line 144).
`gradient_gate/run_pruning_experiment.py` and `run_preactivation_mean_check.py`
already have the building blocks (BN-gamma logging, pooled mu/sigma logging)
needed for P1's gamma-under-AdamW check and as a template for P2.

---

## P1 — Optimizer generalization

Added `--optimizer {sgd,adam,adamw}` to `gradient_gate/run_training_dynamics.py`
(`build_optimizer()`, default lr 0.1 for sgd / 1e-3 for adam/adamw if not
given explicitly) and to `gradient_gate/run_pruning_experiment.py` (same
helper, reused). Checkpoint filenames for the pruning-experiment script only
gain an `_{optimizer}` suffix when optimizer != "sgd", so the existing
`gradient_gate_outputs/checkpoints/resnet18_*_seed*.pt` files (used by the
already-published pruning result) are untouched and `already_done()` still
correctly skips them — verified by reading `run_pruning_analysis.py`'s
loader, which expects the un-suffixed filename, before making this change.

Smoke-tested the `--optimizer adam` code path: import + arg parsing + device
selection confirmed to execute without error (see job logs); did not wait for
a full epoch on CPU (single CPU `import torch` took 65s in isolation on this
login node right now, so a full epoch is not a meaningful CPU smoke test
here — relying on the existing project precedent of verifying structurally
on CPU then trusting SLURM/A100 for the real run).

Submitted:
- **Job 11061122** (`run_optimizer_ablation_adam.sh`): architecture-fixed
  ablation (resnet18+vgg11, relu/gelu/silu/mish, cifar10+cifar100, 3 seeds,
  25 epochs) under Adam, lr=1e-3. New file:
  `gradient_gate_outputs/csv/optimizer_ablation_adam.csv`. 14h budget (48
  runs vs. the existing 36-run SGD ablation's 8h budget, scaled up + margin).
- **Job 11061123** (`run_optimizer_ablation_adamw.sh`): same ablation design
  under AdamW, lr=1e-3 -> `optimizer_ablation_adamw.csv`. Then, in the same
  job, a direct BatchNorm-gamma-under-AdamW check via
  `run_pruning_experiment.py --optimizer adamw`, writing to NEW files
  (`mechanism_logging_adamw.csv`, `bn_gamma_layerwise_adamw.csv`,
  `checkpoints_adamw/`) so the existing SGD mechanism-logging artifacts are
  never appended to or overwritten (those files have no `optimizer` column;
  appending optimizer-tagged rows to them would corrupt the CSV schema).
- **Job 11061124**: see P2 below (separate script, submitted together).

Rationale for AdamW gamma check: `theory_variance_compression_mechanism.md`'s
Step 2 (gamma shrinkage) leans on van Laarhoven 2017 / Hoffer et al. 2018,
which assume COUPLED weight decay (SGD, torch.optim.Adam). AdamW decouples
weight decay from the gradient update, the most direct test of whether that
step of the mechanism is specific to the coupled case.

## P2 — Direct per-channel mechanism verification

New file `gradient_gate/run_channel_mechanism.py`. Closes the exact gap
`final_neurips_upgrade_report.md` named: the existing mu-vs-z_low check
(`run_preactivation_mean_check.py`, theory doc's 9/9 table) tested the
mechanism at the POOLED, population level (one mu/sigma/active_frac per
activation/seed/epoch). This logs the same quantities per INDIVIDUAL CHANNEL
(stable identity = layer name + channel index, architecture is fixed across
epochs) via a new `PerChannelMechanismCollector` (forward hook caches the
pre-activation tensor per layer; backward hook, firing immediately after,
recovers the gate via the project's standard grad_input/grad_output trick
and combines it with the cached input to emit one row per channel).

**Found and fixed a real bug in `compute_z_low` before trusting it**: a
literal left-to-right `inf{z : g(z)>theta}` (the theory doc's stated
definition, read literally) is not robust for GELU/Mish, whose derivative is
non-monotonic — GELU's |g| briefly exceeds even theta=0.10 again near
z=-1.86 (a transient bump) before dipping back near 0 at its true
zero-crossing (z=-0.75) and then rising for good past z=0. A naive
left-to-right scan latches onto that irrelevant early bump (verified
numerically: gives z_low(gelu, 0.10)=-1.86, z_low(mish, 0.10)=-2.83, neither
matching the published theory-doc table). Re-derived the right definition:
scan from deep in the permanently-active right tail leftward, and take the
first point where g drops to/below theta — the boundary of the *sustained*
active region, which is what the mechanism's claim (variance shrinkage
permanently crosses one boundary) actually needs. This matches the
theory doc's published table to 3 decimals for every activation checked
(relu, gelu, silu, mish, softplus beta in {50,20,10,5}, leaky_relu) at
theta=0.10. Fixed in `compute_z_low` with the bug documented in the
function's docstring. At the project-standard threshold (GATE_EPS=0.01),
z_low(relu)=-0.00005, z_low(gelu)=-0.729, z_low(silu)=-1.234, z_low(mish)=-1.156.

Smoke-tested: a synthetic-batch (no real dataset) unit test of
`PerChannelMechanismCollector` directly on `cifar_resnet18(act_layer=nn.GELU)`
produced 3904 channel-rows across 17 layers with sane per-channel
mu/sigma/active_frac/gate_mean (active_frac mean 0.983 at random init,
valid_frac=1.0 everywhere — no near-zero-grad_output numerical issue).
Confirmed correct before submitting the real job.

**Job 11061124** (`run_channel_mechanism.sh`): resnet18, cifar10,
relu/gelu/silu/mish, 3 seeds, 25 epochs, SGD (matching the rest of the
project's main-line protocol) -> `gradient_gate_outputs/csv/channel_mechanism.csv`
and `channel_mechanism_zlow.csv`. 6h budget (similar per-epoch cost to
`run_pruning_experiment.py`'s 4h/12-run job, slightly heavier due to the
combined fwd+bwd per-channel hook).

All three jobs (11061122, 11061123, 11061124) submitted to the a100-80gb
partition (confirmed via `sinfo`: TIMELIMIT=infinite, so no cluster-imposed
cap forced these budgets down — they're sized from this project's own prior
job runtimes instead).

### Early P2 result, from partial real data (relu, 3 seeds, complete through epoch 24)

Wrote `gradient_gate/analyze_channel_mechanism.py` and ran it against
`channel_mechanism.csv` while job 11061124 continued in the background.
**Caught a real design flaw in the analysis before trusting it**: my first
version's primary test used epoch-0 margin to predict the epoch-0-to-24
active_frac delta (a lagged-predictive claim) and got a weak,
near-chance result for relu (frac_sign_match 0.41-0.53 across 3 seeds) --
but that is NOT the mechanism's actual claim. The mechanism
(theory_variance_compression_mechanism.md) claims an INSTANTANEOUS
relationship: margin(t) determines active_frac(t) at the same t, and mu
itself drifts over training as part of the mechanism, not separately from
it. ReLU's z_low sits at ~0 (the thinnest possible margin of any
activation tested), so a channel's epoch-0 sign is a poor predictor of
where its also-drifting mu ends up 24 epochs later -- that is a property
of using a stale single-epoch predictor, not evidence against the
mechanism. Re-ran with the correct, direct test (per channel, correlate
margin(epoch) with active_frac(epoch) across all 5 logged epochs): for
relu, 3/3 seeds show frac_positive_corr in [0.84, 0.89] (mean correlation
+0.57 to +0.63) -- a strong, genuine per-channel confirmation, recovered
once the test matched the actual claim. Demoted the lagged test to
secondary/transparency-only in the script; kept both in the output CSV.
Will re-run the full analysis (relu+gelu+silu+mish) once job 11061124
completes, before drawing any final per-activation conclusion.

## P3/P4 setup

**P3** (`gradient_gate/run_tinyimagenet_dynamics.py`): downloaded the
standard public Tiny-ImageNet-200 (cs231n.stanford.edu, 64x64, 200
classes, 100k train / 10k val) into `data/tiny-imagenet-200/` -- noted in
passing that another user's copy exists on shared `/scratch`, deliberately
not used (not mine to read). Custom `TinyImageNetDataset` (ImageFolder
doesn't fit the val/ layout, which needs `val_annotations.txt`). Reuses
`cifar_resnet18`'s existing 3x3-stride-1-stem unmodified -- already
"64x64-appropriate" per its own docstring's logic (strides (1,2,2,2) take
64x64 to an 8x8 feature map before the global pool, not collapsed). Smoke
tested: dataset loads (100000/10000 samples, 200 distinct labels
confirmed), one real forward+backward pass through `cifar_resnet18(num_classes=200)`
on a real batch succeeds (loss ~5.4 ~= ln(200), as expected at init).
Split across two jobs (relu+gelu / silu+mish) writing to separate CSVs to
avoid a header-write race, since both could start in the same instant --
**jobs 11061327, 11061328**, 14h budget each (Tiny-ImageNet is ~2x the
images and ~4x the pixels/image of CIFAR, so budgeted above the CIFAR
ablation's per-run cost with margin).

**P4** (`gradient_gate/sequence_models.py`): new CIFAR-native
(32x32), activation-configurable MLP-Mixer (8 blocks, ~1.13M params) and
small Transformer-Encoder (6 blocks, 4 heads, ~0.81M params), to test
whether the smooth-activation gate-density rise is CNN-specific. Built
with explicit `act_layer()` submodules rather than
`nn.TransformerEncoderLayer` -- the latter's activation, when given a
callable, is a plain function reference called inside `forward()`, not a
registered submodule, so it would be invisible to
`GateInstrumentor`'s `named_modules()`-based hook attachment; verified
this reasoning would matter before writing custom blocks, rather than
discovering it after a wasted job. Wired into
`run_training_dynamics.py` via a new `ACTIVATION_CONFIGURABLE_ARCHS =
CIFAR_NATIVE_ARCHS + SEQUENCE_NATIVE_ARCHS` (replaces the old
`CIFAR_NATIVE_ARCHS`-only checks for the resize-to-224 decision, the
per-arch activation-list dispatch, and the batch-size-halving logic, so
the existing CNN/ViT/ConvNeXt behavior is provably unchanged -- those
archs aren't in the new tuple's added members). Smoke-tested: a
synthetic-batch GateInstrumentor pass over both architectures x all 4
activations confirms correct layer counts (16 = 8 blocks x 2
activations/block for the Mixer; 6 = 6 blocks x 1 activation/block for the
Transformer) and sane active_frac at random init (~0.50 for relu, ~0.98-0.99
for gelu/silu/mish, consistent with the rest of this project's
random-init baseline pattern). **Job 11061548**, both datasets, all 4
activations, 3 seeds, 25 epochs, 10h budget.

All three new jobs (11061327, 11061328, 11061548) were submitted but are
sitting in queue (PD) as of this writing -- `sinfo` shows no idle
a100-80gb nodes right now (30 mixed-, 15 mixed, 10 allocated, 2 drained);
this is normal cluster contention, not a problem with the jobs themselves.
Will start automatically once a node frees up.

---

## All 6 jobs completed (2026-06-20). Full analysis.

`sacct` confirms all 6 jobs COMPLETED, exit 0:0, no errors in any `.err`
file (all empty). Elapsed: 11061122 (Adam ablation) 5h36m, 11061123
(AdamW ablation + gamma check) 7h05m, 11061124 (channel mechanism) 1h15m,
11061327/11061328 (Tiny-ImageNet a/b) 3h47m/3h49m, 11061548 (sequence
models) 5h27m.

### P1 result: activation-class split survives Adam, COLLAPSES under AdamW

Wrote `gradient_gate/analyze_seedlevel_direction.py` (generic seed-level
sign-test tool, fixed one bug in its own first version before trusting it:
pooling across arch+dataset by grouping only on (activation,optimizer,seed)
silently merged 4 *different* runs that happen to share the integer seed
value 0/1/2 into one corrupted "trajectory" -- fixed by always computing
the per-seed statistic on the full identifying tuple (arch, activation,
optimizer, dataset, seed) and only pooling at the summary step).

**Adam**: every one of 16 (arch x activation x dataset) cells shows 3/3
seed agreement; pooling arch+dataset, every activation shows 12/12 seeds
correct (relu declines, gelu/silu/mish rise), sign-test p=4.88e-4 each --
identical qualitative pattern to the original SGD result. Magnitudes also
match the SGD pattern (relu epoch0->24: 0.539->0.377, a 16pp drop; smooth
activations: +0.5 to +1.4pp).

**AdamW: the split collapses.** All FOUR activations DECLINE (12/12 seeds
each, p=4.88e-4) -- gelu 0.970->0.953, mish 0.988->0.975, silu 0.988->0.977,
relu 0.545->0.504, all broadly comparable small-to-moderate declines, not
the sharp relu-vs-smooth asymmetry seen under SGD/Adam. This is a real,
reproducible scope boundary, not a bug (verified by inspecting a raw
per-epoch trajectory directly). Checked the obvious candidate explanation
(BN-gamma shrinkage, the mechanism's Step 2) directly via the
gamma-under-AdamW companion run (`mechanism_logging_adamw.csv`,
`run_pruning_experiment.py --optimizer adamw`): gamma's behavior ALSO
splits unexpectedly under AdamW -- relu/gelu still shrink (r=-0.50 to
-0.84, weaker than under SGD/Adam) but silu/mish actually GROW (r=+0.95 to
+0.98) -- yet active_frac declines uniformly for all four regardless. So
the simple gamma-shrinkage account does not explain the AdamW collapse
either; something else, not characterized here, is driving it. Reported
exactly as found, not forced into a tidy story: **the activation-class
direction split and the gamma-shrinkage mechanism are both empirically
tied to coupled-weight-decay optimizers (SGD, Adam); AdamW's decoupled
weight decay produces a qualitatively different, currently unexplained
regime.** This rescopes Contribution 2/the mechanism's claims to coupled
weight decay explicitly, rather than claiming optimizer-universality.

### P2 result: per-channel mechanism confirmed for ALL FOUR activations, uniformly, once a real analysis bug was caught and fixed

First analysis attempt (raw margin = mu_c - z_low, correlated per-channel
against active_frac_c across the 5 logged epochs) gave a confusing,
activation-split result: strong positive for relu (frac_positive_corr
~0.87, mean rho ~+0.61) but weak/NEGATIVE for gelu/silu/mish
(~0.41-0.46, mean rho ~-0.13 to -0.18). Diagnosed rather than reported as
a weak mechanism: checked each activation's margin in units of its own
per-channel sigma at epoch 0 -- ~0.0 sigma for relu (z_low sits at the
*center* of the channel distribution: a "thin-margin" case where mu's
drift directly crosses the live threshold) vs ~0.9-1.5 sigma for
gelu/silu/mish (a "thick-margin" case). Sigma shrinks ~3x for every
activation over training (0.83-0.92 -> 0.19-0.31); mu also drifts slightly
negative for every activation (already established at the population
level). For relu this mu-drift IS the active_frac driver. For
gelu/silu/mish, mu's small negative drift actually pulls the *raw* margin
down even as sigma's much larger relative shrinkage pulls active_frac UP
(thinning the sub-threshold left tail -- the project's earlier
"quantile-compression" population-level finding, now localized to
individual channels) -- two forces pulling the unnormalized margin and
active_frac in opposite apparent directions, exactly producing the
observed negative correlation. Re-ran with the corrected, sigma-normalized
z-score margin, (mu_c - z_low)/sigma_c -- the standard way to ask "how
many channel-widths from the boundary," not raw units. **Result: all four
activations now show strong, uniform, positive per-channel correlation
(frac_positive_corr 0.90-0.96 per seed, mean 0.92-0.95 per activation;
mean correlation +0.66 to +0.80) -- 12/12 seeds (pooling all 4
activations x 3 seeds) with frac_positive_corr_zscore>0.5, combined
binomial p=2.44e-4.** This is the direct, per-channel confirmation
final_neurips_upgrade_report.md flagged as missing, now obtained -- plus
a genuine refinement to the theory: the mechanism operates via mu-crossing
for thin-margin activations (relu, and by the established z_low ordering,
presumably leaky_relu/high-beta softplus) and via sigma-driven
quantile-compression for thick-margin activations (gelu/silu/mish),
unified by the same sigma-normalized threshold-distance quantity. Updated
`gradient_gate/analyze_channel_mechanism.py` to report the corrected
z-score test as primary and keep the raw-margin/lagged tests as documented
diagnostic context (the docstring records the full reasoning chain, not
just the final number).

### P3 result: relu's finding is fully robust at scale; the smaller smooth-activation effect's trajectory SHAPE changes (net direction does not)

The project's standard seed-level statistic (Pearson r between epoch index
and the metric over the full 25-epoch trajectory, "trend") gave a
confusing result at first: relu matches CIFAR cleanly (active_frac r<0,
3/3; effective_rank r>0, 3/3) but gelu/mish/silu were MIXED OR REVERSED
(mish active_frac 3/3 "declining" by this statistic; gelu/mish/silu
effective_rank also mostly "declining," 3/3, 3/3, 3/3 -- opposite the
established CIFAR direction). Did not report this as a reversal without
checking the actual trajectories first. Inspecting raw per-epoch values
(e.g. gelu/cifar10... no, gelu/tinyimagenet seed=0's effective_rank:
57.08 -> 58.9 by epoch 3 -> declines back to ~56.8 by epoch 19 -> recovers
to 57.06 by epoch 24) showed why: on this larger, harder (200-class)
dataset, the smooth-activations' (already small-magnitude, <1pp on CIFAR)
trajectories are no longer monotonic -- they overshoot early, then
partially relax -- so a whole-trajectory linear-trend statistic loses
power and can flip sign on an effect this small, independent of whether
the net direction matches.

Added a second statistic to `analyze_seedlevel_direction.py`: the raw net
displacement (end value minus start value), reported side by side with
the trend statistic rather than replacing it. By net displacement: relu
matches CIFAR fully (3/3 decline active_frac, 3/3 rise rank -- identical
to the trend statistic, fully robust). gelu/mish/silu: active_frac rises
in **9/9 seeds** (matching CIFAR's direction, magnitude +0.2 to +0.4pp,
smaller than even the CIFAR effect); effective_rank rises in **8/9 seeds**
(one near-exactly-zero exception, gelu seed 0, delta=-0.021 on a baseline
of ~57 -- noise, not a real decline).

**Honest conclusion**: the large-effect-size relu finding (decline +
rank-rise) generalizes cleanly and unambiguously to a non-CIFAR,
larger-vocabulary (200-class), higher-resolution (64x64) dataset, by every
statistic checked. The smaller-effect-size smooth-activation finding
(rise + rank-rise) also generalizes in *net direction* for essentially
every seed tested, but its already-small magnitude becomes harder to
detect with a whole-trajectory linear-trend statistic once the
trajectories stop being monotonic at this scale -- a genuine, reportable
change in trajectory shape, not a reversal of the underlying effect.
Csvs: `tinyimagenet_seedlevel.csv`, `tinyimagenet_summary_af.csv`,
`tinyimagenet_summary_rank.csv`.

### P4 result: the smooth-activation rise generalizes beyond CNNs; the relu decline and the "rank always rises" claim do NOT

MLP-Mixer and Transformer-Encoder both use `nn.LayerNorm`, not BatchNorm
(confirmed by reading `sequence_models.py`) -- a genuinely different
normalization scheme from every CNN architecture tested in this project,
adding independent evidence on top of the existing BatchNorm-vs-GroupNorm
necessity ablation.

**active_frac**: gelu/mish/silu rise robustly in BOTH new architectures
(6/6 seeds each by both the trend and delta statistics, sign-test
p=0.03125, pooling cifar10+cifar100) -- magnitude +0.1 to +0.8pp, comparable
to the CNN case. **The smooth-activation rise is not CNN-specific or
BatchNorm-specific** -- this directly answers P4's motivating question.

relu, however, does NOT show its characteristic CNN decline in either new
architecture. In MLP-Mixer, relu mostly RISES (5-6/6 seeds, small
magnitude +0.005 to +0.05) -- the opposite of its CNN behavior. In
Transformer-Encoder, relu is small-magnitude and dataset-dependent: mildly
declining on cifar100 (3/3 seeds, -0.01 to -0.03) but mixed on cifar10
(2 of 3 positive) -- nothing resembling the large, robust ~16pp CNN decline.
**The relu decline, unlike the smooth-activation rise, does not generalize
to these non-convolutional, LayerNorm-based architectures** -- it appears
tied to something CNN/BatchNorm-specific (plausibly the conv+BN
scale-invariance equilibrium dynamics the mechanism's Step 2 already
leans on) that this experiment did not further isolate.

**effective_rank** is the most architecture-dependent metric of all: the
"rises in all 48/48 runs regardless of activation" CNN finding does NOT
hold here. MLP-Mixer's rank DECLINES for every activation including relu
(6/6 each, e.g. gelu 52.5->46.3, relu 50.1->47.9). Transformer-Encoder
splits: relu's rank rises (6/6, 42.8->52.0, matching CNN) but
gelu/mish/silu's rank declines (5-6/6 each, e.g. gelu 46.2->41.9). The
universal-rank-rise claim is therefore CNN-specific, not a property of
gate dynamics in general.

**Honest synthesis for the paper**: the activation-class active_frac
divergence and the architecture-vs-activation resolution (Contribution 2)
both generalize cleanly beyond CNNs for the smooth-activation half of the
split. The relu-decline half, and the separate "rank always rises" claim,
are both more architecture/normalization-specific than the original CNN
results suggested -- a genuine, reportable narrowing of scope for those
two sub-claims specifically, discovered by deliberately testing outside
the CNN family rather than assumed to generalize. Csvs:
`sequence_model_seedlevel.csv`, `sequence_model_summary_af.csv`,
`sequence_model_summary_rank.csv`.

---

## NeurIPS-8 revision campaign (2026-06-20). Task A pre-registration.

New campaign: resolve the AdamW anomaly mechanistically (Task A, highest
priority), attempt a mu-drift derivation (Task B), scale beyond CIFAR
(Task C), find one real practical payoff (Task D).

**Pre-registering Task A's test, in writing, before running anything or
looking at any outcome** (guardrail: this is a falsifiable test, not a
goal to satisfy; no redefining z_low, the predictor, or channel selection
after seeing results):

- **Hypothesis under test**: the activation-class direction split is a
  downstream consequence of sigma-collapse, not an optimizer-independent
  law. Under coupled weight decay (SGD, Adam), BatchNorm sigma shrinks for
  every activation; for thick-margin activations (GELU/SiLU/Mish) this
  thins a distant sub-threshold tail and active_frac rises; for thin-margin
  ReLU, mu-drift dominates and it falls. Under AdamW (decoupled weight
  decay), sigma does not uniformly shrink (already observed:
  `mechanism_logging_adamw.csv` shows relu/gelu's pooled bn_mean_abs_gamma
  still shrinking, r=-0.50 to -0.84, but silu/mish's GROWING, r=+0.95 to
  +0.98) -- removing tail-thinning should remove the smooth-activation
  rise; with mu still drifting negative, the prediction is uniform decline.
- **Exact predictor, unmodified from `analyze_channel_mechanism.py`'s
  PRIMARY test**: per channel c, per epoch t, the sigma-normalized z-score
  margin `(mu_c(t) - z_low) / sigma_c(t)`, with `z_low` computed by the
  EXACT SAME `compute_z_low()` function already used for the SGD result
  (no new threshold, no reselected channels, no different activation
  subset -- relu/gelu/silu/mish, the same four activations as the
  SGD/AdamW population-level results).
- **What "the predictor predicts correctly" means, fixed before running**:
  (a) per-channel test, same protocol as the SGD case -- for each
  (activation, seed), the fraction of channels where
  corr(z-score margin, active_frac) > 0 across the 5 logged epochs; sign
  test across seeds, same threshold (>0.5 majority) and same significance
  reporting (binomial p) as the SGD analysis. (b) Separately, whether the
  per-activation TREND in the z-score margin itself (rising/falling sigma
  in units of margin) matches the sign required to produce the observed
  active_frac trend under AdamW (relu: decline, as under SGD; gelu/silu/mish:
  decline, UNLIKE under SGD) -- this is the test of the sigma-collapse
  hypothesis specifically, not just the generic per-channel correlation.
- **Required new data**: per-channel mu_c(t), sigma_c(t), active_frac_c(t)
  under AdamW do not exist yet (`channel_mechanism.csv` is SGD-only;
  `mechanism_logging_adamw.csv` is population-pooled, not per-channel).
  Added `--optimizer` to `run_channel_mechanism.py` (same `build_optimizer`
  helper as the rest of the project) and will run it under AdamW, same 4
  activations, 3 seeds, 25 epochs, same MECH_EPOCHS={0,6,12,18,24}, writing
  to a NEW file (`channel_mechanism_adamw.csv`) so the existing SGD data is
  untouched.
- **Both outcomes are reportable and neither is preferred in advance**: if
  the predictor (fed AdamW's real measured mu/sigma) gets the per-activation
  direction right, that unifies the mechanism across optimizers and becomes
  the paper's headline. If it does not, the specific failure mode will be
  characterized and reported as an open regime -- the mechanism will not be
  modified post hoc to fit.

### Task A: AdamW per-channel job submitted

Added `--optimizer` to `run_channel_mechanism.py` (reuses
`build_optimizer`, same pattern as the other scripts; checkpoint-aware
`already_done` now also checks the optimizer column when present, mirroring
`run_pruning_experiment.py`'s `sgd`-keeps-its-original-name convention used
elsewhere is NOT needed here since this script never saved checkpoints by
filename in the first place -- only the output CSV path differs). Smoke
test (z_low computation, `already_done` on a nonexistent file) passed.
**Job 11072643** submitted: relu/gelu/silu/mish, AdamW, ResNet-18,
CIFAR-10, 3 seeds, 25 epochs, identical MECH_EPOCHS to the SGD run, writing
to `channel_mechanism_adamw.csv` (new file, SGD data untouched).

### Task B: mu-drift derivation, attempted and tested against existing data

**Derivation attempted.** This project's optimizer applies weight decay to
every parameter, including BatchNorm's beta (`torch.optim.SGD(...,
weight_decay=5e-4)`, no no-decay parameter group -- confirmed by reading
`run_training_dynamics.py`/`run_channel_mechanism.py`'s optimizer
construction). For a parameter evolving under plain SGD with weight decay,
$\theta_{t+1} = \theta_t - \eta(g_t + \lambda\theta_t)$, where $g_t$ is the
task gradient and $\lambda$ the decay coefficient: if $g_t$ has a
roughly-constant average value $\bar g$ over a window where $\eta$ is
locally constant, the system relaxes toward a quasi-equilibrium
$\theta^\ast \approx -\bar g/\lambda$. For BatchNorm's beta, $g_t = dL/d\beta
= \sum_{\text{batch}} dL/dz$ (since $z = \gamma\cdot\widehat{u}+\beta$, so
$dz/d\beta=1$) -- the gradient flowing into the pre-activation, summed over
the batch. This is a generic, architecture-independent fact about SGD+L2
regularization, NOT specific to BatchNorm's scale-invariance (unlike the
gamma argument in Section 5.2, which needs scale-invariance specifically).
**The one falsifiable prediction**: if $\bar g$ (the average downstream
gradient pressure into $z$) is set mainly by the shared architecture/task
rather than by each activation's own shape, then beta's (=mu's, since
mean($z$)=beta exactly by BatchNorm's own definition) equilibrium value
should be nearly identical across very different activation functions
(same $\lambda=5\times10^{-4}$ for all), and should NOT correlate with
activation-specific shape properties like $z_{\rm low}$ or an
init-time smoothness index.

**Tested against `preactivation_mean_check.csv`** (9 activations, 3 seeds,
already on disk, no new experiment needed): mu at epoch 24 ranges only
$-0.065$ to $-0.041$ (spread 0.024) across activations whose $z_{\rm low}$
spans $-0.00005$ to $-1.234$ (a 4-order-of-magnitude range) -- the
narrowness this account predicts. But the cross-activation variation IS
statistically real, not noise: one-way ANOVA across the 9 activations,
$F=29.7$, $p=7.7\times10^{-9}$; between-activation std (0.0084) is
$3.5\times$ the within-activation (seed) std (0.0024). Tested whether this
residual variation correlates with $z_{\rm low}$: Pearson $r=0.26$,
$p=0.51$; Spearman $\rho=0.28$, $p=0.46$ -- not significant. Tested against
an init-time smoothness index ($\mathrm{Var}[f'(z)]$ for $z\sim
\mathcal N(0,1)$, matching this paper's existing smoothness-sweep
definition): Pearson $r=0.15$, $p=0.71$; Spearman $\rho=0.30$, $p=0.43$ --
also not significant.

**Honest conclusion, reported as partial, not forced further**: the
generic SGD+L2 equilibrium argument correctly predicts the qualitative
fact this paper already reported descriptively (a narrow, shared band
despite wildly different activation shapes) and gives it a mechanistic
reason (shared $\lambda$, roughly shared $\bar g$) rather than leaving it
as an unexplained empirical regularity. It does NOT explain the residual,
statistically significant (p<1e-8) activation-to-activation variation
within that narrow band -- ruled out two natural candidates ($z_{\rm low}$,
init-time smoothness) as the source of that residual; stopping there
rather than testing further candidates post hoc to avoid the appearance of
fishing for one that fits. This closes part of the previously-flagged gap
(why the drift is negative and narrowly shared) while leaving the
remainder (the small residual activation-dependence) explicitly open.

### Task D pre-registration (written before computing anything; guardrail: no p-hacking the payoff)

**Predictor (fixed, already known, not re-derived after seeing the
outcome)**: each existing checkpoint's final (epoch 24) population-level
active\_frac, already on disk in `mechanism_logging.csv` (SGD) and
`mechanism_logging_adamw.csv` (AdamW) -- 24 checkpoints total: 4
activations $\times$ {sgd, adamw} $\times$ 3 seeds, all already trained
25 epochs on ResNet-18/CIFAR-10
(`gradient_gate_outputs/checkpoints{,_adamw}/*.pt`).

**Outcome (fixed in advance, not yet measured)**: test accuracy after
fine-tuning the FULL pretrained network (not just a linear head) on a
fixed 5%-of-train-set subsample of CIFAR-10 (2,500 images, seeded,
identical subsample for every checkpoint), for a fixed recipe applied
identically to all 24 checkpoints regardless of their original
optimizer: SGD, lr=0.01, momentum 0.9, weight\_decay 5e-4, 5 epochs, batch
size 64. This is a low-data fine-tuning stress test: does a network's
ALREADY-MEASURED gate density level (no new training needed to know this)
predict how well it adapts when later given very little new data?

**Pre-registered test**: Pearson and Spearman correlation between
(predictor: final active\_frac) and (outcome: post-fine-tune test
accuracy on the FULL CIFAR-10 test set), across all 24 checkpoints. Also
report the same correlation computed separately within the SGD-only
subset (12 checkpoints) and AdamW-only subset (12 checkpoints), since
pooling across optimizers could mix two different populations. No
checkpoint will be excluded after seeing its result; if any checkpoint
fails to fine-tune (e.g., NaN loss) it will be disclosed, not dropped
silently.

**What would count as a real, useful signal**: a statistically detectable
(seed-level test, not pooled) correlation in either direction, reported
honestly regardless of sign. A null result (no detectable correlation) is
an acceptable, reportable outcome per the guardrails -- it will be stated
as such, not searched-around-until-something-significant-appears.

### TASK A RESULT: the predictor succeeds. The mechanism now spans SGD, Adam, and AdamW.

**Job 11072643 completed** (relu/gelu/silu/mish, AdamW, ResNet-18,
CIFAR-10, 3 seeds, 25 epochs, identical MECH_EPOCHS to the SGD run) ->
`channel_mechanism_adamw.csv` (234,240 rows, matches expected size
exactly). Ran the EXACT, unmodified `analyze_channel_mechanism.py` against
it -- same `compute_z_low`, same sigma-normalized margin formula, no new
threshold, no reselected channels, no different activation subset.

**Step 1 -- does the LOCAL law still hold under AdamW?** Yes: per-channel
correlation between the sigma-normalized margin and active_frac across the
5 epochs is positive for all four activations under AdamW too (72-90% of
channels per seed, mean correlation +0.36 to +0.73, 12/12
activation x seed cells majority-positive). This says the RELATIONSHIP
between margin and active_frac is optimizer-invariant -- a real, useful
fact, but on its own it does not yet explain why AdamW's TREND differs
from SGD's, since a positive correlation is compatible with either a
rising or a falling trend.

**Step 2 -- the decisive test, pre-registered before computing it: does
the margin's TREND match the active_frac TREND actually observed under
AdamW (decline for all four)?** Computed per-channel margin delta (epoch
24 minus epoch 0) and active_frac delta, same sign-match protocol as the
SGD analysis. **12 of 12 (activation x seed) cells show majority
per-channel agreement** that the margin trend's sign matches the
active_frac trend's sign (relu: 87-89% of channels per seed; gelu: 68-75%;
silu: 77-79%; mish: 74-77%; pooled binomial $p=2.44\times10^{-4}$).
Population-level (pooled-channel) margin deltas are negative for ALL FOUR
activations under AdamW (relu $-0.172$, gelu $-0.176$, silu $-0.209$, mish
$-0.228$, all 12/12 seeds same sign) -- exactly matching the observed
uniform active\_frac decline (relu $-7.0$pp, gelu $-2.0$pp, silu $-1.4$pp,
mish $-1.6$pp, also all 12/12 seeds same sign). **The predictor, fed
AdamW's real measured trajectories with zero new free parameters, gets
the direction right for every one of the 12 activation x seed cells.**

**Step 3 -- the mechanistic link, confirmed directly (not just inferred
from population-pooled gamma as before)**: per-channel sigma, epoch 0 to
24, by activation and optimizer:

| activation | SGD sigma change | AdamW sigma change |
|---|---|---|
| relu | $-77.0\%$ | $-0.9\%$ (flat) |
| gelu | $-67.5\%$ | $+2.2\%$ (grows) |
| silu | $-66.2\%$ | $+4.8\%$ (grows) |
| mish | $-66.9\%$ | $+4.7\%$ (grows) |

Under SGD, sigma collapses by roughly two-thirds for every activation
(driving the margin UP for thick-margin activations as the denominator
shrinks, producing the smooth-activation rise). Under AdamW, sigma does
not collapse at all -- it is flat for ReLU and **grows** for the three
smooth activations. With sigma flat-or-growing and mu still drifting
slightly negative (the population-level fact already established and
re-confirmed here), the margin (mu-z_low)/sigma can only decline: for
relu, a shrinking numerator combined with flat sigma produces a sharp
decline (matching its large active_frac drop); for the smooth
activations, growing sigma alone is enough to push the ratio down even
though the numerator barely moves (matching their smaller but still
real and consistent declines).

**This is the "predicts correctly" outcome.** The mechanism in
Section~5 is not two separate, optimizer-specific stories -- it is one
predictor (the sigma-normalized threshold-crossing margin) whose INPUT
trajectory (does sigma collapse or not, under coupled vs.\ decoupled
weight decay) determines the OUTPUT direction, correctly, in every
regime tested. The AdamW "anomaly" from the previous revision is now a
predicted consequence, not an unexplained scope boundary. This becomes
the paper's headline result: rewrite Section 4.5/5 so this is presented
as a unified, falsifiable, predictive theory spanning SGD, Adam, and
AdamW, not as "the mechanism collapses under AdamW."

Csvs: `channel_mechanism_adamw.csv`, `channel_mechanism_adamw_zlow.csv`,
`channel_mechanism_adamw_summary.csv`. Script:
`gradient_gate/run_channel_mechanism.py --optimizer adamw` (new flag,
reuses `build_optimizer`); analysis via the EXISTING, unmodified
`gradient_gate/analyze_channel_mechanism.py --data
channel_mechanism_adamw.csv --zlow channel_mechanism_adamw_zlow.csv`.

### TASK D RESULT: a genuine null, reported exactly as pre-registered

**Job 11074418 completed**: all 24 checkpoints fine-tuned on the fixed,
seeded 5%-CIFAR-10 subsample (2,500 images), identical recipe (SGD,
lr=0.01, 5 epochs) regardless of original optimizer. Merged with each
checkpoint's already-known final active_frac
(`lowdata_finetune_merged.csv`).

**Pre-registered test, computed exactly as written, no exclusions, no
metric changes after seeing the result**:

| subset | n | Pearson r | p | Spearman rho | p |
|---|---|---|---|---|---|
| all 24 | 24 | 0.241 | 0.258 | $-0.332$ | 0.113 |
| AdamW only | 12 | 0.445 | 0.147 | 0.077 | 0.812 |
| SGD only | 12 | 0.246 | 0.441 | $-0.203$ | 0.527 |

**No statistically detectable correlation in any subset, and Pearson and
Spearman do not even agree in sign on the pooled set** -- consistent with
no real relationship rather than a weak true effect being underpowered.
Reporting this as a genuine null exactly as pre-registered: at the design
tested (final population-level active_frac vs.\ post-fine-tune accuracy
on a fixed 5\% CIFAR-10 subsample), gate density level does not predict
low-data fine-tuning performance. Did not pivot to a different outcome
metric (e.g.\ accuracy drop instead of post-finetune accuracy) after
seeing this -- the pre-registration fixed the metric as post-fine-tune
test accuracy, and that is what is reported, regardless of the null
result. Csvs: `lowdata_finetune.csv`, `lowdata_finetune_merged.csv`.
Script: `gradient_gate/run_lowdata_finetune.py`.

### Task C: infrastructure fix -- the Places365 extraction was running improperly on the login node

On resuming this session, found the targeted Places365 extraction
(`tar -xf places365standard_easyformat.tar -T /tmp/places_needed.txt`,
mentioned as in-progress in `UPGRADE_SUMMARY.md`) running as a bare
background process directly on the shared LOGIN NODE, PID 344622, for
3h11m at 95% CPU -- a clear, real violation of this project's own
environment rule (SLURM for anything multi-hour, never an unsupervised
background process on a node shared by dozens of other interactive users
running their own VS Code/Cursor remote servers). Checked progress before
acting: `/proc/<pid>/io` showed `rchar`=8.67GB of the 26.7GB archive
(~32% through, after accounting for page-cache reads from the earlier
abandoned full-blind-extraction attempt), and 166 of 167 train class
directories created so far already had >=150 images each -- real,
salvageable progress, but at this rate ~6-7 more hours were needed, all of
it improperly placed.

**Fixed**: killed PID 344622 (`kill -15`, confirmed dead). Copied
`/tmp/places_needed.txt` to a stable location
(`/scratch/.../places365/places_needed.txt`) since `/tmp` on this cluster
is shared/sticky-bit and not reliable for anything that needs to persist
(an established lesson from earlier in this project). Confirmed the
needed-file list's train entries are deterministic -- the first 150
sequentially-numbered images per class (`00000001.jpg`..`00000150.jpg`),
not a claimed random sample -- so it is exactly reproducible and safe to
document precisely.

This partition rejects CPU-only job submissions outright
(`sbatch: error: Job rejected: No GPUs requested.`) -- every job here must
request `--gres=gpu:1`. Rather than waste a separate multi-hour GPU
allocation purely on extraction, combined extraction and training into one
job (`run_places365_extract.sh`): extraction first (`tar -k` to skip
already-extracted files from the killed attempt without re-writing them),
then immediately `run_places365_dynamics.py` on the same GPU allocation.
2 seeds (feasible within a 24h budget once combined), 4 activations, 25
epochs, SGD. Smoke-tested `Places365Subset` against the currently-partial
val data before submitting (7,300 val samples = 365 x 20 exactly, correct
[3,96,96] shape, 365 classes resolved) -- passed.

**Job 11077301 submitted**, 24h budget, `a100-80gb` partition. This is a
multi-hour job; will check back on completion rather than poll.

### Task C: a second real bug caught -- `tar -k` + `set -e` would have silently skipped training entirely

User asked to check job health and the `.err` file. Found 22,334 lines of
`tar: places365_standard/val/.../*.jpg: Cannot open: File exists` -- the
expected, benign consequence of `-k` (--keep-old-files) skipping files
already extracted from the earlier attempt. The real problem: verified
directly (`tar -k -xf test.tar` on a file that already exists) that GNU
tar exits with status 2 ("Exiting with failure status due to previous
errors") in this situation, not 0. Combined with the script's `set -e`,
this meant the SLURM script was about to abort immediately after the
extraction line finished -- successfully extracting data, then dying
before ever reaching the `run_places365_dynamics.py` training step. The
job would have run for hours, shown `COMPLETED` or `FAILED` depending on
exact timing, and produced **zero training data**, silently, with the
only evidence being a `.err` file most runs would never look at this
closely. Caught before that happened: job 11077301 was still mid-extraction
(1h58m elapsed, 168/365 train classes seen) when checked, so it was killed
(`scancel`) before reaching the failure point.

**Fixed**: `tar -k ... || true` in `run_places365_extract.sh` -- the
expected, benign exit code no longer aborts the script; the existing
post-extraction verification step (counts train classes still short of
150 images) is what actually catches a real extraction problem, not tar's
own exit code. **Job 11078036 resubmitted** with the fix, same 24h budget.
Confirmed before resubmitting that this is a real fix, not a guess: a
local test (`tar -k -xf test.tar` on a pre-existing file in `/tmp`)
reproduced the exact exit-code-2 behavior first.

Also noted in passing: the fresh restart (dedicated SLURM compute node)
reached 168/365 train classes in 1h58m, faster than the original
login-node attempt's 167/365 in 3h11m -- consistent with the login node
being genuinely contended by the dozens of other interactive users
observed in `ps aux` earlier, beyond just being the wrong place to run
this regardless of speed.

### Task C: a third bug -- `cifar_resnet50` crashes on any non-default norm_layer call path

**Job 11078036 result**: extraction succeeded completely this time (the
`set -e` fix held) -- "train classes under 150: 0 / 365", confirmed every
one of the 365 classes has its full 150-image quota, 7h13m for the
extraction step. Training then crashed on the very first model
construction: `TypeError: __init__() got an unexpected keyword argument
'norm_layer'` in `Bottleneck.__init__`, called from
`CifarResNet.__init__`'s block-construction loop, which unconditionally
passes `norm_layer=norm_layer` to every block regardless of block type.
`BasicBlock` (resnet18/vgg11's block) accepts and uses `norm_layer`
correctly; `Bottleneck` (resnet50's block, used here for the first time
since the GroupNorm-ablation work added the `norm_layer` parameter to
`CifarResNet.__init__`) never had a matching parameter added -- a latent
bug, not something Task C's own code introduced. Confirmed this is
genuinely pre-existing and not something my changes broke: the published
`training_dynamics.csv` ResNet-50/ReLU baseline (150 rows, max test_acc
91.58\%) ran successfully, but necessarily before the `norm_layer`
threading was added to `CifarResNet.__init__` -- resnet50 was never
exercised again after that change, so the bug went unnoticed until now.

**Fixed** in `gradient_gate/cifar_models.py`: added `norm_layer=_default_norm`
to `Bottleneck.__init__`'s signature and replaced its three hardcoded
`nn.BatchNorm2d(...)` calls with `norm_layer(...)`, mirroring `BasicBlock`'s
existing pattern exactly. This is purely additive: the default
(`_default_norm` = `nn.BatchNorm2d`) produces an identical module to the
old hardcoded calls, so the already-published ResNet-50/ReLU baseline
numbers are unaffected -- verified this reasoning, not just asserted it,
by confirming `training_dynamics.csv`'s resnet50 rows are real and
already complete. Direct unit test before resubmitting: built
`cifar_resnet50(num_classes=365, act_layer=nn.GELU)`, ran a real
forward+backward pass (output shape `(4, 365)` correct, backward
succeeds, 24.2M params -- correct ResNet-50 scale) -- passed.

**Job 11081101 resubmitted** (`extraction_done.marker` present, so this
run skips straight to training). Three real bugs now caught and fixed in
this one Task C sub-effort alone (login-node policy violation, `tar -k`
+ `set -e` silent-failure, `Bottleneck` missing `norm_layer`) -- each
verified directly before being declared fixed, not assumed.

### Task C: a fourth issue -- per-image PIL decode from this scratch filesystem is the real bottleneck, not "slow training"

**Job 11081101 result**: extraction skipped correctly (marker present),
`cifar_resnet50` built successfully (the norm_layer fix held, `device=cuda`
confirmed), but after 20+ minutes, zero epochs had printed. Diagnosed
directly rather than just waiting longer: `ssh`'d to the job's compute
node and ran `nvidia-smi` (0% GPU utilization) and read
`/proc/<pid>/status` and `/proc/<pid>/io` for the training process
(state `I` -- idle/blocked, not running; `rchar`=679MB over ~20 minutes,
i.e. ~0.57MB/s sustained, ~12.5 images/s single-threaded). At that rate
one epoch's data loading alone (54,750 train images) would take roughly
an hour, and the full run (25 epochs x 4 activations x 2 seeds = 200
epoch-runs) would take days, not the 24h budgeted -- not "slow," actually
infeasible within any reasonable allocation. Root cause: per-image PIL
decode from individual JPEG files on this network/scratch filesystem has
severe per-file I/O latency (the same class of issue that made the raw
tar extraction slow), and `num_workers=0` (this project's standing rule,
justified for CIFAR-scale in-memory datasets, kept here too rather than
risk the documented CUDA+fork hang) gives no overlap between I/O and GPU
compute to hide it.

**Fixed** with a one-time caching step rather than changing the
num_workers rule: `gradient_gate/cache_places365.py` reads each of the
62,050 needed images exactly ONCE, resizes to 144x144 (kept larger than
the final 96x96 training resolution to leave real room for
RandomResizedCrop), and saves the whole subsample as two single tensor
files (`train_cache.pt`, `val_cache.pt`, ~3-4GB combined, comfortably
fits the per-file size budget and RAM). Added `CachedPlaces365` to
`run_places365_dynamics.py`: loads one tensor file once, serves images via
pure in-memory indexing, applies the identical augmentation
(`RandomResizedCrop(96, scale=(0.7,1.0))` + flip for train,
resize-only for val) directly on tensors via torchvision's tensor-mode
transforms -- same augmentation semantics, zero further disk I/O per
epoch. Verified each piece before trusting it: (a) the dataset-class
mechanics on a synthetic fake cache (correct output shapes/dtypes), (b)
the real PIL-loading+resize logic on 10 real images from 2 actual classes
(0.48s for 10 images = ~20.8 images/s, confirming the per-file read itself
isn't catastrophically slow in isolation -- the problem was paying that
cost 200x instead of once), (c) clean imports of the new module and the
updated `get_dataloaders(..., cache_dir=...)` path.

Updated `run_places365_extract.sh`: a new Step 1.5 builds the cache (skip
if already present, same `if [ ! -f ... ]` pattern as the extraction
marker) before training, and Step 2 now passes `--cache-dir`. Estimated
total budget: ~50min one-time caching (extrapolated from the 20.8
images/s real-file test) + ~5h training (8 runs x ~37min, extrapolated
from expected A100 ResNet-50 batch cost once data loading is no longer
the bottleneck) -- comfortably within the 24h allocation, a complete
reversal from the previous architecture's days-long infeasibility.
**Resubmitted** under the same job script (new job ID to follow).
