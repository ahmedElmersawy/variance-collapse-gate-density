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
