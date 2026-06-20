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
