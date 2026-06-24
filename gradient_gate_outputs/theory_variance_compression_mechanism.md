# Why does gate density diverge by activation class? A derivation, not just a description

**Revision note:** an earlier draft of this document argued smooth
activations rise because their gate has an interior maximum that the
pre-activation distribution concentrates toward as variance shrinks.
That argument is wrong for Softplus, whose derivative is *exactly*
$\mathrm{sigmoid}(\beta z)$ — strictly monotonic, with no interior maximum
at all. Since the Softplus-$\beta$ family is the cleanest test available
of this whole derivation, getting it wrong there means the argument was
wrong, not just incomplete. This revision replaces it with a single,
unified mechanism that does not depend on whether the gate has a hump,
verified numerically below before any new experiment was run. Each claim
is labeled **(Exact)**, **(Established)**, **(Proposition)**, or
**(Conjecture, testable)** as before.

## Step 1 — the post-BatchNorm pre-activation distribution **(Exact)**

Unchanged from the first draft. For a channel with pre-BN output $u$,

$$z = \gamma \cdot \frac{u - \mathrm{mean}(u)}{\sqrt{\mathrm{var}(u)+\epsilon}} + \beta
\quad\Rightarrow\quad \mathrm{mean}(z)\approx\beta,\ \ \mathrm{var}(z)\approx\gamma^2$$

to first order, by BatchNorm's own definition.

## Step 2 — why $\gamma$ shrinks **(Established, applied here)**

Unchanged. A conv layer feeding into BatchNorm is scale-invariant; under
SGD with weight decay, prior work (van Laarhoven 2017; Hoffer et al. 2018)
shows such layers' effective scale is pulled toward an equilibrium that
shrinks as the learning rate decays — consistent with, and a plausible
explanation for, $\gamma$'s empirically observed monotonic shrinkage in
every one of 12 independent trajectories (sign-test $p=2.44\times10^{-4}$).

## Step 3 — the corrected, unified mechanism **(Proposition)**

Define $z_{\mathrm{low}}(\theta)$, for a gate $g=|f'|$ and threshold
$\theta$, as the point where $g$ crosses $\theta$ on its left (negative)
tail: $z_{\mathrm{low}} = \inf\{z : g(z) > \theta\}$. This is a property
of the activation function and the chosen threshold *alone* — fixed,
computable in advance, and **independent of training**. We computed it
exactly via autograd for $\theta=0.10$:

| activation | $z_{\mathrm{low}}$ | argmax of $g$ | empirical trend (this project) |
|---|---|---|---|
| relu, all LeakyReLU slopes | 0.000 | 0.000 | decline |
| softplus β=50 | −0.044 | 0.333 | decline |
| softplus β=20 | −0.110 | 0.832 | **rise** |
| softplus β=10 | −0.219 | 1.665 | rise |
| softplus β=5 | −0.439 | 3.327 | rise |
| gelu | −0.554 | 1.414 | rise |
| mish | −0.891 | 1.490 | rise |
| silu | −0.912 | 2.400 | rise |

$z_{\mathrm{low}}$ moves monotonically more negative exactly along the
hypothesized smoothness order, and **the empirical sign transition (decline
→ rise, between Softplus β=50 and β=20) lines up with $z_{\mathrm{low}}$
moving from −0.044 to −0.110** — both close to zero but on opposite sides
of whatever the typical pre-activation drift turns out to be. This does
not require an interior maximum anywhere; it only requires that $g$ rises
from 0 at $-\infty$, and that *how far left* it has to rise before
crossing $\theta$ depends on smoothness.

**The mechanism, stated plainly:** if $z \sim \mathcal{N}(\mu,\sigma^2)$
with $\sigma\to 0$ (Steps 1–2) and $\mu$ drifts by a *similar* small
amount across activations (a claim about training behavior, tested below,
not assumed), then $\mathrm{active\_frac}(\theta) = P(g(z)>\theta) \to 0$
or $1$ depending only on whether $\mu$ ends up below or above
$z_{\mathrm{low}}(\theta)$ — variance shrinkage amplifies whichever side
$\mu$ is already on into a degenerate limit. Activations whose
$z_{\mathrm{low}}$ sits close to 0 (ReLU, LeakyReLU, high-$\beta$ Softplus)
have very little margin: a small negative drift in $\mu$ (which we already
have indirect evidence for, since $|\beta|$ grows substantially during
training in every activation tested) is enough to put them on the
"decline" side. Smoother activations push $z_{\mathrm{low}}$ further
negative, giving the *same* small drift in $\mu$ more room to stay on the
"rise" side.

## Part 2 — the conjecture, tested

We logged signed pre-activation mean $\mu$ and std $\sigma$ (forward-hook
only, pooled per-channel, no backward pass needed) at epochs
{0,6,12,18,24} for 9 representative activations spanning the empirical
sign transition (ReLU, LeakyReLU(0.01), Softplus at $\beta\in\{50,20,10,5\}$,
GELU, SiLU, Mish), 3 seeds, 25 epochs — `preactivation_mean_check.csv`.

**The conjecture holds.** At initialization, $\mu$ is small and positive
for every activation (0.014–0.033, no meaningful difference between
them). By epoch 6, $\mu$ has flipped negative for **every single
activation tested**, and stays in a narrow, shared band through epoch 24:
$-0.041$ (SiLU) to $-0.066$ (Mish) — a roughly *common* drift, not nine
different activation-specific trajectories. This is exactly what the
mechanism requires: the explanatory burden falls on $z_{\mathrm{low}}$
(fixed, activation-specific, computed independently of training), not on
$\mu$ behaving differently per activation.

**The mechanism's predictions, checked against the smoothness sweep's
actual results:**

| activation | $\mu$(epoch 24) | $z_{\mathrm{low}}$ | margin ($\mu-z_{\mathrm{low}}$) | predicted | observed | match | observed $\rho$ |
|---|---|---|---|---|---|---|---|
| relu | −0.0458 | 0.000 | −0.0458 | decline | decline | ✓ | −0.976 |
| leaky_relu_0.01 | −0.0478 | 0.000 | −0.0478 | decline | decline | ✓ | −0.986 |
| softplus β=50 | −0.0476 | −0.044 | **−0.0036** | decline | decline | ✓ | **−0.588** |
| softplus β=20 | −0.0568 | −0.110 | +0.0532 | rise | rise | ✓ | +0.731 |
| softplus β=10 | −0.0635 | −0.219 | +0.1555 | rise | rise | ✓ | +0.999 |
| softplus β=5 | −0.0586 | −0.439 | +0.3804 | rise | rise | ✓ | +0.998 |
| gelu | −0.0508 | −0.554 | +0.5032 | rise | rise | ✓ | +0.996 |
| silu | −0.0413 | −0.912 | +0.8707 | rise | rise | ✓ | +0.990 |
| mish | −0.0655 | −0.891 | +0.8255 | rise | rise | ✓ | +0.971 |

**9/9 predictions match**, derived entirely from (a) a $z_{\mathrm{low}}$
value computed once, analytically, with no training involved, and (b) a
$\mu$-drift trajectory whose near-universality across very different
activation functions was verified rather than assumed. The theory also
predicts *which* case should be fragile: Softplus($\beta=50$) sits at a
margin of $-0.0036$ — within noise of the critical boundary — and is
exactly the one condition in the entire smoothness sweep with a
conspicuously weak, high-variance correlation ($\rho=-0.588$, the
weakest-magnitude and highest cross-seed variance of all nine conditions,
versus $|\rho|>0.96$ for every other activation). The mechanism does not
just get the sign right everywhere; it correctly identifies in advance
which sign it should be least confident about.

## Honest residual uncertainty

This derivation explains the *direction* of the gate-density trend via a
shared, empirically-verified $\mu$-drift interacting with an
activation-specific, exactly-computable threshold location. It does
**not** explain *why* $\mu$ drifts negative in the first place (a
question about the loss landscape's interaction with BatchNorm's $\beta$
parameter under weight decay, not addressed here), nor does it predict the
*magnitude* of the trend, only its sign — the magnitude is governed by how
fast $\sigma$ shrinks relative to where $\mu$ sits, which we have not
derived a closed form for. We regard the sign-prediction result as solid;
we do not claim a complete theory of the magnitude.
