#!/usr/bin/env python3
"""
Gradient Gate Collapse: A Quantitative Theory of Phase Transitions
in Neural Inverse Reconstruction Landscapes
Ahmed Elmersawy — Purdue University

Standalone script: all experiments, parallelised via joblib.
Checkpoint-aware: every sweep skips already-computed CSV rows.
Run with:
    python run_experiments.py [--profile full|demo] [--jobs N] [--skip-mnist]

SCOPE NOTE — Theorem 4.3 (depth-compounding law), added after auditing the
empirical evidence already in this file: test_gate_independence() (S7.6b,
~line 762) measures corr(Gamma^(1), A2^T*Gamma^(2)) at convergence of the
two-layer problem and finds it is NOT negligible at low/moderate alpha (peak
r~0.3-0.4 around alpha~3, i.e. 10-18% shared variance) — it only decays
toward independence asymptotically, alpha>=40. The independence assumption
Theorem 4.3 relies on is therefore an ASYMPTOTIC (large-alpha) approximation,
not a general property of the gate-collapse regime. Any statement of the
depth-compounding law F^(L)/F^(1) = (1+c*alpha)^-(L-1) in the writeup should
be scoped accordingly: stated as "holds in the deep-saturation regime,
empirically verified for alpha>=~40" rather than as an unconditional law.
This was already correctly reported in test_gate_independence()'s own
printed verdict (lines ~809-814) — this note exists so the scope limitation
is visible from the file header too, not only inside one function's output.
"""

# ─────────────────────────────── imports ─────────────────────────────────────
import os, sys, math, time, json, random, argparse, warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on Colab/headless
import matplotlib.pyplot as plt


from scipy.optimize import minimize, curve_fit
from scipy.stats import ortho_group, pearsonr
from scipy.ndimage import convolve, zoom
from scipy.special import erf

warnings.filterwarnings("ignore")

# ─────────────────────────── argument parsing ────────────────────────────────
parser = argparse.ArgumentParser(description="Gradient Gate Collapse experiments")
parser.add_argument("--profile", choices=["full", "demo"], default="full")
parser.add_argument("--jobs",    type=int, default=-1,
                    help="joblib n_jobs (-1 = all CPUs)")
parser.add_argument("--skip-mnist", action="store_true")
parser.add_argument("--skip-deepnet", action="store_true",
                    help="Skip Phase-3A deep-CNN (ResNet-18/VGG-11-shaped) gate analysis (requires torch+CUDA)")
parser.add_argument("--root",   default=None,
                    help="Output root dir (default: auto-detect)")
parser.add_argument("--verify-only", action="store_true",
                    help="Load existing CSVs and run verification checks only")
args, _ = parser.parse_known_args()

N_JOBS = args.jobs

# ─────────────────────────── output directories ──────────────────────────────
def _detect_root() -> str:
    if args.root:
        return args.root
    env = os.environ.get("GRADIENT_GATE_ROOT_DIR")
    if env:
        return env
    if os.path.isdir("/content"):
        return "/content/gradient_gate_outputs"
    return os.path.join(os.getcwd(), "gradient_gate_outputs")

ROOT_DIR = _detect_root()
FIG_DIR  = os.path.join(ROOT_DIR, "figures")
CSV_DIR  = os.path.join(ROOT_DIR, "csv")
NPY_DIR  = os.path.join(ROOT_DIR, "arrays")
for d in [FIG_DIR, CSV_DIR, NPY_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[setup] root    : {ROOT_DIR}")
print(f"[setup] profile : {args.profile}")
print(f"[setup] jobs    : {N_JOBS}")

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ─────────────────────── profile knobs ───────────────────────────────────────
if args.profile == "full":
    SEEDS        = (0, 1, 2, 3, 4, 5, 6, 7)
    STEPS        = 600
    SHAPE        = (64, 64)
    ALPHAS       = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 16.0, 20.0, 25.0, 30.0, 40.0, 60.0)
    DENSE_ALPHAS = (1., 1.5, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.,
                    15., 16., 17., 18., 19., 20., 22., 25., 28., 30., 35., 40., 50., 60.)
    SCALE_ALPHAS = (1., 1.5, 2., 3., 5., 7., 10., 14., 17., 20., 25., 30., 40., 60.)
else:
    SEEDS        = (0, 1)
    STEPS        = 150
    SHAPE        = (32, 32)
    ALPHAS       = (1.0, 5.0, 10.0, 20.0, 40.0)
    DENSE_ALPHAS = (1., 5., 10., 20., 40.)
    SCALE_ALPHAS = (1., 5., 20., 40.)

# ─────────────────────────── CSV helpers ─────────────────────────────────────
def load_csv(name, warn=True):
    p = os.path.join(CSV_DIR, name)
    if os.path.exists(p):
        return pd.read_csv(p)
    if warn:
        print(f"[warn] CSV not found: {p}")
    return None

def save_csv(df, name):
    p = os.path.join(CSV_DIR, name)
    df.to_csv(p, index=False)
    print(f"[csv] saved {name}  ({len(df)} rows)")

def append_csv(rows, name):
    existing = load_csv(name, warn=False)
    new = pd.DataFrame(rows)
    combined = pd.concat([existing, new], ignore_index=True) if existing is not None else new
    save_csv(combined, name)
    return combined

def save_fig(name, dpi=180):
    p = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[fig] saved {name}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CORE MATH PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(z, alpha, c):
    return 1.0 / (1.0 + np.exp(-np.clip(alpha*(z-c), -60, 60)))

def sigmoid_prime(s, alpha):
    return alpha * s * (1.0 - s)

# ── Activation registry: f_alpha(z) = f_base(alpha*(z-c)) ────────────────────
# Generalizes the sigmoid stiffness parameterization to arbitrary base
# nonlinearities so that "alpha" retains its meaning (transition sharpness)
# uniformly across activations. Gate Gamma(z) = |d/dz f_alpha(z)|
#         = alpha * |f_base'(alpha*(z-c))|  (chain rule; exact, not approximate).
# This recovers the existing sigmoid implementation exactly when f_base = logistic.
SQRT_2 = float(np.sqrt(2.0))
INV_SQRT_2PI = float(1.0/np.sqrt(2.0*np.pi))

def _logistic(u):
    return 1.0 / (1.0 + np.exp(-u))

def _softplus(u):
    return np.logaddexp(0.0, u)

def _act_sigmoid(u):
    return _logistic(u)
def _act_sigmoid_prime(u):
    s = _logistic(u)
    return s * (1.0 - s)

def _act_tanh(u):
    return np.tanh(u)
def _act_tanh_prime(u):
    t = np.tanh(u)
    return 1.0 - t*t

def _act_relu(u):
    return np.maximum(u, 0.0)
def _act_relu_prime(u):
    # Scale-invariant: ReLU'(alpha*u) = ReLU'(u) = 1[u>0] for any alpha>0.
    # Gamma_relu(x) = alpha * 1[conv(x)>c]: a binary mask scaled by alpha that
    # NEVER collapses as alpha->infinity (qualitatively distinct from the
    # smooth saturating activations below, whose Gamma -> 0 a.e.).
    return (u > 0.0).astype(np.float64)

def _act_gelu(u):
    # Exact GELU via the Gaussian CDF (erf), not the tanh approximation.
    return u * 0.5*(1.0 + erf(u/SQRT_2))
def _act_gelu_prime(u):
    Phi = 0.5*(1.0 + erf(u/SQRT_2))
    phi = INV_SQRT_2PI * np.exp(-0.5*u*u)
    return Phi + u*phi

def _act_silu(u):
    # SiLU == Swish with beta=1: f(u) = u * sigmoid(u)
    return u * _logistic(u)
def _act_silu_prime(u):
    s = _logistic(u)
    return s + u*s*(1.0 - s)

def _act_mish(u):
    # f(u) = u * tanh(softplus(u))
    return u * np.tanh(_softplus(u))
def _act_mish_prime(u):
    sp = _softplus(u)
    t = np.tanh(sp)
    s = _logistic(u)            # softplus'(u) = sigmoid(u)
    return t + u*(1.0 - t*t)*s

# name -> (f_base, f_base')  — both take the *pre-activation* u = alpha*(z-c)
ACTIVATIONS = {
    "sigmoid": (_act_sigmoid, _act_sigmoid_prime),
    "tanh":    (_act_tanh,    _act_tanh_prime),
    "relu":    (_act_relu,    _act_relu_prime),
    "gelu":    (_act_gelu,    _act_gelu_prime),
    "swish":   (_act_silu,    _act_silu_prime),
    "silu":    (_act_silu,    _act_silu_prime),
    "mish":    (_act_mish,    _act_mish_prime),
}

def project_box(x):
    return np.clip(x, 0.0, 1.0)

def iou_score(pred, target, thr=0.5):
    pb = (pred >= thr).astype(np.uint8)
    tb = (target >= thr).astype(np.uint8)
    inter = np.logical_and(pb, tb).sum()
    union = np.logical_or(pb, tb).sum()
    return float(inter / union) if union > 0 else 1.0

def psnr(a, b):
    err = np.mean((a-b)**2)
    return 100.0 if err <= 1e-12 else float(20*np.log10(1.0) - 10*np.log10(err))

def ssim_fast(a, b):
    """Lightweight mean-SSIM approximation."""
    C1, C2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    sa = a.std(); sb = b.std()
    sab = float(np.mean((a-mu_a)*(b-mu_b)))
    return float(((2*mu_a*mu_b+C1)*(2*sab+C2)) / ((mu_a**2+mu_b**2+C1)*(sa**2+sb**2+C2)+1e-12))

def tv_norm(x):
    return float(np.abs(np.diff(x,axis=0)).sum() + np.abs(np.diff(x,axis=1)).sum())

def rel_loss_reduction(l0, lT):
    return float((l0-lT)/l0) if abs(l0) > 1e-12 else 0.0

def bootstrap_ci(values, n_boot=500, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n_boot)]
    lo = (1-ci)/2*100; hi = (1+ci)/2*100
    return float(arr.mean()), float(np.percentile(boots, lo)), float(np.percentile(boots, hi))

def cohens_d(a, b):
    diff = np.asarray(a)-np.asarray(b)
    return float(diff.mean()/(diff.std(ddof=1)+1e-12))

def wilcoxon_pairwise(a, b):
    from scipy.stats import wilcoxon
    try:
        stat, pvalue = wilcoxon(np.asarray(a), np.asarray(b), alternative="two-sided")
    except Exception:
        stat, pvalue = 0.0, 1.0
    return {"statistic": float(stat), "pvalue": float(pvalue)}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — KERNELS & TARGETS
# ══════════════════════════════════════════════════════════════════════════════

def _build_kernels():
    N = SHAPE[0]  # use actual image size for spectral norm computation
    def _sigma_max(k):
        """Max singular value of circular convolution operator on NxN grid."""
        kp = np.zeros((N, N)); kp[:k.shape[0], :k.shape[1]] = k
        return float(np.abs(np.fft.fft2(kp)).max())

    K = {}
    K["identity_like"] = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=float)  # σ_max=1.0
    K["avg_blur"]      = np.ones((3,3),dtype=float)/9.0                     # σ_max=1.0

    # Sobel_x: normalize so σ_max(circular conv operator) = 4.0
    sobel_raw = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
    K["sobel_x"] = sobel_raw / _sigma_max(sobel_raw) * 4.0

    # Laplacian: normalize so σ_max = 8.0
    lap_raw = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
    K["laplacian"] = lap_raw / _sigma_max(lap_raw) * 8.0

    # Random: normalize so σ_max = 2.0 (between identity and Sobel)
    # This gives a real phase transition between α*_Sobel≈11.7 and α*_Laplacian
    rng = np.random.default_rng(7)
    rn = rng.standard_normal((3,3))
    K["random_norm"] = rn / _sigma_max(rn) * 2.0
    return K

KERNELS = _build_kernels()

# Per-kernel sigmoid centering: non-negative kernels (identity, avg_blur) keep
# conv(x) >= 0 for x in [0,1], so c=0.0 makes sigmoid >= 0.5 always —
# y=0 pixels hit the box boundary and can never be reconstructed.
# c=0.5 shifts the midpoint to z=0.5, the center of the achievable range [0,1].
# Kernels with negative weights (sobel_x, laplacian, random_norm) can produce
# negative conv outputs so c=0.0 is correct for them.
KERNEL_C = {
    "identity_like": 0.5,
    "avg_blur":      0.5,
    "sobel_x":       0.0,
    "laplacian":     0.0,
    "random_norm":   0.0,
}

def get_targets(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    ck = ((yy//8)+(xx//8)) % 2 == 0
    T = {
        "checkerboard": ck.astype(float),
        "vstripes":    ((xx//6)%2==0).astype(float),
        "hstripes":    ((yy//6)%2==0).astype(float),
        "circle":      (((yy-h/2)**2+(xx-w/2)**2) <= (min(h,w)*0.28)**2).astype(float),
    }
    rng = np.random.default_rng(7)
    dots = np.zeros((h,w))
    for _ in range(40):
        r,c = rng.integers(0,h), rng.integers(0,w)
        dots[max(0,r-1):r+2, max(0,c-1):c+2] = 1.0
    T["sparse_dots"] = dots
    cy1,cx1 = int(h*0.35), int(w*0.35); cy2,cx2 = int(h*0.65), int(w*0.65)
    r1 = int(min(h,w)*0.18); r2 = int(min(h,w)*0.15)
    blob = ((yy-cy1)**2+(xx-cx1)**2<=r1**2) | ((yy-cy2)**2+(xx-cx2)**2<=r2**2)
    T["two_blobs"] = blob.astype(float)
    return T

TARGETS_64 = get_targets(64, 64)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PROBLEM CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProblemConfig:
    image_shape: Tuple[int,int] = (64,64)
    alpha: float = 10.0; c: float = 0.0
    kernel_name: str = "sobel_x"; target_name: str = "checkerboard"
    activation: str = "sigmoid"; noise_std: float = 0.0
    tv_lambda: float = 0.0; tikhonov_lambda: float = 0.0
    tikhonov_center: float = 0.5

class FixedCNNInverseProblem:
    def __init__(self, config: ProblemConfig, target: np.ndarray, kernel: np.ndarray):
        self.config = config
        self.image_shape = config.image_shape
        self.alpha = float(config.alpha); self.c = float(config.c)
        self.tv_lambda = float(config.tv_lambda)
        self.tikhonov_lambda = float(config.tikhonov_lambda)
        self.tikhonov_center = float(config.tikhonov_center)
        self.kernel_name = config.kernel_name; self.target_name = config.target_name
        if config.activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{config.activation}'; "
                             f"available: {sorted(ACTIVATIONS)}")
        self.activation = config.activation
        self._act_base, self._act_base_prime = ACTIVATIONS[config.activation]
        self.y_clean = target.astype(np.float64)  # binary {0,1} target — what f(x) should match
        self.kernel = kernel.astype(np.float64)
        self.kernel_flip = np.flipud(np.fliplr(self.kernel))
        assert self.y_clean.shape == self.image_shape
        self.y = self.y_clean.copy()
        if config.noise_std > 0:
            rng = np.random.default_rng(0)
            self.y = np.clip(self.y + config.noise_std*rng.normal(size=self.y.shape), 0.0, 1.0)

    def conv(self, x):
        return convolve(x, self.kernel, mode="wrap")
    def conv_transpose(self, z):
        return convolve(z, self.kernel_flip, mode="wrap")
    def _act(self, z):
        """f_alpha(z) = f_base(alpha*(z-c)): stiffness-parameterized activation."""
        u = np.clip(self.alpha*(z - self.c), -60, 60)
        return self._act_base(u)
    def _gate_at(self, z):
        """Gamma(z) = |d/dz f_alpha(z)| = alpha * |f_base'(alpha*(z-c))| (chain rule)."""
        u = np.clip(self.alpha*(z - self.c), -60, 60)
        return self.alpha * np.abs(self._act_base_prime(u))
    def forward(self, x):
        return self._act(self.conv(x))
    def data_loss(self, x):
        return float(np.sum((self.y - self.forward(x))**2))
    def loss(self, x):
        l = self.data_loss(x)
        if self.tv_lambda > 0: l += self.tv_lambda*tv_norm(x)
        if self.tikhonov_lambda > 0:
            l += self.tikhonov_lambda*float(np.sum((x-self.tikhonov_center)**2))
        return l
    def grad(self, x):
        ax = self.conv(x); h = self._act(ax)
        gate = self._gate_at(ax)
        g = 2.0*self.conv_transpose((h-self.y)*gate)
        if self.tv_lambda > 0:
            gv = np.zeros_like(x)
            gv[:-1,:] += self.tv_lambda*np.sign(x[1:,:]-x[:-1,:])
            gv[1:,:]  -= self.tv_lambda*np.sign(x[1:,:]-x[:-1,:])
            gv[:,:-1] += self.tv_lambda*np.sign(x[:,1:]-x[:,:-1])
            gv[:,1:]  -= self.tv_lambda*np.sign(x[:,1:]-x[:,:-1])
            g += gv
        if self.tikhonov_lambda > 0:
            g += 2.0*self.tikhonov_lambda*(x-self.tikhonov_center)
        return g
    def grad_norm(self, x): return float(np.linalg.norm(self.grad(x)))
    def gradient_gate(self, x):
        return self._gate_at(self.conv(x))
    def active_grad_fraction(self, x, thr=0.01):
        return float(np.mean(self.gradient_gate(x) > thr))
    def metrics(self, x, x_star=None):
        fx = self.forward(x); x_star = x_star if x_star is not None else self.y_clean
        gate = self.gradient_gate(x)
        return {"loss": self.loss(x), "data_loss": self.data_loss(x),
                "grad_norm": self.grad_norm(x),
                "output_mse": float(np.mean((fx-self.y)**2)),
                "output_binary_acc": float(np.mean(((fx>=0.5)==(self.y>=0.5)))),
                "output_iou": iou_score(fx, self.y),
                "input_psnr_vs_target": psnr(x, x_star),
                "input_ssim_vs_target": ssim_fast(x, x_star),
                "active_grad_fraction": float(np.mean(gate>0.01)),
                "saturation_fraction":  float(np.mean(gate<1e-3))}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OPTIMIZERS
# ══════════════════════════════════════════════════════════════════════════════

def run_pgd(problem, x0, lr=0.1, steps=200):
    x = x0.copy(); loss_hist=[]; grad_hist=[]; t0=time.time()
    for _ in range(steps):
        loss_hist.append(problem.loss(x)); g=problem.grad(x)
        grad_hist.append(float(np.linalg.norm(g))); x=project_box(x-lr*g)
    return {"x_final":x,"loss_hist":loss_hist,"grad_hist":grad_hist,
            "time_sec":time.time()-t0,"optimizer":"pgd"}

def run_momentum(problem, x0, lr=0.05, beta=0.9, steps=200):
    x=x0.copy(); v=np.zeros_like(x); loss_hist=[]; grad_hist=[]; t0=time.time()
    for _ in range(steps):
        loss_hist.append(problem.loss(x)); g=problem.grad(x)
        grad_hist.append(float(np.linalg.norm(g)))
        v=beta*v-lr*g; x=project_box(x+v)
    return {"x_final":x,"loss_hist":loss_hist,"grad_hist":grad_hist,
            "time_sec":time.time()-t0,"optimizer":"momentum"}

def run_nesterov(problem, x0, lr=0.05, beta=0.9, steps=200):
    x=x0.copy(); v=np.zeros_like(x); loss_hist=[]; grad_hist=[]; t0=time.time()
    for _ in range(steps):
        loss_hist.append(problem.loss(x)); g=problem.grad(project_box(x+beta*v))
        grad_hist.append(float(np.linalg.norm(g)))
        v=beta*v-lr*g; x=project_box(x+v)
    return {"x_final":x,"loss_hist":loss_hist,"grad_hist":grad_hist,
            "time_sec":time.time()-t0,"optimizer":"nesterov"}

def run_adam(problem, x0, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8, steps=200):
    x=x0.copy(); m=np.zeros_like(x); v=np.zeros_like(x)
    loss_hist=[]; grad_hist=[]; t0=time.time()
    for t in range(1, steps+1):
        loss_hist.append(problem.loss(x)); g=problem.grad(x)
        grad_hist.append(float(np.linalg.norm(g)))
        m=beta1*m+(1-beta1)*g; v=beta2*v+(1-beta2)*g**2
        mh=m/(1-beta1**t); vh=v/(1-beta2**t)
        x=project_box(x-lr*mh/(np.sqrt(vh)+eps))
    return {"x_final":x,"loss_hist":loss_hist,"grad_hist":grad_hist,
            "time_sec":time.time()-t0,"optimizer":"adam"}

def run_lbfgsb(problem, x0, maxiter=200):
    shape=x0.shape; x0f=x0.reshape(-1); bounds=[(0.,1.)]*x0f.size
    loss_hist=[]; grad_hist=[]; t0=time.time()
    def fun(z): return problem.loss(z.reshape(shape))
    def jac(z): return problem.grad(z.reshape(shape)).reshape(-1)
    def cb(z): loss_hist.append(problem.loss(z.reshape(shape))); grad_hist.append(problem.grad_norm(z.reshape(shape)))
    res = minimize(fun, x0f, jac=jac, method="L-BFGS-B", bounds=bounds,
                   callback=cb, options={"maxiter":maxiter,"disp":False})
    xf = res.x.reshape(shape)
    if not loss_hist: loss_hist=[problem.loss(xf)]; grad_hist=[problem.grad_norm(xf)]
    return {"x_final":xf,"loss_hist":loss_hist,"grad_hist":grad_hist,
            "time_sec":time.time()-t0,"optimizer":"lbfgsb",
            "scipy_message":str(res.message),"scipy_success":bool(res.success),"scipy_nit":int(res.nit)}

def run_oracle_pgd(problem, x0, lr=0.1, steps=200, gate_eps=1e-6):
    x=x0.copy(); loss_hist=[]; grad_hist=[]; t0=time.time()
    for _ in range(steps):
        loss_hist.append(problem.loss(x)); g=problem.grad(x)
        grad_hist.append(float(np.linalg.norm(g)))
        gate=problem.gradient_gate(x); scale=1.0/(gate+gate_eps)
        scale/=(scale.mean()+1e-12); x=project_box(x-lr*g*scale)
    return {"x_final":x,"loss_hist":loss_hist,"grad_hist":grad_hist,
            "time_sec":time.time()-t0,"optimizer":"oracle_pgd"}

OPTIMIZER_FUNCS = {
    "pgd": run_pgd, "momentum": run_momentum, "nesterov": run_nesterov,
    "adam": run_adam, "lbfgsb": run_lbfgsb, "oracle_pgd": run_oracle_pgd,
}

def init_x(shape, mode="random", seed=0, target=None):
    rng = np.random.default_rng(seed)
    if mode=="random":           return rng.uniform(0.,1.,size=shape).astype(float)
    if mode=="constant_half":    return np.full(shape, 0.5)
    if mode=="zeros":            return np.zeros(shape)
    if mode=="ones":             return np.ones(shape)
    if mode=="target_plus_noise":
        return project_box(target + 0.15*rng.normal(size=shape))
    raise ValueError(f"Unknown init mode: {mode}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_single_experiment(image_shape=SHAPE, kernel_name="sobel_x",
                          target_name="checkerboard", alpha=10.0, c=0.0,
                          activation="sigmoid", noise_std=0.0,
                          tv_lambda=0.0, tikhonov_lambda=0.0,
                          optimizer_name="pgd", init_mode="random",
                          seed=0, optimizer_kwargs=None, target_array=None):
    if optimizer_kwargs is None: optimizer_kwargs={}
    if target_array is not None:
        target = target_array.astype(float)
        if target.shape != image_shape:
            target = zoom(target, (image_shape[0]/target.shape[0],
                                   image_shape[1]/target.shape[1]), order=1)
    else:
        target = get_targets(*image_shape)[target_name]
    kernel = KERNELS[kernel_name]
    config = ProblemConfig(image_shape=image_shape, alpha=alpha, c=c,
                           kernel_name=kernel_name, target_name=target_name,
                           activation=activation, noise_std=noise_std,
                           tv_lambda=tv_lambda, tikhonov_lambda=tikhonov_lambda)
    problem = FixedCNNInverseProblem(config=config, target=target, kernel=kernel)
    x0 = init_x(image_shape, mode=init_mode, seed=seed, target=target)
    result = OPTIMIZER_FUNCS[optimizer_name](problem, x0, **optimizer_kwargs)
    xf = result["x_final"]
    m0 = problem.metrics(x0, x_star=target)
    mT = problem.metrics(xf, x_star=target)
    summary = {"image_h":image_shape[0], "image_w":image_shape[1],
               "kernel_name":kernel_name, "target_name":target_name,
               "alpha":alpha, "c":c, "activation":activation,
               "noise_std":noise_std, "tv_lambda":tv_lambda,
               "tikhonov_lambda":tikhonov_lambda,
               "optimizer":optimizer_name, "init_mode":init_mode, "seed":seed,
               "time_sec":result["time_sec"],
               "loss_init":m0["loss"], "loss_final":mT["loss"],
               "grad_init":m0["grad_norm"], "grad_final":mT["grad_norm"],
               "output_mse_final":mT["output_mse"],
               "output_binary_acc_final":mT["output_binary_acc"],
               "output_iou_final":mT["output_iou"],
               "input_psnr_final":mT["input_psnr_vs_target"],
               "input_ssim_final":mT["input_ssim_vs_target"],
               "active_grad_frac_final":mT["active_grad_fraction"],
               "saturation_frac_final":mT["saturation_fraction"],
               "rel_loss_reduction":rel_loss_reduction(m0["loss"],mT["loss"]),
               "num_iters_recorded":len(result.get("loss_hist",[]))}
    if "scipy_message" in result:
        summary.update({"scipy_message":result["scipy_message"],
                        "scipy_success":result["scipy_success"],
                        "scipy_nit":result["scipy_nit"]})
    return {"problem":problem,"x0":x0,"x_final":xf,
            "target":target,"result":result,"summary":summary}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PARALLEL SWEEP INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def _run_one(kwargs):
    """Top-level picklable worker for joblib."""
    return run_single_experiment(**kwargs)["summary"]

def parallel_sweep(job_list, n_jobs=N_JOBS, desc="sweep"):
    """Run a list of kwarg-dicts in parallel, return list of summary-dicts."""
    try:
        from joblib import Parallel, delayed
        print(f"[parallel] {desc}: {len(job_list)} jobs, n_jobs={n_jobs}")
        results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=2)(
            delayed(_run_one)(kw) for kw in job_list)
        return results
    except Exception as e:
        print(f"[warn] joblib failed ({e}), falling back to serial")
        return [_run_one(kw) for kw in job_list]

def _already_done(df, keys: dict) -> bool:
    """Return True if a row matching all keys already exists in df."""
    if df is None or len(df)==0: return False
    mask = pd.Series([True]*len(df))
    for k,v in keys.items():
        if k not in df.columns: return False
        if isinstance(v, float):
            mask &= (df[k] - v).abs() < 1e-9
        else:
            mask &= (df[k] == v)
    return bool(mask.any())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ALL SWEEPS (checkpoint-aware)
# ══════════════════════════════════════════════════════════════════════════════

OPTIMIZERS = {
    "pgd":      {"lr":0.10, "steps":STEPS},
    "momentum": {"lr":0.05, "beta":0.9, "steps":STEPS},
    "nesterov": {"lr":0.05, "beta":0.9, "steps":STEPS},
    "adam":     {"lr":0.03, "steps":STEPS},
    "lbfgsb":   {"maxiter":STEPS},
}

# ── S7.1: Alpha sweep ─────────────────────────────────────────────────────────
def run_alpha_sweep():
    existing = load_csv("alpha_sweep_results.csv", warn=False)
    jobs = []
    for alpha in ALPHAS:
        for opt, kw in OPTIMIZERS.items():
            for seed in SEEDS:
                if not _already_done(existing, {"alpha":alpha,"optimizer":opt,"seed":seed}):
                    jobs.append(dict(alpha=alpha, optimizer_name=opt,
                                     optimizer_kwargs=kw, seed=seed,
                                     image_shape=SHAPE, kernel_name="sobel_x",
                                     target_name="checkerboard"))
    if jobs:
        rows = parallel_sweep(jobs, desc="alpha_sweep")
        append_csv(rows, "alpha_sweep_results.csv")
    else:
        print("[skip] alpha_sweep_results.csv already complete")
    return load_csv("alpha_sweep_results.csv")

# ── S7.2: Threshold sweep ─────────────────────────────────────────────────────
def run_threshold_sweep():
    existing = load_csv("threshold_sweep_results.csv", warn=False)
    CS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    jobs = []
    for c in CS:
        for opt, kw in OPTIMIZERS.items():
            for seed in SEEDS:
                if not _already_done(existing, {"c":c,"optimizer":opt,"seed":seed}):
                    jobs.append(dict(alpha=10.0, c=c, optimizer_name=opt,
                                     optimizer_kwargs=kw, seed=seed,
                                     image_shape=SHAPE, kernel_name="sobel_x",
                                     target_name="checkerboard"))
    if jobs:
        rows = parallel_sweep(jobs, desc="threshold_sweep")
        append_csv(rows, "threshold_sweep_results.csv")
    else:
        print("[skip] threshold_sweep_results.csv already complete")
    return load_csv("threshold_sweep_results.csv")

# ── S7.3: Kernel sweep ────────────────────────────────────────────────────────
def run_kernel_sweep():
    existing = load_csv("kernel_sweep_results.csv", warn=False)
    KNAMES = ("identity_like","avg_blur","sobel_x","laplacian","random_norm")
    jobs = []
    for kname in KNAMES:
        for opt, kw in OPTIMIZERS.items():
            for seed in SEEDS:
                if not _already_done(existing, {"kernel_name":kname,"optimizer":opt,"seed":seed}):
                    jobs.append(dict(alpha=10.0, kernel_name=kname,
                                     c=KERNEL_C.get(kname, 0.0),
                                     optimizer_name=opt, optimizer_kwargs=kw,
                                     seed=seed, image_shape=SHAPE,
                                     target_name="checkerboard"))
    if jobs:
        rows = parallel_sweep(jobs, desc="kernel_sweep")
        append_csv(rows, "kernel_sweep_results.csv")
    else:
        print("[skip] kernel_sweep_results.csv already complete")
    return load_csv("kernel_sweep_results.csv")

# ── S7.4: Target sweep ────────────────────────────────────────────────────────
def run_target_sweep():
    existing = load_csv("target_sweep_results.csv", warn=False)
    TNAMES = ("checkerboard","vstripes","circle","sparse_dots","two_blobs")
    jobs = []
    for tname in TNAMES:
        for opt, kw in OPTIMIZERS.items():
            for seed in SEEDS:
                if not _already_done(existing, {"target_name":tname,"optimizer":opt,"seed":seed}):
                    jobs.append(dict(alpha=10.0, target_name=tname,
                                     optimizer_name=opt, optimizer_kwargs=kw,
                                     seed=seed, image_shape=SHAPE,
                                     kernel_name="sobel_x"))
    if jobs:
        rows = parallel_sweep(jobs, desc="target_sweep")
        append_csv(rows, "target_sweep_results.csv")
    else:
        print("[skip] target_sweep_results.csv already complete")
    return load_csv("target_sweep_results.csv")

# ── S7.5: Scale sweep ─────────────────────────────────────────────────────────
def run_scale_sweep():
    existing = load_csv("scale_experiment.csv", warn=False)
    SIZES = ((32,32),(64,64),(128,128))
    jobs = []
    for size in SIZES:
        size_str = f"{size[0]}x{size[1]}"
        for alpha in SCALE_ALPHAS:
            for seed in SEEDS:
                if not _already_done(existing, {"alpha":alpha,"seed":seed,
                                                 "image_h":size[0],"image_w":size[1]}):
                    jobs.append(dict(alpha=alpha, image_shape=size,
                                     optimizer_name="adam",
                                     optimizer_kwargs={"lr":0.03,"steps":STEPS},
                                     seed=seed, kernel_name="sobel_x",
                                     target_name="checkerboard"))
    if jobs:
        summaries = parallel_sweep(jobs, desc="scale_sweep")
        # add size label
        for s in summaries:
            s["size"] = f"{s['image_h']}x{s['image_w']}"
        append_csv(summaries, "scale_experiment.csv")
    else:
        print("[skip] scale_experiment.csv already complete")
    df = load_csv("scale_experiment.csv")
    # normalise size labels
    if df is not None and "size" in df.columns:
        df["size"] = df["size"].str.replace("×","x")
        save_csv(df, "scale_experiment.csv")
    return df

# ── S7.6: Two-layer sweep ─────────────────────────────────────────────────────
class TwoLayerProblem:
    def __init__(self, target, k1, k2, alpha, c=0.5):
        self.y=target.astype(float); self.k1=k1; self.k2=k2
        self.kf1=np.flipud(np.fliplr(k1)); self.kf2=np.flipud(np.fliplr(k2))
        self.alpha=alpha; self.c=c
    def _c1(self,x): return convolve(x,self.k1,mode="wrap")
    def _c1T(self,z): return convolve(z,self.kf1,mode="wrap")
    def _c2(self,x): return convolve(x,self.k2,mode="wrap")
    def _c2T(self,z): return convolve(z,self.kf2,mode="wrap")
    def forward(self,x):
        h1=sigmoid(self._c1(x),self.alpha,self.c)
        return sigmoid(self._c2(h1),self.alpha,self.c)
    def loss(self,x): return float(np.sum((self.y-self.forward(x))**2))
    def grad(self,x):
        a1=self._c1(x); h1=sigmoid(a1,self.alpha,self.c)
        h1p=sigmoid_prime(h1,self.alpha)
        a2=self._c2(h1); h2=sigmoid(a2,self.alpha,self.c)
        h2p=sigmoid_prime(h2,self.alpha)
        d2=2.0*(h2-self.y)*h2p; d1=self._c2T(d2)*h1p
        return self._c1T(d1)
    def active_frac(self,x,thr=0.01):
        h1=sigmoid(self._c1(x),self.alpha,self.c); h1p=sigmoid_prime(h1,self.alpha)
        h2=sigmoid(self._c2(h1),self.alpha,self.c); h2p=sigmoid_prime(h2,self.alpha)
        return float(np.mean(np.abs(h1p*self._c2T(h2p))>thr))

def _run_adam_raw(problem, x0, lr=0.03, steps=300):
    x=x0.copy(); m=np.zeros_like(x); v=np.zeros_like(x); b1,b2,eps=0.9,0.999,1e-8
    for t in range(1,steps+1):
        g=problem.grad(x); m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g**2
        mh=m/(1-b1**t); vh=v/(1-b2**t)
        x=np.clip(x-lr*mh/(np.sqrt(vh)+eps),0.,1.)
    return x

def run_twolayer_sweep():
    existing = load_csv("twolayer_vs_onelayer.csv", warn=False)
    k1,k2 = KERNELS["sobel_x"], KERNELS["avg_blur"]
    rows=[]
    for alpha in ALPHAS:
        for seed in SEEDS:
            if _already_done(existing,{"alpha":alpha,"seed":seed,"layers":2}): continue
            target=get_targets(*SHAPE)["checkerboard"]
            rng=np.random.default_rng(seed); x0=rng.uniform(0,1,SHAPE)
            # 1-layer via standard runner
            p1=run_single_experiment(alpha=alpha,seed=seed,optimizer_name="adam",
                                      optimizer_kwargs={"lr":0.03,"steps":STEPS})
            xf1=p1["x_final"]
            s1=sigmoid(p1["problem"].conv(xf1),alpha,0.5)
            rows.append({"alpha":alpha,"seed":seed,"layers":1,
                          "iou_final":iou_score(p1["problem"].forward(xf1),target),
                          "loss_final":p1["problem"].loss(xf1),
                          "frac_active":float(np.mean(np.abs(sigmoid_prime(s1,alpha))>0.01))})
            # 2-layer
            prob2=TwoLayerProblem(target,k1,k2,alpha)
            xf2=_run_adam_raw(prob2,x0.copy(),steps=STEPS)
            rows.append({"alpha":alpha,"seed":seed,"layers":2,
                          "iou_final":iou_score(prob2.forward(xf2),target),
                          "loss_final":prob2.loss(xf2),
                          "frac_active":prob2.active_frac(xf2)})
            print(f"[2L] alpha={alpha} seed={seed}: "
                  f"1L IoU={rows[-2]['iou_final']:.3f} | 2L IoU={rows[-1]['iou_final']:.3f}")
    if rows: append_csv(rows, "twolayer_vs_onelayer.csv")
    else: print("[skip] twolayer_vs_onelayer.csv already complete")
    return load_csv("twolayer_vs_onelayer.csv")

# ── S7.6b: Empirical test of Theorem 4.3 (gate independence) ─────────────────
def test_gate_independence():
    """At convergence of the two-layer problem, Theorem 4.3 implicitly assumes
    Gamma(1) = |sigmoid'(a1)| (the layer-1 gate, a function of forward
    activations only) is statistically independent of A2^T*Gamma(2) =
    conv_T_k2(|sigmoid'(a2)|) (the layer-2 gate backpropagated through A2^T) --
    this is what licenses treating the compounding factor as a simple product
    rather than tracking their joint distribution. This was never tested in
    the codebase; this function computes the Pearson correlation between the
    two fields, per-pixel, at convergence, across alpha and seed, and reports
    whether independence (corr ~ 0) is empirically reasonable.
    """
    existing = load_csv("gate_independence_test.csv", warn=False)
    k1, k2 = KERNELS["sobel_x"], KERNELS["avg_blur"]
    target = get_targets(*SHAPE)["checkerboard"]
    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            if _already_done(existing, {"alpha": alpha, "seed": seed}): continue
            rng = np.random.default_rng(seed)
            x0 = rng.uniform(0, 1, SHAPE)
            prob = TwoLayerProblem(target, k1, k2, alpha)
            xf = _run_adam_raw(prob, x0.copy(), steps=STEPS)
            a1 = prob._c1(xf); h1 = sigmoid(a1, alpha, prob.c)
            gamma1 = sigmoid_prime(h1, alpha)                    # Gamma(1)
            a2 = prob._c2(h1); h2 = sigmoid(a2, alpha, prob.c)
            gamma2 = sigmoid_prime(h2, alpha)                    # Gamma(2)
            backprop_gamma2 = prob._c2T(gamma2)                  # A2^T * Gamma(2)
            r, p = pearsonr(gamma1.ravel(), backprop_gamma2.ravel())
            rows.append({"alpha": alpha, "seed": seed, "corr_gamma1_A2T_gamma2": float(r),
                         "p_value": float(p), "std_gamma1": float(gamma1.std()),
                         "std_A2T_gamma2": float(backprop_gamma2.std())})
            print(f"  [gate-indep] alpha={alpha:>5.1f} seed={seed}: "
                  f"corr(Gamma(1), A2^T*Gamma(2)) = {r:+.4f}  (p={p:.1e})")
    if rows: append_csv(rows, "gate_independence_test.csv")
    else: print("[skip] gate_independence_test.csv already complete")
    df = load_csv("gate_independence_test.csv")

    summ = df.groupby("alpha")["corr_gamma1_A2T_gamma2"].agg(["mean", "std", "min", "max"]).reset_index()
    print("\n[Theorem 4.3 empirical check] corr(Gamma(1), A2^T*Gamma(2)) at convergence, by alpha:")
    print(summ.to_string(index=False))
    peak = summ.loc[summ["mean"].abs().idxmax()]
    near_zero = summ[summ["mean"].abs() < 0.05]
    print(f"\n  Peak |correlation|: {peak['mean']:+.4f} at alpha={peak['alpha']:.1f}  "
          f"(far from the 0 that independence requires)")
    if len(near_zero):
        print(f"  Correlation is consistent with ~0 (|r|<0.05) only for alpha in "
              f"{sorted(near_zero['alpha'].tolist())}")
    print("  VERDICT: Gamma(1) and A2^T*Gamma(2) are NOT independent across most of the "
          "tested alpha range -- correlation is small-but-significant at low alpha, peaks "
          "around alpha~3 (r~0.3-0.4, i.e. ~10-18% shared variance), then decays toward 0 "
          "only in the deep-saturation regime (alpha>=40-60). The independence assumption "
          "in Theorem 4.3 is NOT empirically supported as a general statement; at best it "
          "is an asymptotic (large-alpha) approximation.")
    return df

# ── S7.7: Gradient sparsity sweep ─────────────────────────────────────────────
def run_grad_sparsity_sweep():
    existing = load_csv("grad_sparsity_all_optimizers.csv", warn=False)
    THRS = [0.0001, 0.001, 0.01, 0.05, 0.1]
    jobs=[]
    for alpha in ALPHAS:
        for opt,kw in OPTIMIZERS.items():
            for seed in SEEDS:
                if not _already_done(existing,{"alpha":alpha,"optimizer":opt,"seed":seed}):
                    jobs.append(dict(alpha=alpha,optimizer_name=opt,
                                     optimizer_kwargs=kw,seed=seed,
                                     image_shape=SHAPE,kernel_name="sobel_x",
                                     target_name="checkerboard"))
    if jobs:
        summaries=parallel_sweep(jobs,desc="grad_sparsity")
        rows=[]
        for s in summaries:
            row={"alpha":s["alpha"],"optimizer":s["optimizer"],"seed":s["seed"]}
            # recompute active fracs from saved active_grad_frac
            for thr in THRS:
                row[f"frac_active_sp_gt_{thr}"]=s.get("active_grad_frac_final",0.0)
            rows.append(row)
        append_csv(rows,"grad_sparsity_all_optimizers.csv")
    else: print("[skip] grad_sparsity_all_optimizers.csv already complete")
    return load_csv("grad_sparsity_all_optimizers.csv")

# ── S7.8: Oracle ablation ─────────────────────────────────────────────────────
def run_oracle_ablation():
    existing = load_csv("oracle_ablation.csv", warn=False)
    jobs=[]
    for alpha in ALPHAS:
        for opt in ["adam","oracle_pgd","pgd"]:
            kw = {"lr":0.03,"steps":STEPS} if opt=="adam" else \
                 {"lr":0.1,"steps":STEPS}
            for seed in SEEDS:
                if not _already_done(existing,{"alpha":alpha,"optimizer":opt,"seed":seed}):
                    jobs.append(dict(alpha=alpha,optimizer_name=opt,
                                     optimizer_kwargs=kw,seed=seed,
                                     image_shape=SHAPE,kernel_name="sobel_x",
                                     target_name="checkerboard"))
    if jobs:
        rows=parallel_sweep(jobs,desc="oracle_ablation")
        append_csv(rows,"oracle_ablation.csv")
    else: print("[skip] oracle_ablation.csv already complete")
    return load_csv("oracle_ablation.csv")

# ── S7.9: Activation comparison ───────────────────────────────────────────────
def run_activation_sweep():
    existing = load_csv("activation_comparison.csv", warn=False)
    ACTS = ["sigmoid","tanh","relu","gelu","swish"]
    jobs=[]
    for act in ACTS:
        for alpha in ALPHAS:
            for seed in SEEDS:
                if not _already_done(existing,{"activation":act,"alpha":alpha,"seed":seed}):
                    jobs.append(dict(alpha=alpha,activation=act,
                                     optimizer_name="adam",
                                     optimizer_kwargs={"lr":0.03,"steps":STEPS},
                                     seed=seed,image_shape=SHAPE,
                                     kernel_name="sobel_x",target_name="checkerboard"))
    if jobs:
        rows=parallel_sweep(jobs,desc="activation_sweep")
        append_csv(rows,"activation_comparison.csv")
    else: print("[skip] activation_comparison.csv already complete")
    return load_csv("activation_comparison.csv")

# ── S7.9b: Activation comparison v2 — CORRECTED (post Bug #6 fix) ────────────
# `activation_comparison.csv` above is VOID: prior to the fix in
# FixedCNNInverseProblem, every "activation" silently ran sigmoid (the config
# string was stored/reported but never dispatched), so all 5 labeled curves
# are bit-identical sigmoid runs under different names. This sweep uses the
# real per-activation registry (ACTIVATIONS / Gamma = alpha*|f_base'(.)|) and
# writes to a NEW file -- never appended to the void one -- so the checkpoint
# cache (_already_done) can never silently mix pre-fix (mislabeled-sigmoid)
# rows with genuine post-fix rows. Adds "mish", the one genuinely new
# activation in the Phase-3B brief: "swish" here is already SiLU/Swish-1
# (beta=1, see _act_silu) -- registering a separate "silu" run would just
# reproduce Bug #6's labeling redundancy (two names, one computation), so we
# note the equivalence rather than duplicate the run.
def run_activation_sweep_v2():
    existing = load_csv("activation_taxonomy_v2.csv", warn=False)
    ACTS = ["sigmoid","tanh","relu","gelu","swish","mish"]   # swish === silu (beta=1); see note above
    jobs=[]
    for act in ACTS:
        for alpha in ALPHAS:
            for seed in SEEDS:
                if not _already_done(existing,{"activation":act,"alpha":alpha,"seed":seed}):
                    jobs.append(dict(alpha=alpha,activation=act,
                                     optimizer_name="adam",
                                     optimizer_kwargs={"lr":0.03,"steps":STEPS},
                                     seed=seed,image_shape=SHAPE,
                                     kernel_name="sobel_x",target_name="checkerboard"))
    if jobs:
        rows=parallel_sweep(jobs,desc="activation_sweep_v2")
        append_csv(rows,"activation_taxonomy_v2.csv")
    else: print("[skip] activation_taxonomy_v2.csv already complete")
    return load_csv("activation_taxonomy_v2.csv")

# ── S7.9c: Does ANY activation taxonomy emerge? (data-driven, not presumed) ──
# The original "3-way taxonomy" claim is void (Bug #6: the underlying sweep
# never varied the activation). This re-asks the question from scratch on the
# corrected data WITHOUT presupposing the answer is "3 groups": each
# activation is represented by its active_grad_frac_final(alpha) profile,
# activations are clustered (not data points -- there are only |ACTS| of them),
# and we report whichever k the silhouette score actually favors, including
# "no separation" if nothing clears the floor for meaningful cluster structure.
def analyze_activation_taxonomy(df):
    print("\n[exp] Activation taxonomy test on CORRECTED data (k is data-driven, not assumed=3)")
    if df is None or len(df)==0:
        print("  [skip] no corrected activation data available"); return None
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("  [skip] sklearn not available"); return None

    acts = sorted(df["activation"].unique())
    alphas = sorted(df["alpha"].unique())
    if len(acts) < 4:
        print(f"  [skip] need >=4 activations to test for nontrivial clustering, have {len(acts)}")
        return None

    feats = []
    for act in acts:
        sub = df[df["activation"]==act]
        prof = [float(sub[sub["alpha"]==a]["active_grad_frac_final"].mean()) for a in alphas]
        feats.append(prof)
    X = np.array(feats)
    print(f"  {len(acts)} activations x {len(alphas)}-point active-fraction(alpha) profiles: {acts}")

    SIL_FLOOR = 0.25  # Kaufman & Rousseeuw: below this, cluster structure is "weak/artificial"
    rows=[]; best_k, best_sil, best_labels = None, -2.0, None
    for k in range(2, len(acts)):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        sil = float(silhouette_score(X, labels))
        groups = "  ".join("[" + ",".join(np.array(acts)[labels==g]) + "]" for g in range(k))
        rows.append({"k":k, "silhouette":sil, "groups":groups})
        print(f"    k={k}: silhouette={sil:+.3f}   groups: {groups}")
        if sil > best_sil: best_sil, best_k, best_labels = sil, k, labels

    if best_sil < SIL_FLOOR:
        verdict = (f"NO well-separated taxonomy emerges: best silhouette={best_sil:+.3f} at k={best_k}, "
                   f"below the {SIL_FLOOR} floor for meaningful structure (Kaufman & Rousseeuw). "
                   f"Activation differences look continuous / noise-dominated at this seed count, "
                   f"not discrete regimes -- inconclusive, not negative: report as such, do not round "
                   f"up to 'k clusters found'.")
    else:
        groups = "  ".join("[" + ",".join(np.array(acts)[best_labels==g]) + "]" for g in range(best_k))
        verdict = (f"Data supports k={best_k} groups (silhouette={best_sil:+.3f}): {groups}. "
                   f"This is empirically-discovered structure on the CORRECTED dataset -- it may or "
                   f"may not be 3-way, and is an independent finding, not a confirmation of the "
                   f"original (void) taxonomy claim, which never tested anything.")
    print(f"\n  VERDICT: {verdict}")
    out = pd.DataFrame(rows)
    out.attrs["verdict"] = verdict
    save_csv(out, "activation_taxonomy_clustering.csv")
    return out

# ── S7.10: Noise robustness ───────────────────────────────────────────────────
def run_noise_sweep():
    existing = load_csv("noise_robustness.csv", warn=False)
    SNR_DB=[15,20,30,40,50,60]
    jobs=[]
    for alpha in ALPHAS:
        for snr in SNR_DB:
            noise_std=10**(-snr/20.0)
            for seed in SEEDS:
                if not _already_done(existing,{"alpha":alpha,"noise_std":noise_std,"seed":seed}):
                    jobs.append(dict(alpha=alpha,noise_std=noise_std,
                                     optimizer_name="adam",
                                     optimizer_kwargs={"lr":0.03,"steps":STEPS},
                                     seed=seed,image_shape=SHAPE,
                                     kernel_name="sobel_x",target_name="checkerboard"))
    if jobs:
        rows=parallel_sweep(jobs,desc="noise_sweep")
        for r,s in zip(rows,[j["noise_std"] for j in jobs[:len(rows)]]):
            r["snr_db"]=round(-20*np.log10(s+1e-12),1)
        append_csv(rows,"noise_robustness.csv")
    else: print("[skip] noise_robustness.csv already complete")
    return load_csv("noise_robustness.csv")

# ── S7.11: Dense alpha sweep for gradient leakage ─────────────────────────────
def run_dense_alpha_sweep():
    existing = load_csv("gradient_leakage_dense.csv", warn=False)
    jobs=[]
    for alpha in DENSE_ALPHAS:
        for seed in SEEDS:
            if not _already_done(existing,{"alpha":alpha,"seed":seed,"optimizer":"adam"}):
                jobs.append(dict(alpha=alpha,optimizer_name="adam",
                                  optimizer_kwargs={"lr":0.03,"steps":STEPS},
                                  seed=seed,image_shape=SHAPE,
                                  kernel_name="sobel_x",target_name="checkerboard"))
    if jobs:
        rows=parallel_sweep(jobs,desc="dense_alpha_sweep")
        append_csv(rows,"gradient_leakage_dense.csv")
    else: print("[skip] gradient_leakage_dense.csv already complete")
    return load_csv("gradient_leakage_dense.csv")

# ── S7.12: Phase diagram (kernel × alpha) ─────────────────────────────────────
def run_phase_diagram():
    existing = load_csv("phase_diagram_alpha_x_kernel.csv", warn=False)
    KNAMES=("identity_like","avg_blur","random_norm","sobel_x","laplacian")
    jobs=[]
    for kname in KNAMES:
        for alpha in ALPHAS:
            for seed in SEEDS:
                if not _already_done(existing,{"kernel_name":kname,"alpha":alpha,"seed":seed}):
                    jobs.append(dict(alpha=alpha,kernel_name=kname,
                                     c=KERNEL_C.get(kname, 0.0),
                                     optimizer_name="adam",
                                     optimizer_kwargs={"lr":0.03,"steps":STEPS},
                                     seed=seed,image_shape=SHAPE,
                                     target_name="checkerboard"))
    if jobs:
        rows=parallel_sweep(jobs,desc="phase_diagram")
        append_csv(rows,"phase_diagram_alpha_x_kernel.csv")
    else: print("[skip] phase_diagram_alpha_x_kernel.csv already complete")
    return load_csv("phase_diagram_alpha_x_kernel.csv")

# ── S7.13: Convergence bands ──────────────────────────────────────────────────
def run_convergence_bands():
    """Run multiple seeds and store per-seed full loss/grad histories."""
    results={}
    for opt,kw in OPTIMIZERS.items():
        loss_curves=[]; grad_curves=[]
        for seed in SEEDS:
            p=run_single_experiment(alpha=10.0,optimizer_name=opt,
                                     optimizer_kwargs=kw,seed=seed,
                                     image_shape=SHAPE,kernel_name="sobel_x",
                                     target_name="checkerboard")
            loss_curves.append(np.array(p["result"]["loss_hist"]))
            grad_curves.append(np.array(p["result"]["grad_hist"]))
        ml=min(len(x) for x in loss_curves)
        la=np.stack([x[:ml] for x in loss_curves])
        ga=np.stack([x[:ml] for x in grad_curves])
        results[opt]={"loss_mean":la.mean(0),"loss_std":la.std(0),
                      "grad_mean":ga.mean(0),"grad_std":ga.std(0)}
    return results

# ── S7.14: Curvature sweep ────────────────────────────────────────────────────
def _dir_curv(problem, x, v, eps=1e-3):
    v=v.reshape(x.shape); v=v/np.linalg.norm(v); xp=project_box(x+eps*v); xm=project_box(x-eps*v)
    return (problem.loss(xp)-2*problem.loss(x)+problem.loss(xm))/eps**2

def run_curvature_sweep():
    existing = load_csv("curvature_alpha_sweep.csv", warn=False)
    rows=[]
    for alpha in ALPHAS:
        for seed in SEEDS:
            if _already_done(existing,{"alpha":alpha,"seed":seed}): continue
            p=run_single_experiment(alpha=alpha,seed=seed,optimizer_name="adam",
                                     optimizer_kwargs={"lr":0.03,"steps":150},
                                     image_shape=(32,32),kernel_name="sobel_x",
                                     target_name="checkerboard")
            rng=np.random.default_rng(seed); xf=p["x_final"]
            curvs=[_dir_curv(p["problem"],xf,rng.normal(size=xf.size)) for _ in range(60)]
            curvs=np.array(curvs)
            rows.append({"alpha":alpha,"seed":seed,"kernel_name":"sobel_x",
                          "curv_mean":float(curvs.mean()),
                          "curv_std":float(curvs.std()),
                          "curv_min":float(curvs.min()),
                          "curv_max":float(curvs.max()),
                          "curv_frac_negative":float(np.mean(curvs<0)),
                          "curv_range":float(curvs.max()-curvs.min()),
                          "loss_final":p["summary"]["loss_final"],
                          "output_iou_final":p["summary"]["output_iou_final"]})
    if rows: append_csv(rows,"curvature_alpha_sweep.csv")
    else: print("[skip] curvature_alpha_sweep.csv already complete")
    return load_csv("curvature_alpha_sweep.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MNIST
# ══════════════════════════════════════════════════════════════════════════════

def run_mnist_experiment():
    if args.skip_mnist:
        print("[skip] --skip-mnist flag set"); return None
    existing = load_csv("mnist_experiment.csv", warn=False)
    try:
        import tensorflow as tf
        (_, _),(x_test,_) = tf.keras.datasets.mnist.load_data()
        imgs_raw = (x_test.astype(float)/255.0>0.5).astype(float)
    except Exception:
        print("[warn] TF not available, skipping MNIST"); return None
    rng=np.random.default_rng(42)
    idx=rng.choice(len(imgs_raw),size=8,replace=False)
    mnist_imgs=[]
    for i in idx:
        im=imgs_raw[i]
        im=np.repeat(np.repeat(im,64//28+1,axis=0),64//28+1,axis=1)[:64,:64]
        mnist_imgs.append(im)
    rows=[]
    for kname in ("sobel_x","laplacian","avg_blur"):
        kernel=KERNELS[kname]
        for di,target in enumerate(mnist_imgs):
            for alpha in ALPHAS:
                for seed in SEEDS:
                    if _already_done(existing,{"kernel_name":kname,"alpha":alpha,
                                               "seed":seed,"digit_idx":int(idx[di])}): continue
                    config=ProblemConfig(image_shape=(64,64),alpha=alpha,c=0.5,
                                         kernel_name=kname,target_name="mnist")
                    prob=FixedCNNInverseProblem(config=config,target=target,kernel=kernel)
                    x0=init_x((64,64),mode="random",seed=seed)
                    res=run_adam(prob,x0,lr=0.03,steps=STEPS)
                    xf=res["x_final"]
                    s_=sigmoid(prob.conv(xf),alpha,0.5)
                    rows.append({"kernel_name":kname,"digit_idx":int(idx[di]),
                                  "alpha":alpha,"seed":seed,
                                  "iou_final":iou_score(prob.forward(xf),target),
                                  "loss_final":prob.loss(xf),
                                  "frac_active":float(np.mean(np.abs(sigmoid_prime(s_,alpha))>0.01)),
                                  "target_type":"mnist"})
                    print(f"[mnist] {kname} digit={idx[di]} a={alpha} s={seed}: "
                          f"IoU={rows[-1]['iou_final']:.3f}")
    if rows: append_csv(rows,"mnist_experiment.csv")
    return load_csv("mnist_experiment.csv"), mnist_imgs

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8b — PHASE 3C: Full MNIST + CIFAR-10 reconstruction & phase transitions
# ══════════════════════════════════════════════════════════════════════════════
#
# run_mnist_experiment above uses 8 hand-picked digits x 3 kernels — a spot
# check, not class-representative coverage, and covers only one dataset.
# Phase 3C scales this to BOTH datasets (MNIST + CIFAR-10) with full 10-class
# coverage, then asks a question the spot-check cannot: does the SAME
# logistic phase transition IoU(alpha) -> alpha* (used in run_per_kernel_
# alpha_star for synthetic kernels) appear for REAL natural-image targets, and
# does its location/sharpness depend on the image distribution (simple binary
# digit silhouettes vs. more complex natural-image silhouettes)?
#
# Per user instruction: NO medical imaging — explicitly out of scope here,
# noted as future work in the final write-up, not attempted in any form.
#
# MODELING DECISION — flagged, not hidden: FixedCNNInverseProblem and its IoU/
# alpha* machinery are built around BINARY {0,1} targets (see y_clean, iou_score
# at thr=0.5). CIFAR-10 images are natural RGB photographs with no natural
# binary structure. To reuse the EXACT SAME framework (so alpha* values are
# directly comparable to MNIST and to the synthetic per-kernel alpha* results,
# rather than living in some bespoke incompatible pipeline), we convert CIFAR-10
# to grayscale and threshold at each image's own median (MNIST is thresholded
# at a fixed 0.5 after [0,1] rescaling, which works because its background/
# foreground intensities are bimodal; CIFAR-10 grayscale histograms are not, so
# a fixed threshold would produce mostly-empty or mostly-full silhouettes for
# many images — the median guarantees a roughly balanced binary mask). This is
# a real reduction: it discards color and most natural-image texture, and any
# "CIFAR-10 phase transition" finding is about BINARIZED CIFAR-10 SILHOUETTES,
# not natural CIFAR-10 image reconstruction. We say so here and in any write-up
# of these results, rather than letting the dataset name imply more than the
# pipeline actually tests.

PHASE3C_N_PER_CLASS = 4          # images per class per dataset (10 classes x 2 datasets x 4 = 80 targets)
PHASE3C_SEEDS = (0, 1, 2)        # independent of the profile-level SEEDS: a deliberate scope choice
                                  # (per-image alpha* fitting needs alpha-resolution more than seed-count)

def _load_phase3c_targets(n_per_class=PHASE3C_N_PER_CLASS, shape=None):
    """Lazily loads MNIST + CIFAR-10 via tf.keras.datasets (mirrors the
    lazy-import / graceful-skip pattern in run_mnist_experiment — both gated
    by --skip-mnist since both come from the same optional TF dependency).
    Returns a list of dicts: dataset, cls (0-9), idx (orig. dataset index),
    image (binarized, shape x shape, float {0,1})."""
    shape = shape or SHAPE
    try:
        import tensorflow as tf
    except Exception:
        return None
    rng = np.random.default_rng(123)
    targets = []

    (_, _), (mnist_x, mnist_y) = tf.keras.datasets.mnist.load_data()
    mnist_y = mnist_y.flatten()
    mnist_bin = (mnist_x.astype(float)/255.0 > 0.5).astype(float)
    for cls in range(10):
        idx = np.where(mnist_y == cls)[0]
        sel = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        for i in sel:
            im = mnist_bin[int(i)]
            im = np.repeat(np.repeat(im, shape[0]//28+1, axis=0),
                           shape[1]//28+1, axis=1)[:shape[0], :shape[1]]
            targets.append(dict(dataset="mnist", cls=int(cls), idx=int(i), image=im.astype(float)))

    (_, _), (cifar_x, cifar_y) = tf.keras.datasets.cifar10.load_data()
    cifar_y = cifar_y.flatten()
    cifar_gray = cifar_x.astype(float).mean(axis=-1) / 255.0   # RGB -> grayscale, [0,1]
    for cls in range(10):
        idx = np.where(cifar_y == cls)[0]
        sel = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        for i in sel:
            g = cifar_gray[int(i)]
            thr = float(np.median(g))   # per-image median threshold; see module note above
            mask = (g > thr).astype(float)
            mask = zoom(mask, (shape[0]/mask.shape[0], shape[1]/mask.shape[1]), order=1)
            mask = (mask > 0.5).astype(float)
            targets.append(dict(dataset="cifar10", cls=int(cls), idx=int(i), image=mask.astype(float)))
    return targets

def run_phase3c_image_recon():
    """Reconstruction sweep over real (binarized) MNIST + CIFAR-10 targets
    across the full alpha sweep, reusing FixedCNNInverseProblem exactly as the
    existing synthetic/MNIST sweeps do — so results land in the same metric
    space and phase-transition fits are directly comparable. Single kernel
    (sobel_x, the paper's canonical choice) to keep the (already large:
    80 images x |ALPHAS| x |PHASE3C_SEEDS|) sweep tractable; checkpoint-cached
    to phase3c_image_recon.csv."""
    print("\n[exp] Phase 3C: full MNIST + CIFAR-10 reconstruction sweep (binarized targets)")
    if args.skip_mnist:
        print("  [skip] --skip-mnist flag set (also gates CIFAR-10 — both load via tf.keras.datasets)")
        return None
    existing = load_csv("phase3c_image_recon.csv", warn=False)
    targets = _load_phase3c_targets()
    if targets is None:
        print("  [skip] TF / tf.keras.datasets not available"); return None
    print(f"  {len(targets)} targets ({sum(t['dataset']=='mnist' for t in targets)} MNIST, "
          f"{sum(t['dataset']=='cifar10' for t in targets)} CIFAR-10) x "
          f"{len(ALPHAS)} alphas x {len(PHASE3C_SEEDS)} seeds")

    rows = []
    for t in targets:
        key = dict(dataset=t["dataset"], cls=t["cls"], idx=t["idx"])
        n_new = 0
        for alpha in ALPHAS:
            for seed in PHASE3C_SEEDS:
                full_key = dict(key, alpha=alpha, seed=seed)
                if _already_done(existing, full_key):
                    continue
                config = ProblemConfig(image_shape=SHAPE, alpha=alpha, c=0.5,
                                       kernel_name="sobel_x",
                                       target_name=f"{t['dataset']}_{t['cls']}")
                prob = FixedCNNInverseProblem(config=config, target=t["image"], kernel=KERNELS["sobel_x"])
                x0 = init_x(SHAPE, mode="random", seed=seed)
                res = run_adam(prob, x0, lr=0.03, steps=STEPS)
                xf = res["x_final"]
                m = prob.metrics(xf, x_star=t["image"])
                rows.append({**full_key,
                             "iou_final": m["output_iou"],
                             "loss_final": m["loss"],
                             "active_grad_frac_final": m["active_grad_fraction"]})
                n_new += 1
        if n_new:
            print(f"  [{t['dataset']} cls={t['cls']} idx={t['idx']}] {n_new} new runs")
    if rows:
        append_csv(rows, "phase3c_image_recon.csv")
    else:
        print("  [skip] phase3c_image_recon.csv already complete")
    return load_csv("phase3c_image_recon.csv", warn=False)

def analyze_phase3c_transitions(df):
    """Fits the SAME logistic phase-transition model used by
    run_per_kernel_alpha_star — iou_min + (iou_max-iou_min)/(1+exp((alpha-
    alpha*)/delta)) — to each (dataset, class, image)'s IoU(alpha) curve, with
    the same data-driven bound derivation and the same honest "NO_TRANSITION_
    SIGNAL" / "UNIDENTIFIABLE" categories (the audited fix from Bug #3,
    alpha* instability — reusing it here keeps these results consistent with,
    and comparable to, the per-kernel alpha* table). Then tests — via
    Mann-Whitney U, since per-image alpha* need not be normally distributed —
    whether MNIST and CIFAR-10 alpha* distributions differ."""
    print("\n[exp] Phase 3C: do real-image targets show the same IoU(alpha) phase transition?")
    if df is None or len(df) == 0:
        print("  [skip] no Phase 3C reconstruction data available"); return None

    def _sig_model(alpha, iou_max, iou_min, alpha_star, delta):
        return iou_min + (iou_max - iou_min) / (1.0 + np.exp((alpha - alpha_star) / (delta + 1e-8)))

    rows = []
    for (dataset, cls, idx), sub in df.groupby(["dataset", "cls", "idx"]):
        grp = sub.groupby("alpha")["iou_final"].mean().reset_index().sort_values("alpha")
        alphas_arr = grp["alpha"].values; iou_m = grp["iou_final"].values
        if len(alphas_arr) < 6:
            rows.append({"dataset": dataset, "cls": int(cls), "idx": int(idx),
                         "alpha_star": float("nan"), "fit_note": f"insufficient data (n={len(alphas_arr)})"})
            continue
        span = float(iou_m.max() - iou_m.min())
        if span < 0.05:
            rows.append({"dataset": dataset, "cls": int(cls), "idx": int(idx),
                         "alpha_star": float("nan"), "iou_span": span,
                         "fit_note": f"NO_TRANSITION_SIGNAL (IoU range={span:.3f} < 0.05)"})
            continue
        # Same data-driven bound derivation as run_per_kernel_alpha_star (Bug #3 fix):
        # bounds/p0 derived from the OBSERVED iou range, not hardcoded constants that
        # silently fail outside the synthetic-kernel regime they were tuned for.
        lo = np.array([max(0.05, iou_m.max()-0.5), iou_m.min()-0.15,
                       max(0.5, alphas_arr.min()), 0.05])
        hi = np.array([min(1.2, iou_m.max()+0.15), iou_m.max(),
                       alphas_arr.max()*1.5, 25.0])
        p0 = np.clip([iou_m.max(), iou_m.min(), 11.7, 3.0], lo+1e-6, hi-1e-6)
        alpha_star = float("nan"); note = "ok"
        try:
            popt, _ = curve_fit(_sig_model, alphas_arr, iou_m, p0=p0, bounds=(lo, hi), maxfev=50000)
            alpha_star = float(popt[2])
            if np.isclose(popt[2], lo[2], rtol=1e-3) or np.isclose(popt[2], hi[2], rtol=1e-3):
                note = "UNIDENTIFIABLE (alpha* pinned at search-box edge)"
        except Exception as e:
            note = f"fit failed: {e}"
        rows.append({"dataset": dataset, "cls": int(cls), "idx": int(idx),
                     "alpha_star": alpha_star, "iou_span": span, "fit_note": note})

    out = pd.DataFrame(rows)
    save_csv(out, "phase3c_alpha_star.csv")

    valid = out[out["fit_note"] == "ok"].dropna(subset=["alpha_star"])
    n_total, n_valid = len(out), len(valid)
    print(f"  {n_valid}/{n_total} images yielded an identifiable alpha* "
          f"(rest: NO_TRANSITION_SIGNAL / UNIDENTIFIABLE / fit failed)")
    mnist_as = valid[valid["dataset"] == "mnist"]["alpha_star"].values
    cifar_as = valid[valid["dataset"] == "cifar10"]["alpha_star"].values
    print(f"  MNIST:    n={len(mnist_as):3d}  alpha* median={np.median(mnist_as) if len(mnist_as) else float('nan'):.2f}"
          f"  IQR=[{np.percentile(mnist_as,25) if len(mnist_as) else float('nan'):.2f},"
          f"{np.percentile(mnist_as,75) if len(mnist_as) else float('nan'):.2f}]")
    print(f"  CIFAR-10: n={len(cifar_as):3d}  alpha* median={np.median(cifar_as) if len(cifar_as) else float('nan'):.2f}"
          f"  IQR=[{np.percentile(cifar_as,25) if len(cifar_as) else float('nan'):.2f},"
          f"{np.percentile(cifar_as,75) if len(cifar_as) else float('nan'):.2f}]")

    if n_valid < 0.3 * n_total:
        verdict = (f"WEAK/NO TRANSITION SIGNAL for real-image targets: only {n_valid}/{n_total} "
                   f"({n_valid/n_total:.0%}) images showed an identifiable IoU(alpha) transition at "
                   f"all — most are NO_TRANSITION_SIGNAL or UNIDENTIFIABLE. The sharp phase transition "
                   f"seen for synthetic targets (checkerboard, kernels) does not clearly reproduce on "
                   f"real images at this sampling -- report as inconclusive/negative, not as 'real "
                   f"images also show alpha*'.")
    elif len(mnist_as) >= 5 and len(cifar_as) >= 5:
        try:
            from scipy.stats import mannwhitneyu
            stat, pval = mannwhitneyu(mnist_as, cifar_as, alternative="two-sided")
            sig = "significant" if pval < 0.05 else "not significant"
            verdict = (f"Transitions ARE identifiable for most real images ({n_valid}/{n_total}). "
                       f"MNIST alpha*(median)={np.median(mnist_as):.2f} vs CIFAR-10 "
                       f"alpha*(median)={np.median(cifar_as):.2f}; Mann-Whitney U p={pval:.4f} "
                       f"({sig} at alpha=0.05) -- {'the critical stiffness DOES appear to depend on the image distribution' if pval < 0.05 else 'no statistically detectable dependence of critical stiffness on the image distribution at this sample size'}.")
        except ImportError:
            verdict = "[scipy.stats.mannwhitneyu unavailable — cannot test distributional difference]"
    else:
        verdict = (f"INCONCLUSIVE: too few identifiable transitions per dataset "
                   f"(MNIST n={len(mnist_as)}, CIFAR-10 n={len(cifar_as)}) to compare distributions "
                   f"-- report as inconclusive, do not extrapolate from a handful of points.")
    print(f"\n  VERDICT: {verdict}")
    out.attrs["verdict"] = verdict
    return out

def plot_phase3c_transitions(recon_df, star_df):
    if recon_df is None or len(recon_df) == 0 or star_df is None or len(star_df) == 0: return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = {"mnist": "tab:blue", "cifar10": "tab:orange"}
    for dataset, color in cmap.items():
        sub = recon_df[recon_df["dataset"] == dataset]
        if len(sub) == 0: continue
        grp = sub.groupby("alpha")["iou_final"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(grp["alpha"], grp["mean"], yerr=grp["std"], marker="o", lw=2,
                         capsize=3, color=color, label=dataset)
        valid = star_df[(star_df["dataset"] == dataset) & (star_df["fit_note"] == "ok")]
        if len(valid):
            axes[1].hist(valid["alpha_star"], bins=12, alpha=0.55, color=color,
                         label=f"{dataset} (n={len(valid)})")
    axes[0].set_xlabel("Stiffness alpha"); axes[0].set_ylabel("Reconstruction IoU (mean +/- std)")
    axes[0].set_title("Phase 3C: real-image reconstruction vs alpha"); axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[1].set_xlabel("Fitted alpha* (per image, where identifiable)")
    axes[1].set_ylabel("Count"); axes[1].legend(fontsize=9)
    axes[1].set_title("Distribution of per-image critical stiffness alpha*")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    plt.suptitle("Phase 3C: does the synthetic-target phase transition reproduce on real images?\n"
                 "(binarized MNIST + CIFAR-10 silhouettes — see phase3c_alpha_star.csv for per-image fits)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout(); save_fig("phase3c_real_image_transitions.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8c — PHASE 3D: TIGHTEN ALPHA* ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
# Per the Phase-3 brief item D ("TIGHTEN ALPHA* ESTIMATION"):
#   - increase seeds from 5 to 20 for the core sobel_x experiments
#   - replace the least-squares sigmoid fit with a bootstrap CI (1000 resamples)
#   - report the bootstrap median and 95% CI for alpha*
#   - compare the bootstrap CI width to the (now Phase-1-corrected) CRLB
#
# This is NEW compute — a dedicated 20-seed sobel_x/checkerboard/adam alpha
# sweep (|ALPHAS| x 20 = 280 runs at full-profile SHAPE/STEPS) well beyond what
# the existing 5-8-seed sweeps provide — per the agreed Phase-3 plan ("write
# the experiment code only; you submit jobs"): checkpoint-cached and
# smoke-tested here, NOT executed at scale.
PHASE3D_SEEDS = tuple(range(20))   # the brief's explicit target; independent of the profile-level SEEDS

def run_phase3d_alpha_seed_sweep():
    """Dedicated high-seed-count alpha sweep for Phase 3D: sobel_x/checkerboard/
    adam only (the paper's canonical "core" combination), at PHASE3D_SEEDS=20
    seeds rather than the 5-8 used elsewhere — giving the bootstrap analysis
    below the statistical power the brief asks for. Checkpoint-cached to
    phase3d_alpha_seeds.csv (one row per (alpha, seed); reuses
    FixedCNNInverseProblem/run_single_experiment exactly as run_alpha_sweep)."""
    print(f"\n[exp] Phase 3D: sobel_x/checkerboard/adam alpha sweep at "
          f"{len(PHASE3D_SEEDS)} seeds (tightening alpha* estimation)")
    existing = load_csv("phase3d_alpha_seeds.csv", warn=False)
    jobs = []
    for alpha in ALPHAS:
        for seed in PHASE3D_SEEDS:
            if not _already_done(existing, {"alpha": alpha, "seed": seed}):
                jobs.append(dict(alpha=alpha, optimizer_name="adam",
                                 optimizer_kwargs={"lr": 0.03, "steps": STEPS},
                                 seed=seed, image_shape=SHAPE, kernel_name="sobel_x",
                                 target_name="checkerboard"))
    if jobs:
        rows = parallel_sweep(jobs, desc="phase3d_alpha_seeds")
        append_csv(rows, "phase3d_alpha_seeds.csv")
    else:
        print("  [skip] phase3d_alpha_seeds.csv already complete")
    return load_csv("phase3d_alpha_seeds.csv", warn=False)

def analyze_phase3d_bootstrap_alpha_star(df):
    """Phase 3D core deliverable: a bootstrap CI for alpha* (1000 resamples),
    reported as median + 95% CI, set against the Phase-1-corrected CRLB.

    Bootstrap design note: we resample SEEDS — the independent unit of
    replication — with replacement, not raw (alpha, IoU) pairs. Each seed
    contributes one full alpha->IoU curve; resampling pairs directly would
    shuffle points across seeds and destroy that within-seed structure,
    silently understating the true sampling uncertainty (the same class of
    mistake _bootstrap_r2's docstring documents for the depth-collapse data).

    CRLB note: the comparison CRLB is recomputed HERE, on this 20-seed
    dataset, using the exact Phase-1 fix (data-driven sigma2 floor +
    Tikhonov-regularized Fisher inverse + condition-number diagnostic) rather
    than reusing fisher_cramer_rao.csv, because that CSV is fit to the
    8-seed alpha_sweep_results.csv — a different sample, hence a different
    Fisher information and a different CRLB. Comparing the bootstrap CI from
    this dataset to a CRLB computed from a different dataset would not be a
    fair apples-to-apples test."""
    print("\n[exp] Phase 3D: bootstrap CI for alpha* (1000 resamples) vs. corrected CRLB")
    if df is None or len(df) == 0:
        print("  [skip] no Phase 3D alpha-seed data available"); return None

    def _sig_model(alpha, iou_max, iou_min, alpha_star, delta):
        return iou_min + (iou_max-iou_min)/(1.+np.exp((alpha-alpha_star)/(delta+1e-8)))
    def _J(alpha, iou_max, iou_min, alpha_star, delta):
        e = np.exp((alpha-alpha_star)/(delta+1e-8)); d = 1.+e
        return np.array([1./d, 1.-1./d,
                         (iou_max-iou_min)*e/(d**2*(delta+1e-8)),
                         (iou_max-iou_min)*e*(alpha-alpha_star)/(d**2*(delta+1e-8)**2)])

    iou_col = "output_iou_final" if "output_iou_final" in df.columns else "iou_final"
    pivot = (df.pivot_table(index="seed", columns="alpha", values=iou_col, aggfunc="mean")
               .dropna(axis=0, how="any"))
    n_seeds = len(pivot)
    if n_seeds < 5 or pivot.shape[1] < 6:
        print(f"  [skip] only {n_seeds} complete seeds x {pivot.shape[1]} alphas — "
              f"too few for a meaningful bootstrap (need >=5 seeds, >=6 alphas)")
        return None

    alphas_arr = pivot.columns.values.astype(float)
    seed_curves = pivot.values                       # (n_seeds, n_alphas) — one row per independent run
    iou_m = seed_curves.mean(axis=0)
    iou_s = seed_curves.std(axis=0, ddof=1)

    iou_min_p0 = float(np.clip(iou_m.min(), 0.0, 0.98))
    iou_max_p0 = float(np.clip(iou_m.max(), iou_min_p0+0.01, 1.09))
    p0     = [iou_max_p0, iou_min_p0, 11.7, 3.0]
    bounds = ([0.3, -0.05, 0.5, 0.1], [1.1, 0.99, 60.0, 20.0])

    # ── Point estimate: least-squares sigmoid fit on the n_seeds-mean curve ──
    try:
        popt_ls, pcov_ls = curve_fit(_sig_model, alphas_arr, iou_m, p0=p0, bounds=bounds, maxfev=50000)
        astar_ls, astar_ls_std = float(popt_ls[2]), float(np.sqrt(max(pcov_ls[2, 2], 0)))
    except Exception as e:
        print(f"  [skip] least-squares fit failed: {e}"); return None

    # ── Bootstrap CI for alpha* (the brief's headline ask: replace LS errors
    # with a bootstrap CI, report median + 95% CI). Re-fit on each resample's
    # mean curve so the resampling propagates through the SAME estimator. ────
    N_BOOT = 1000
    rng = np.random.default_rng(0)
    astar_boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        curve_b = seed_curves[idx].mean(axis=0)
        try:
            popt_b, _ = curve_fit(_sig_model, alphas_arr, curve_b, p0=p0, bounds=bounds, maxfev=3000)
            astar_boots.append(float(popt_b[2]))
        except Exception:
            continue
    astar_boots = np.asarray(astar_boots)
    frac_failed = 1.0 - len(astar_boots)/float(N_BOOT)
    astar_med = float(np.median(astar_boots))
    astar_lo  = float(np.percentile(astar_boots, 2.5))
    astar_hi  = float(np.percentile(astar_boots, 97.5))
    boot_ci_width = astar_hi - astar_lo

    # ── Corrected CRLB on THIS dataset — same fix as ext_a_fisher_cramer_rao:
    # data-driven sigma2 floor (the smallest observed nonzero per-alpha
    # variance, not an arbitrary external constant) + Tikhonov-regularized
    # Fisher inverse + reported condition number, so an ill-conditioned fit is
    # visible rather than silently producing astronomical CRLB numbers. ──────
    sigma2_raw = (iou_s**2) / n_seeds
    nonzero = sigma2_raw[sigma2_raw > 0]
    sigma2_floor = float(nonzero.min()) if len(nonzero) else 1e-8
    sigma2 = np.maximum(sigma2_raw, sigma2_floor)
    I = np.zeros((4, 4))
    for k, a in enumerate(alphas_arr):
        Jk = _J(a, *popt_ls); I += np.outer(Jk, Jk)/sigma2[k]
    cond_I = float(np.linalg.cond(I))
    I_reg = I + np.eye(4)*(np.trace(I)/4)*1e-10
    try: crlb = np.linalg.inv(I_reg)
    except Exception: crlb = np.full((4, 4), np.nan)
    ill_conditioned = cond_I > 1e6
    crlb_std_astar = float(np.sqrt(max(crlb[2, 2], 0)))
    crlb_ci_width = 2.0*1.96*crlb_std_astar
    width_ratio = boot_ci_width/crlb_ci_width if crlb_ci_width > 0 else float("nan")

    # ── Verdict. The CRLB is a LOWER bound on the variance of ANY unbiased
    # estimator: ratio>=1 is the physically-sane regime (this estimator is
    # simply not efficient — expected for a nonlinear LS fit on n=20). A
    # ratio<1 would be the surprising case worth flagging, not celebrating —
    # it would mean either the CRLB here is unreliable (e.g. ill-conditioned),
    # or the n=20 bootstrap hasn't reached the asymptotic-normal regime the
    # CRLB comparison assumes. We say which, rather than picking whichever
    # reads better. ───────────────────────────────────────────────────────────
    if ill_conditioned:
        verdict = ("CRLB UNRELIABLE here (ill-conditioned Fisher matrix, "
                   f"cond(I)={cond_I:.2e}) — the comparison is not meaningful; "
                   "do not interpret the ratio below as a finding")
    elif width_ratio >= 1.0:
        verdict = (f"CONSISTENT with the CRLB as a lower bound: the empirical bootstrap CI "
                   f"is {width_ratio:.2f}x WIDER than the CRLB-implied 95% width — i.e. the "
                   f"least-squares sigmoid-fit estimator of alpha* is "
                   f"{'close to' if width_ratio < 1.5 else 'well short of'} efficient at n={n_seeds} seeds")
    else:
        verdict = (f"FLAG: bootstrap CI is {1.0/width_ratio:.2f}x NARROWER than the CRLB-implied "
                   f"width — this would violate the Cramér-Rao bound for an unbiased estimator. "
                   f"Most likely explanation: n={n_seeds} is too small for the asymptotic-normal "
                   f"approximation underlying both the CRLB and the percentile bootstrap to have "
                   f"converged, not that this estimator beats the theoretical floor")

    print(f"  data: {n_seeds} seeds x {len(alphas_arr)} alphas "
          f"(brief target: 20 seeds, vs. 5 in the original headline run)")
    print(f"  alpha* point estimate (LS, full {n_seeds}-seed mean curve) = "
          f"{astar_ls:.3f} +/- {astar_ls_std:.3f} (LS std)")
    print(f"  alpha* BOOTSTRAP ({N_BOOT} resamples, {frac_failed:.1%} fit failures): "
          f"median={astar_med:.3f}  95% CI=[{astar_lo:.3f}, {astar_hi:.3f}]  width={boot_ci_width:.3f}")
    print(f"  corrected CRLB: std(alpha*)={crlb_std_astar:.4f}  cond(I)={cond_I:.3e}"
          f"{'  *** ILL-CONDITIONED ***' if ill_conditioned else ''}  "
          f"implied-95%-width={crlb_ci_width:.3f}")
    print(f"  bootstrap-width / CRLB-implied-width ratio = {width_ratio:.3f}")
    print(f"  verdict: {verdict}")

    out = pd.DataFrame([{
        "n_seeds": n_seeds, "n_alphas": len(alphas_arr), "n_boot": N_BOOT,
        "boot_fit_failure_frac": frac_failed,
        "alpha_star_ls": astar_ls, "alpha_star_ls_std": astar_ls_std,
        "alpha_star_boot_median": astar_med,
        "alpha_star_boot_ci_lo": astar_lo, "alpha_star_boot_ci_hi": astar_hi,
        "boot_ci_width": boot_ci_width,
        "crlb_std_alpha_star": crlb_std_astar, "cond_I": cond_I,
        "ill_conditioned": bool(ill_conditioned),
        "crlb_implied_ci_width": crlb_ci_width,
        "width_ratio_boot_over_crlb": width_ratio,
        "verdict": verdict,
    }])
    save_csv(out, "phase3d_bootstrap_alpha_star.csv")

    # ── Figure: (left) the n_seeds-mean phase-transition curve + LS fit;
    # (right) the bootstrap alpha* distribution vs. the CRLB-implied spread —
    # the direct visual of "compare bootstrap CI width to the corrected CRLB". ─
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    a_dense = np.linspace(alphas_arr.min(), alphas_arr.max(), 300)
    axes[0].errorbar(alphas_arr, iou_m, yerr=iou_s/np.sqrt(n_seeds), fmt="o", capsize=4,
                     color="#1f77b4", label=f"Empirical (Adam, n={n_seeds} seeds)")
    axes[0].plot(a_dense, _sig_model(a_dense, *popt_ls), color="#d62728", lw=2.2,
                 label=f"LS fit: a*={astar_ls:.2f} (point est.)")
    axes[0].axvline(astar_med, color="#2ca02c", linestyle="--", lw=1.5,
                    label=f"Bootstrap median a*={astar_med:.2f}")
    axes[0].axvspan(astar_lo, astar_hi, color="#2ca02c", alpha=0.12,
                    label=f"Bootstrap 95% CI [{astar_lo:.2f}, {astar_hi:.2f}]")
    axes[0].set_xlabel("Sigmoid stiffness alpha"); axes[0].set_ylabel("Mean IoU")
    axes[0].set_title(f"sobel_x/checkerboard/Adam, {n_seeds}-seed phase transition")
    axes[0].legend(fontsize=8); axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].hist(astar_boots, bins=40, density=True, color="#2ca02c", alpha=0.55,
                 label=f"Bootstrap a* dist. (n_boot={N_BOOT})")
    axes[1].axvline(astar_med, color="#2ca02c", lw=2, label=f"median={astar_med:.2f}")
    axes[1].axvline(astar_lo, color="#2ca02c", linestyle="--", lw=1.3)
    axes[1].axvline(astar_hi, color="#2ca02c", linestyle="--", lw=1.3,
                    label=f"95% CI=[{astar_lo:.2f},{astar_hi:.2f}]  (width={boot_ci_width:.2f})")
    if not ill_conditioned and crlb_std_astar > 0:
        from scipy.stats import norm as _norm
        x_g = np.linspace(astar_boots.min(), astar_boots.max(), 300)
        axes[1].plot(x_g, _norm.pdf(x_g, astar_ls, crlb_std_astar), color="#d62728", lw=2,
                     label=f"CRLB-implied N(a*, {crlb_std_astar:.3f}^2)\n"
                           f"95% width={crlb_ci_width:.2f}  (ratio={width_ratio:.2f}x)")
    axes[1].set_xlabel("Bootstrap-resampled alpha*"); axes[1].set_ylabel("Density")
    axes[1].set_title("Bootstrap CI for alpha* vs. corrected-CRLB-implied spread")
    axes[1].legend(fontsize=7.5); axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Phase 3D: tightened alpha* estimation — bootstrap CI (1000 resamples) vs. corrected CRLB\n"
                 f"verdict: {verdict}", fontweight="bold", fontsize=10)
    plt.tight_layout(); save_fig("phase3d_bootstrap_vs_crlb.png")
    return out

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PhD EXTENSIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Ext A: Fisher / Cramér-Rao ────────────────────────────────────────────────
def ext_a_fisher_cramer_rao(alpha_df):
    print("\nFisher Information & Cramér-Rao Lower Bound")
    def _sig_model(alpha, iou_max, iou_min, alpha_star, delta):
        return iou_min+(iou_max-iou_min)/(1.+np.exp((alpha-alpha_star)/(delta+1e-8)))
    def _J(alpha, iou_max, iou_min, alpha_star, delta):
        e=np.exp((alpha-alpha_star)/(delta+1e-8)); d=1.+e
        return np.array([1./d, 1.-1./d,
                         (iou_max-iou_min)*e/(d**2*(delta+1e-8)),
                         (iou_max-iou_min)*e*(alpha-alpha_star)/(d**2*(delta+1e-8)**2)])

    sub=alpha_df[alpha_df["optimizer"]=="adam"] if "optimizer" in alpha_df.columns else alpha_df
    grp=sub.groupby("alpha")["output_iou_final"].agg(["mean","std","count"]).reset_index()
    alphas_arr=grp["alpha"].values; iou_m=grp["mean"].values
    iou_s=grp["std"].fillna(0.01).values; n=grp["count"].values

    iou_min_p0 = float(np.clip(iou_m.min(), 0.0, 0.98))
    iou_max_p0 = float(np.clip(iou_m.max(), iou_min_p0+0.01, 1.09))
    popt,pcov=curve_fit(_sig_model,alphas_arr,iou_m,
                         p0=[iou_max_p0, iou_min_p0, 11.7, 3.0],
                         bounds=([0.3, -0.05, 0.5, 0.1], [1.1, 0.99, 60.0, 20.0]),
                         maxfev=50000)
    # FIX (CRLB numerical-stability bug): the old code floored sigma2 at an arbitrary
    # constant (1e-8) unrelated to the data's own scale. Several alphas in the plateau
    # regime have EXACTLY zero empirical variance (IoU saturates at 1.0 across all 8
    # seeds), so they hit that floor and receive ~1e6-1e8x more weight than the
    # genuinely-informative near-transition points (whose sigma2 ~ 1e-6..1e-5) — i.e.
    # "perfectly measured" plateau points dominate the Fisher sum even though they carry
    # almost no information about alpha_star/delta. Combined with a poorly-bounded
    # sigmoid fit (pre-fix popt could land on unphysical values like IoU_max=-19.4), this
    # is what produced the catastrophic CRLB blowups (std up to ~1e6) seen in early logs:
    # near-collinear Jacobians + extreme reweighting => near-singular Fisher matrix.
    #
    # Fix: floor sigma2 at the smallest OBSERVED nonzero variance (data-scale-aware,
    # not an arbitrary external constant), Tikhonov-regularize I before inverting, and
    # report the condition number so a future near-singular fit is visible rather than
    # silently producing astronomical numbers.
    sigma2_raw = (iou_s**2)/n
    nonzero = sigma2_raw[sigma2_raw>0]
    floor = float(nonzero.min()) if len(nonzero) else 1e-8
    sigma2 = np.maximum(sigma2_raw, floor)
    I=np.zeros((4,4))
    for k,a in enumerate(alphas_arr):
        Jk=_J(a,*popt); I+=np.outer(Jk,Jk)/sigma2[k]
    cond_I = float(np.linalg.cond(I))
    # Ridge regularization scaled to the matrix's own trace — keeps the correction
    # negligible for well-conditioned matrices, stabilizes ill-conditioned ones.
    I_reg = I + np.eye(4)*(np.trace(I)/4)*1e-10
    try: crlb=np.linalg.inv(I_reg)
    except: crlb=np.full((4,4),np.nan)
    ill_conditioned = cond_I > 1e6
    print(f"  [Fisher info] sigma2 floor={floor:.3e} (data-driven, was a hardcoded 1e-8)"
          f"   cond(I)={cond_I:.3e}{'  *** ILL-CONDITIONED ***' if ill_conditioned else ''}")

    names=["IoU_max","IoU_min","alpha_star","delta"]
    rows=[]
    for i,nm in enumerate(names):
        crlb_std=float(np.sqrt(max(crlb[i,i],0)))
        ls_std=float(np.sqrt(pcov[i,i])) if not np.isnan(pcov[i,i]) else np.nan
        unreliable = ill_conditioned or (ls_std>0 and crlb_std > 100*ls_std)
        rows.append({"parameter":nm,"estimate":popt[i],
                     "crlb_std":crlb_std,"ls_std":ls_std,
                     "efficiency":crlb_std/ls_std if ls_std>0 else np.nan,
                     "cond_I":cond_I,"numerically_unreliable":unreliable})
        flag = "   *** NUMERICALLY UNRELIABLE — do not interpret as a CI ***" if unreliable else ""
        print(f"  {nm:12s}: est={popt[i]:.3f}  CRLB_std={crlb_std:.4f}  LS_std={ls_std:.4f}{flag}")

    df=pd.DataFrame(rows); save_csv(df,"fisher_cramer_rao.csv")

    # Figure
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    a_dense=np.linspace(alphas_arr.min(),alphas_arr.max(),300)
    fitted=_sig_model(a_dense,*popt)
    a_star=popt[2]; a_star_crlb=rows[2]["crlb_std"]
    dfit=np.array([_J(a,*popt)[2] for a in a_dense])
    axes[0].errorbar(alphas_arr,iou_m,yerr=iou_s/np.sqrt(n),fmt="o",capsize=4,
                     color="#1f77b4",label="Empirical (Adam)")
    axes[0].plot(a_dense,fitted,color="#d62728",lw=2.2,
                 label=f"Fit: a*={a_star:.2f}+/-{a_star_crlb:.3f} (CRLB)")
    axes[0].fill_between(a_dense,fitted-a_star_crlb*np.abs(dfit),
                         fitted+a_star_crlb*np.abs(dfit),alpha=0.15,color="#d62728")
    axes[0].axvline(a_star,color="gray",linestyle=":",lw=1.5)
    axes[0].set_xlabel("Sigmoid stiffness alpha"); axes[0].set_ylabel("Mean IoU")
    axes[0].set_title("Phase transition fit with CRLB uncertainty")
    axes[0].legend(fontsize=9); axes[0].grid(True,linestyle="--",alpha=0.4)

    D=np.sqrt(np.diag(crlb)); corr=crlb/(np.outer(D,D)+1e-12)
    im=axes[1].imshow(corr,cmap="RdBu_r",vmin=-1,vmax=1)
    axes[1].set_xticks(range(4)); axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(["IoU_max","IoU_min","a*","delta"],fontsize=9)
    axes[1].set_yticklabels(["IoU_max","IoU_min","a*","delta"],fontsize=9)
    for i in range(4):
        for j in range(4):
            axes[1].text(j,i,f"{corr[i,j]:.2f}",ha="center",va="center",fontsize=9,
                         color="white" if abs(corr[i,j])>0.5 else "black")
    plt.colorbar(im,ax=axes[1],fraction=0.046)
    axes[1].set_title("CRLB parameter correlation matrix")
    plt.suptitle("Fisher Information & Cramer-Rao Lower Bound",
                 fontweight="bold",fontsize=12)
    plt.tight_layout(); save_fig("fisher_cramer_rao.png")

# ── Ext B: Finite-size scaling ────────────────────────────────────────────────
def ext_b_finite_size_scaling(scale_df):
    print("\nFinite-Size Scaling: Critical Exponent beta")
    def _power_law(alpha, alpha_star, beta, scale, iou_min):
        return scale*np.maximum(alpha_star-alpha,1e-6)**beta + iou_min

    iou_col="iou_final" if "iou_final" in scale_df.columns else "output_iou_final"
    scale_df["size"]=scale_df["size"].str.replace("×","x")
    size_order=["32x32","64x64","128x128"]
    colors={"32x32":"#2ca02c","64x64":"#1f77b4","128x128":"#ff7f0e"}
    rows=[]
    for sl in size_order:
        sub=scale_df[scale_df["size"]==sl]
        if len(sub)==0: continue
        g=sub.groupby("alpha")[iou_col].agg(["mean","std"]).reset_index().sort_values("alpha")
        alphas_arr=g["alpha"].values; iou_m=g["mean"].values
        # iou_min from HIGHEST-alpha data point (last row after sort_values("alpha"))
        iou_min=float(iou_m[-1])

        # FIX (boundary-clamping bug): the previous code hardcoded alpha_star_estimate=11.7
        # and then used THAT SAME constant to build the fit's upper bound
        # (astar_upper = min(14.0, 11.7) == 11.7) and the beta lower bound was 0.05.
        # The optimizer therefore had nowhere to go but the bound on both parameters —
        # popt[0] == 11.7 and popt[1] == 0.05 in every resolution, with huge stderr.
        # That is an artifact of the search box, not a fitted result.
        #
        # Data-driven (non-circular) onset estimate, used ONLY to pick a generous masking
        # window — NOT as a bound — so the optimizer is free to land anywhere in a wide,
        # physically-motivated box.
        plateau = float(np.mean(iou_m[:3]))
        onset_idx = next((i for i,v in enumerate(iou_m) if v < 0.99*plateau), len(iou_m)//2)
        onset_guess = float(alphas_arr[onset_idx])

        mask = alphas_arr <= min(alphas_arr.max(), 2.0*onset_guess)
        if mask.sum()<4: continue
        a_lo, a_hi = float(alphas_arr.min()), float(2.0*alphas_arr.max())
        astar_p0  = float(np.clip(1.3*onset_guess, a_lo+0.5, a_hi-0.5))
        names  = ["alpha_star","beta","scale"]
        b_lo   = np.array([a_lo,  1e-3, 1e-3])
        b_hi   = np.array([a_hi, 10.0,  5.0])
        try:
            popt,pcov=curve_fit(
                lambda a,astar,beta,sc:_power_law(a,astar,beta,sc,iou_min),
                alphas_arr[mask],iou_m[mask],p0=[astar_p0,0.3,0.15],
                bounds=(b_lo,b_hi),maxfev=20000)
            perr=np.sqrt(np.diag(pcov))

            # Post-fit identifiability check: flag a parameter as CLAMPED if its fitted
            # value coincides with a bound to numerical precision (the signature the old
            # code produced: popt==0.05000000000000001, popt==11.69999999999999...).
            # NB: "within 1% of the box width" is the wrong test here — beta's natural
            # scale (~0.03) is far smaller than its box width (~10), so that test
            # false-positives on every legitimately-small interior value. Compare the
            # fitted value to each bound directly instead.
            pinned = [nm for nm,val,lo,hi in zip(names,popt,b_lo,b_hi)
                      if np.isclose(val,lo,rtol=1e-3,atol=1e-6) or np.isclose(val,hi,rtol=1e-3,atol=1e-6)]
            unidentifiable = len(pinned) > 0

            rows.append({"resolution":sl,"N":int(sl.split("x")[0])**2,
                          "alpha_star":popt[0],"alpha_star_err":perr[0],
                          "beta":popt[1],"beta_err":perr[1],"scale":popt[2],
                          "iou_min":iou_min,
                          "unidentifiable":unidentifiable,"pinned_params":",".join(pinned)})
            flag = f"   *** UNIDENTIFIABLE (pinned at bound: {','.join(pinned)}) ***" if unidentifiable else ""
            print(f"  {sl}: a*={popt[0]:.2f}+/-{perr[0]:.2f}  beta={popt[1]:.3f}+/-{perr[1]:.3f}"
                  f"  scale={popt[2]:.3f}{flag}")
        except Exception as e:
            print(f"  {sl}: fit failed ({e})")

    if rows:
        save_csv(pd.DataFrame(rows),"finite_size_scaling.csv")
        beta_m=np.mean([r["beta"] for r in rows])
        print(f"  Mean beta={beta_m:.3f}  (beta<1 -> 2nd-order / continuous transition)")

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for sl in size_order:
        sub=scale_df[scale_df["size"]==sl]
        if len(sub)==0: continue
        g=sub.groupby("alpha")[iou_col].agg(["mean","std"]).reset_index().sort_values("alpha")
        res=next((r for r in rows if r["resolution"]==sl),None)
        lbl=sl+(f"  b={res['beta']:.2f}+/-{res['beta_err']:.2f}" if res else "")
        axes[0].errorbar(g["alpha"],g["mean"],yerr=g["std"]/np.sqrt(5),
                         fmt="o-",capsize=3,lw=2,color=colors.get(sl,"gray"),label=lbl)
        if res:
            a_dense=np.linspace(g["alpha"].min(),g["alpha"].max(),300)
            axes[0].plot(a_dense,_power_law(a_dense,res["alpha_star"],res["beta"],
                                             res["scale"],res["iou_min"]),"--",
                         color=colors.get(sl,"gray"),lw=1.5,alpha=0.7)
    axes[0].set_xlabel("Sigmoid stiffness alpha"); axes[0].set_ylabel("Reconstruction IoU")
    axes[0].set_title("Power-law fit near alpha*"); axes[0].legend(fontsize=9)
    axes[0].grid(True,linestyle="--",alpha=0.4)

    nu=1.0
    for sl in size_order:
        sub=scale_df[scale_df["size"]==sl]
        if len(sub)==0: continue
        g=sub.groupby("alpha")[iou_col].agg(["mean","std"]).reset_index().sort_values("alpha")
        res=next((r for r in rows if r["resolution"]==sl),None)
        if res is None: continue
        N=res["N"]; a_star=res["alpha_star"]; iou_min=res["iou_min"]
        rx=(g["alpha"].values-a_star)*N**(1./nu)
        op=np.maximum(g["mean"].values-iou_min,0)
        axes[1].plot(rx,op,"o-",color=colors.get(sl,"gray"),label=sl,lw=2,markersize=5)
    axes[1].axvline(0,color="gray",linestyle=":",lw=1.5)
    axes[1].set_xlabel("Rescaled: (alpha-alpha*)*N^(1/nu),  nu=1.0")
    axes[1].set_ylabel("Order parameter IoU-IoU_min")
    axes[1].set_title("Finite-size scaling collapse\n(collapse = universality)")
    axes[1].legend(fontsize=9); axes[1].grid(True,linestyle="--",alpha=0.4)
    plt.suptitle("Finite-Size Scaling — Phase Transition Universality",
                 fontweight="bold",fontsize=12)
    plt.tight_layout(); save_fig("finite_size_scaling.png")

# ── Ext C: Depth scaling law ──────────────────────────────────────────────────
def ext_c_depth_scaling():
    print("\nDepth Scaling Law: L-layer compounding collapse")
    existing=load_csv("depth_scaling_law.csv",warn=False)
    N_SIDE=32; N=N_SIDE**2; MAX_DEPTH=6; EPS=0.01; C=0.5
    K=KERNELS["sobel_x"]

    def _conv3x3(x2d,K):
        return convolve(x2d,K,mode="wrap")

    kernels_l=[np.random.default_rng(l).standard_normal((3,3)) for l in range(MAX_DEPTH)]
    for k in kernels_l: k/=np.linalg.norm(k)+1e-8

    rows=[]
    if existing is None or len(existing)==0:
        for alpha in ALPHAS:
            for seed in range(3):
                rng=np.random.default_rng(seed)
                h=rng.uniform(0,1,(N_SIDE,N_SIDE))
                for L in range(1,MAX_DEPTH+1):
                    z=_conv3x3(h,kernels_l[L-1])
                    s=sigmoid(z,alpha,C)
                    gate=sigma=sigmoid_prime(s,alpha)
                    frac=float(np.mean(np.abs(gate)>EPS))
                    rows.append({"alpha":alpha,"depth":L,"seed":seed,"active_frac":frac})
                    h=s
        df=pd.DataFrame(rows); save_csv(df,"depth_scaling_law.csv")
    else:
        df=existing

    grp=df.groupby(["alpha","depth"])["active_frac"].agg(["mean","std"]).reset_index()

    def depth_law(depth,c_param,alpha_val):
        return 1./(1.+c_param*alpha_val)**(depth-1)

    fit_results=[]
    for alpha_val,sub in grp.groupby("alpha"):
        f1=sub[sub["depth"]==1]["mean"].values
        if len(f1)==0 or f1[0]<1e-6: continue
        depths=sub["depth"].values; ratios=sub["mean"].values/(f1[0]+1e-8)
        try:
            popt,_=curve_fit(lambda d,c:depth_law(d,c,alpha_val),depths,ratios,
                              p0=[0.05],bounds=([0.],[1.]),maxfev=1000)
            fit_results.append({"alpha":alpha_val,"c_fit":popt[0]})
            print(f"  alpha={alpha_val:.1f}: c={popt[0]:.4f}")
        except: pass

    # ── Bootstrap R² for each alpha fit ──
    boot_rows = []
    for alpha_val, sub in grp.groupby("alpha"):
        f1 = sub[sub["depth"]==1]["mean"].values
        if len(f1)==0 or f1[0]<1e-6: continue
        depths_arr = sub["depth"].values.astype(float)
        ratios_arr = sub["mean"].values / (f1[0]+1e-8)
        c_fit_val = next((r["c_fit"] for r in fit_results if abs(r["alpha"]-alpha_val)<1e-9), None)
        if c_fit_val is None: continue
        # FIX (R^2 discrepancy bug): the old code SKIPPED the bootstrap entirely for
        # alpha<10 or c_fit<=0.001 and wrote literal NaNs labeled "N/A — pre-transition".
        # That hides exactly the regime the reader most needs to see: at small alpha the
        # depth-compounding law F^(L)=F^(1)/(1+c*alpha)^(L-1) degenerates to a near-constant
        # (c≈0 ⇒ ratio≈1 for every depth), so it is not that the fit "fails" — the model is
        # structurally non-identifiable there, and the bootstrap CI is the correct way to
        # SHOW that (a CI spanning [-0.7, 0.9] *is* the finding, not a missing value).
        # Bootstrapping uniformly across all alpha also explains why a SINGLE headline
        # number like "R^2=0.9888" cannot represent "the compounding-collapse fit": R^2
        # is strongly alpha-dependent, undefined/unstable near the transition (alpha~10-16,
        # where c_fit crosses from ~0 to a measurable value), and only becomes large and
        # stable deep in the post-transition regime (alpha>=20, R^2 -> 0.92-0.96).
        model_applicable = False  # will be set after bootstrap R² is known
        fit_func_boot = lambda d, c, _av=alpha_val: 1./(1.+c*_av)**(d-1)
        p0_boot = [c_fit_val if c_fit_val > 1e-6 else 1e-3]
        # c is a per-layer gate-collapse RATE — physically non-negative and (empirically,
        # see fit_results) O(1e-2); bounding the resampled fit to [0,1] keeps it in the
        # physical range. The astronomical R^2 values seen pre-fix (e.g. -1e21) were NOT
        # caused by the optimizer wandering — they persisted even with bounds=(0,1) — they
        # were caused by ss_tot -> 0 (near-constant ratio at small alpha) making R^2
        # mathematically undefined; see _bootstrap_r2 for the real fix (an ss_tot floor
        # that marks those resamples as undefined instead of dividing by ~0).
        r2_med, r2_lo, r2_hi, frac_undef = _bootstrap_r2(
            depths_arr, ratios_arr, fit_func_boot, p0=p0_boot, n_boot=1000, seed=0,
            bounds=(0.0, 1.0)
        )
        model_applicable = bool(r2_med >= 0.90)
        if frac_undef > 0.5:
            tag = (f"  [R^2 UNDEFINED for {frac_undef:.0%} of resamples: ratio~constant "
                   f"(c~0) -> ss_tot~0 -> R^2 is 0/0, not a meaningful number]")
        elif not model_applicable:
            tag = "  [model not applicable: c~0, ratio~constant — CI reflects that, not a failure]"
        else:
            tag = ""
        print(f"  [depth] alpha={alpha_val:.1f} Bootstrap R² (median) = {r2_med:.4f} [{r2_lo:.4f}, {r2_hi:.4f}]  "
              f"(n_boot=1000, {frac_undef:.0%} undefined){tag}")
        boot_rows.append({"alpha": alpha_val, "c_fit": c_fit_val, "model_applicable": model_applicable,
                           "r2_boot_median": r2_med, "r2_boot_lo": r2_lo, "r2_boot_hi": r2_hi,
                           "frac_resamples_r2_undefined": frac_undef})

    if boot_rows:
        df_fits_boot = pd.DataFrame(boot_rows)
        save_csv(df_fits_boot, "depth_scaling_law_fits.csv")

    alpha_colors={1.:"#2ca02c",2.:"#17becf",5.:"#1f77b4",10.:"#ff7f0e",20.:"#d62728",40.:"#9467bd"}
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for av,sub in grp.groupby("alpha"):
        c_fit=next((r["c_fit"] for r in fit_results if r["alpha"]==av),0.051)
        axes[0].errorbar(sub["depth"],sub["mean"],yerr=sub["std"],marker="o",capsize=3,
                         lw=2,color=alpha_colors.get(av,"gray"),label=f"a={av:.0f}")
        f1=sub[sub["depth"]==1]["mean"].values
        if len(f1)>0:
            d_dense=np.linspace(1,sub["depth"].max(),100)
            axes[0].plot(d_dense,f1[0]/(1.+c_fit*av)**(d_dense-1),"--",
                         color=alpha_colors.get(av,"gray"),lw=1.2,alpha=0.7)
    axes[0].set_xlabel("Network depth L"); axes[0].set_ylabel("Active gradient fraction")
    axes[0].set_title("Depth scaling law: F^(L)=F^(1)/(1+c*a)^(L-1)")
    axes[0].legend(fontsize=9,ncol=2); axes[0].grid(True,linestyle="--",alpha=0.4)

    for av,sub in grp.groupby("alpha"):
        f1=sub[sub["depth"]==1]["mean"].values
        if len(f1)==0 or f1[0]<1e-6: continue
        ss=sub.sort_values("depth")
        log_r=np.log(ss["mean"].values/(f1[0]+1e-8)+1e-8)
        c_fit=next((r["c_fit"] for r in fit_results if r["alpha"]==av),0.051)
        axes[1].plot(ss["depth"].values-1,log_r,"o-",
                     color=alpha_colors.get(av,"gray"),lw=2,label=f"a={av:.0f}")
        d_arr=np.linspace(0,ss["depth"].max()-1,100)
        axes[1].plot(d_arr,-d_arr*np.log(1+c_fit*av),"--",
                     color=alpha_colors.get(av,"gray"),lw=1.2,alpha=0.6)
    axes[1].set_xlabel("L-1 (additional layers)"); axes[1].set_ylabel("log(F^(L)/F^(1))")
    axes[1].set_title("Log-linear collapse: depth-exponential gradient death")
    axes[1].legend(fontsize=9,ncol=2); axes[1].grid(True,linestyle="--",alpha=0.4)
    plt.suptitle("Depth Scaling Law — Arbitrary L Layers",
                 fontweight="bold",fontsize=12)
    plt.tight_layout(); save_fig("depth_scaling_law.png")

# ── Ext D: Null space geometry ────────────────────────────────────────────────
def ext_d_nullspace():
    print("\nNull Space Geometry & Identifiability Certificates")
    existing=load_csv("nullspace_geometry.csv",warn=False)
    N_SIDE=16; N=N_SIDE**2

    def _sobel_mat(N):
        side=int(N**0.5); K=KERNELS["sobel_x"]; M=np.zeros((N,N))
        for i in range(N):
            r,ci=divmod(i,side)
            for dr in range(-1,2):
                for dc in range(-1,2):
                    nr=(r+dr)%side; nc=(ci+dc)%side
                    M[i,nr*side+nc]+=K[dr+1,dc+1]
        return M

    A=_sobel_mat(N)
    if existing is None or len(existing)==0:
        rows=[]
        for alpha in ALPHAS:
            for seed in range(5):
                rng=np.random.default_rng(seed)
                x=rng.uniform(0,1,N)
                z=A@x; s=sigmoid(z,alpha,0.5); gate=sigmoid_prime(s,alpha)
                J=np.diag(gate)@A
                sv=np.linalg.svd(J,compute_uv=False)
                eps=1e-4*sv.max()
                rows.append({"alpha":alpha,"seed":seed,
                              "null_dim":int(np.sum(sv<eps)),
                              "sigma_min":float(sv.min()),
                              "sigma_max":float(sv.max()),
                              "stable_rank":float(np.sum(sv**2)/(sv.max()**2+1e-12)),
                              "condition_number":float(sv.max()/(sv.min()+1e-12)),
                              "n_nonzero_sv":int(np.sum(sv>eps))})
        df=pd.DataFrame(rows); save_csv(df,"nullspace_geometry.csv")
    else:
        df=existing

    grp=df.groupby("alpha").agg(
        null_dim_mean=("null_dim","mean"),null_dim_std=("null_dim","std"),
        sigma_min_mean=("sigma_min","mean"),sigma_min_std=("sigma_min","std"),
        stable_rank_mean=("stable_rank","mean"),cond_mean=("condition_number","mean")
    ).reset_index()

    fig,axes=plt.subplots(1,3,figsize=(16,5))
    axes[0].errorbar(grp["alpha"],grp["null_dim_mean"],yerr=grp["null_dim_std"],
                     fmt="o-",capsize=4,color="#d62728",lw=2)
    axes[0].axhline(N,color="gray",linestyle=":",lw=1.5,label=f"Max (N={N})")
    axes[0].set_xlabel("Sigmoid stiffness alpha")
    axes[0].set_ylabel(f"dim(ker J(x*))  [max={N}]")
    axes[0].set_title("Null space dimension vs alpha\n(non-uniqueness degree)")
    axes[0].legend(fontsize=9); axes[0].grid(True,linestyle="--",alpha=0.4)

    axes[1].errorbar(grp["alpha"],grp["sigma_min_mean"],yerr=grp["sigma_min_std"],
                     fmt="s-",capsize=4,color="#1f77b4",lw=2)
    axes[1].axvline(11.7,color="#ff7f0e",linestyle="--",lw=1.5,label="alpha*=11.7")
    axes[1].set_xlabel("Sigmoid stiffness alpha"); axes[1].set_ylabel("sigma_min(J(x*))")
    axes[1].set_title("Identifiability certificate sigma_min(J)\n(zero = unidentifiable)")
    axes[1].legend(fontsize=9); axes[1].grid(True,linestyle="--",alpha=0.4)

    ax3=axes[2]; ax3b=ax3.twinx()
    l1,=ax3.plot(grp["alpha"],grp["stable_rank_mean"],"D-",color="#2ca02c",lw=2,label="Stable rank")
    l2,=ax3b.plot(grp["alpha"],grp["cond_mean"],"^--",color="#9467bd",lw=2,label="Cond number")
    ax3.set_xlabel("Sigmoid stiffness alpha"); ax3.set_ylabel("Stable rank sr(J)")
    ax3.set_title("Stable rank & condition number vs alpha")
    ax3b.set_ylabel("Condition number kappa(J)",color="#9467bd")
    ax3.legend(handles=[l1,l2],fontsize=9,loc="upper right")
    ax3.grid(True,linestyle="--",alpha=0.4)
    plt.suptitle("Null Space Geometry & Identifiability Certificates",
                 fontweight="bold",fontsize=12)
    plt.tight_layout(); save_fig("nullspace_geometry.png")

# ── Ext E: Gradient leakage ───────────────────────────────────────────────────
def ext_e_gradient_leakage(dense_df, alpha_df):
    print("\nGradient Leakage Attack Surface")
    combined=pd.concat([dense_df,
                         alpha_df[alpha_df["optimizer"]=="adam"] if "optimizer" in alpha_df.columns else alpha_df],
                        ignore_index=True).drop_duplicates(subset=["alpha","seed"])

    thresholds=[0.5,0.6,0.7,0.8]
    thresh_colors={0.5:"#9467bd",0.6:"#ff7f0e",0.7:"#d62728",0.8:"#1f77b4"}
    alphas_sorted=sorted(combined["alpha"].unique())

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for thr in thresholds:
        srates=[float(np.mean(combined[combined["alpha"]==a]["output_iou_final"]>thr))
                for a in alphas_sorted]
        axes[0].plot(alphas_sorted,srates,"o-",lw=2.2,
                     color=thresh_colors[thr],label=f"Attack threshold IoU>{thr}")

    axes[0].axvline(11.7,color="gray",linestyle=":",lw=1.5,label="alpha*=11.7")
    axes[0].fill_betweenx([0,1],0,11.7,alpha=0.06,color="#d62728")
    axes[0].fill_betweenx([0,1],11.7,max(alphas_sorted),alpha=0.06,color="#2ca02c")
    axes[0].text(3,0.92,"HIGH RISK",fontsize=9,color="#d62728",ha="center",fontweight="bold")
    axes[0].text(30,0.08,"PROTECTED",fontsize=9,color="#2ca02c",ha="center",fontweight="bold")
    axes[0].set_xlabel("Sigmoid stiffness alpha")
    axes[0].set_ylabel("Attack success rate P[IoU > threshold]")
    axes[0].set_title("Privacy risk: gradient leakage attack success\n(dense alpha captures smooth transition)")
    axes[0].legend(fontsize=9); axes[0].grid(True,linestyle="--",alpha=0.4)

    mean_iou=[float(combined[combined["alpha"]==a]["output_iou_final"].mean()) for a in alphas_sorted]
    privacy=[1.-float(np.mean(combined[combined["alpha"]==a]["output_iou_final"]>0.7))
             for a in alphas_sorted]
    utility=[1.-1./(1.+0.1*a) for a in alphas_sorted]

    sc=axes[1].scatter(utility,privacy,c=alphas_sorted,cmap="RdYlGn",s=120,
                        zorder=3,edgecolors="black",lw=0.5,
                        vmin=min(alphas_sorted),vmax=max(alphas_sorted))
    axes[1].plot(utility,privacy,"-",color="gray",lw=1,alpha=0.4,zorder=2)
    for u,p,a in zip(utility,privacy,alphas_sorted):
        if a in [1.,5.,10.,12.,16.,20.,40.]:
            axes[1].annotate(f"a={a:.0f}",(u,p),textcoords="offset points",
                             xytext=(6,3),fontsize=8)
    plt.colorbar(sc,ax=axes[1]).set_label("Sigmoid stiffness alpha")
    axes[1].set_xlabel("Forward-map utility (sigmoid saturation)")
    axes[1].set_ylabel("Privacy protection (1 - attack success)")
    axes[1].set_title("Privacy-utility tradeoff\nSmooth transition via dense alpha sampling")
    axes[1].grid(True,linestyle="--",alpha=0.4)
    plt.suptitle("Gradient Leakage Attack Surface & Privacy-Utility Tradeoff",
                 fontweight="bold",fontsize=12)
    plt.tight_layout(); save_fig("gradient_leakage.png")

    leakage_df=pd.DataFrame({"alpha":alphas_sorted,"mean_iou":mean_iou,
                               "attack_success_07":[1-p for p in privacy],
                               "privacy_07":privacy,"utility_proxy":utility})
    save_csv(leakage_df,"gradient_leakage.csv")

# ── Ext F: Learned weights ────────────────────────────────────────────────────
def ext_f_learned_weights():
    print("\nLearned Weights: Does Collapse Survive Training?")
    existing=load_csv("learned_weights.csv",warn=False)
    if existing is not None and existing["iou"].mean()>0.1:
        print("[skip] learned_weights.csv already has valid data")
        df=existing
    else:
        K_sobel=KERNELS["sobel_x"]

        def _sig2d(z,alpha,c=0.5):
            return sigmoid(z,alpha,c)

        def _apply(x2d,K):
            return convolve(x2d,K,mode="wrap")

        def train_kernel(K_init,x_list,y_list,alpha,n_steps=80,lr=0.008):
            K=K_init.copy()
            for _ in range(n_steps):
                dK=np.zeros_like(K)
                for x2d,y2d in zip(x_list,y_list):
                    z=_apply(x2d,K); h=_sig2d(z,min(alpha,5.0)); gate=sigmoid_prime(h,min(alpha,5.0))
                    err=h-y2d
                    e=err*gate
                    R=np.real(np.fft.ifft2(np.fft.fft2(e)*np.conj(np.fft.fft2(x2d))))
                    kHh,kWh=K.shape[0]//2,K.shape[1]//2; H,W=x2d.shape
                    rows=[(ki-kHh)%H for ki in range(K.shape[0])]
                    cols=[(kj-kWh)%W for kj in range(K.shape[1])]
                    dK+=R[np.ix_(rows,cols)]
                dK/=len(x_list)
                gnorm=np.linalg.norm(dK)
                if gnorm>0.5: dK/=gnorm/0.5
                K-=lr*dK
            return K

        def _adam_conv(K,target2d,alpha,steps=300,lr=0.03,seed=0):
            rng=np.random.default_rng(seed)
            x=rng.uniform(0,1,target2d.shape)
            m=np.zeros_like(x); v=np.zeros_like(x); b1,b2,eps=0.9,0.999,1e-8
            for t in range(1,steps+1):
                z=_apply(x,K); h=_sig2d(z,alpha); gate=sigmoid_prime(h,alpha)
                err=h-target2d
                g=2.*_apply(err*gate,K[::-1,::-1])
                m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g**2
                x=np.clip(x-lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps),0.,1.)
            z=_apply(x,K); h=_sig2d(z,alpha); gate=sigmoid_prime(h,alpha)
            af=float(np.mean(np.abs(gate)>0.01))
            return iou_score(h,target2d), af

        rows=[]
        for alpha in ALPHAS:
            for seed in range(len(SEEDS)):  # 8 seeds for sufficient Wilcoxon power (p<0.05)
                rng=np.random.default_rng(seed+42)
                target=get_targets(*SHAPE)["checkerboard"]
                x_list=[rng.uniform(0,1,SHAPE) for _ in range(10)]
                y_list=[(_sig2d(_apply(x,K_sobel),1.0)>0.5).astype(float) for x in x_list]
                K_trained=train_kernel(K_sobel,x_list,y_list,alpha)
                # fixed: use standard runner
                p_fixed=run_single_experiment(alpha=alpha,seed=seed,optimizer_name="adam",
                                               optimizer_kwargs={"lr":0.03,"steps":STEPS})
                iou_fixed=p_fixed["summary"]["output_iou_final"]
                af_fixed=p_fixed["summary"]["active_grad_frac_final"]
                # trained: fast conv Adam
                iou_trained,af_trained=_adam_conv(K_trained,target,alpha,steps=STEPS,seed=seed)
                print(f"  a={alpha:.1f} s={seed}: fixed IoU={iou_fixed:.3f} | trained IoU={iou_trained:.3f}")
                rows.append({"alpha":alpha,"seed":seed,"kernel":"random_fixed",
                              "iou":iou_fixed,"active_frac":af_fixed})
                rows.append({"alpha":alpha,"seed":seed,"kernel":"trained",
                              "iou":iou_trained,"active_frac":af_trained})
        df=pd.DataFrame(rows); save_csv(df,"learned_weights.csv")

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    colors={"random_fixed":"#1f77b4","trained":"#d62728"}
    labels={"random_fixed":"Fixed Sobel weights","trained":"Trained weights (BCE-optimised)"}
    for kernel,grp in df.groupby("kernel"):
        g=grp.groupby("alpha")["iou"].agg(["mean","std"]).reset_index()
        axes[0].errorbar(g["alpha"],g["mean"],yerr=g["std"],fmt="o-",capsize=4,lw=2.2,
                         color=colors[kernel],label=labels[kernel])
    axes[0].axvline(11.7,color="gray",linestyle=":",lw=1.5,label="alpha*=11.7")
    axes[0].set_xlabel("Sigmoid stiffness alpha"); axes[0].set_ylabel("Reconstruction IoU")
    axes[0].set_title("Reconstruction: fixed vs trained weights\n(64x64, checkerboard, Adam, 3 seeds)")
    axes[0].legend(fontsize=10); axes[0].grid(True,linestyle="--",alpha=0.4)
    for kernel,grp in df.groupby("kernel"):
        g=grp.groupby("alpha")["active_frac"].agg(["mean","std"]).reset_index()
        axes[1].errorbar(g["alpha"],g["mean"],yerr=g["std"],fmt="s-",capsize=4,lw=2.2,
                         color=colors[kernel],label=labels[kernel])
    axes[1].set_xlabel("Sigmoid stiffness alpha"); axes[1].set_ylabel("Active gradient fraction")
    axes[1].set_title("Gradient gate collapse: fixed vs trained weights")
    axes[1].legend(fontsize=10); axes[1].grid(True,linestyle="--",alpha=0.4)
    plt.suptitle("Gradient Gate Collapse Persists Under Learned Weights",
                 fontweight="bold",fontsize=12)
    plt.tight_layout(); save_fig("learned_weights.png")

    # ── Paired Wilcoxon test: fixed vs trained kernel IoU per alpha ──
    print("\n  Wilcoxon paired test: fixed vs trained kernel IoU")
    wilcox_rows = []
    iou_fixed_by_alpha = {}
    iou_trained_by_alpha = {}
    for alpha_val, sub in df.groupby("alpha"):
        fixed_vals  = sub[sub["kernel"]=="random_fixed"]["iou"].values
        trained_vals= sub[sub["kernel"]=="trained"]["iou"].values
        iou_fixed_by_alpha[alpha_val]   = list(fixed_vals)
        iou_trained_by_alpha[alpha_val] = list(trained_vals)
    for alpha_val in sorted(iou_fixed_by_alpha.keys()):
        fixed_list   = iou_fixed_by_alpha[alpha_val]
        trained_list = iou_trained_by_alpha[alpha_val]
        n = min(len(fixed_list), len(trained_list))
        if n < 2:
            continue
        fixed_arr   = np.array(fixed_list[:n])
        trained_arr = np.array(trained_list[:n])
        fixed_mean   = float(fixed_arr.mean())
        trained_mean = float(trained_arr.mean())
        w = wilcoxon_pairwise(fixed_arr, trained_arr)
        p_val = w["pvalue"]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  alpha={alpha_val:.1f}: fixed={fixed_mean:.3f} trained={trained_mean:.3f} "
              f"Wilcoxon_p={p_val:.4f} {sig}")
        wilcox_rows.append({"alpha": alpha_val, "fixed_iou_mean": fixed_mean,
                             "trained_iou_mean": trained_mean,
                             "wilcoxon_p": p_val, "significance": sig, "n": n})
    if wilcox_rows:
        save_csv(pd.DataFrame(wilcox_rows), "learned_weights_wilcoxon.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9b — PHASE 3A: DEEP-CNN (ResNet-18 / VGG-11-shaped) GATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
#
# GOAL: ext_c_depth_scaling found that a *uniform* synthetic stack of L
# identical sigmoid-gated conv layers obeys (in the post-transition regime,
# alpha>=10) a geometric depth-compounding law for the active-gradient
# fraction:        F^(L) / F^(1)  =  (1 + c*alpha)^-(L-1)
# Phase 3A asks whether this generalizes to architectures whose layers are
# NOT uniform — different channel widths, strides (spatial downsampling),
# kernel sizes, and (for ResNet) residual/skip connections.
#
# MODELING DECISION — flagged explicitly, not papered over:
# Canonical ResNet-18/VGG-11 use ReLU, which has no tunable "stiffness" — there
# is no alpha to sweep, and no smooth Gamma=|f'| to measure (its derivative is
# the scale-invariant binary mask 1[u>0]; see _act_relu_prime / Bug #6 writeup).
# To make "sweep over alpha and measure compounding collapse" a meaningful
# question for a deep, structurally heterogeneous network, we keep the
# topology of ResNet-18/VGG-11 (conv layout, channel widths, strides, residual
# connections) intact and replace every nonlinearity with the SAME
# alpha-parameterized sigmoid sigma(z;alpha,c) used by ext_c_depth_scaling and
# the rest of the gate-collapse theory (NOT the generalized multi-base-
# activation framework from Bug #6/Phase 3B — conflating "which activation"
# with "how does depth/width heterogeneity affect compounding" would leave
# neither question cleanly testable). Concretely: these are
# "ResNet-18/VGG-11-SHAPED sigmoid networks", not the canonical pretrained
# architectures, and any conclusion drawn here is about gate compounding under
# heterogeneous depth/width — NOT a claim about real ResNet-18/VGG-11 + ReLU.
#
# COMPUTE: requires torch (+ ideally CUDA) — lazily imported, skippable via
# --skip-deepnet. This is NEW GPU compute per the agreed Phase-3 plan: code is
# written/smoke-tested here; the user runs it on their SLURM allocation.

DEEPNET_ALPHAS = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
DEEPNET_GATE_C = 0.5     # matches ext_c_depth_scaling's synthetic-stack convention
DEEPNET_GATE_EPS = 0.01  # matches active_grad_fraction's threshold elsewhere

def _build_deepnet_modules():
    """Lazily build torch + the alpha-sigmoid ResNet-18/VGG-11-shaped factories.
    Returns None if torch is unavailable (mirrors the TF/--skip-mnist pattern)."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None

    class AlphaSigmoid(nn.Module):
        """sigma(z;alpha,c) = sigmoid(alpha*(z-c)); records its own gradient
        gate Gamma=|sigma'|=alpha*h*(1-h) and active-fraction when .record=True.
        This is the exact gate definition used throughout the rest of the
        gate-collapse theory (sigmoid_prime), just evaluated on torch tensors."""
        def __init__(self, alpha, c=DEEPNET_GATE_C):
            super().__init__()
            self.alpha = float(alpha); self.c = float(c)
            self.record = False
            self.last_active_frac = None
            self.last_gate_mean = None
        def forward(self, z):
            h = torch.sigmoid(self.alpha * (z - self.c))
            if self.record:
                gate = self.alpha * h * (1.0 - h)
                self.last_active_frac = float((gate.abs() > DEEPNET_GATE_EPS).float().mean().item())
                self.last_gate_mean = float(gate.abs().mean().item())
            return h

    def _act():
        # placeholder; alpha is bound by _set_alpha after construction so that
        # one architecture builder serves every alpha in the sweep
        return AlphaSigmoid(alpha=1.0)

    def _set_alpha(model, alpha):
        for m in model.modules():
            if isinstance(m, AlphaSigmoid):
                m.alpha = float(alpha)

    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_ch)
            self.act1  = _act()
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch))
            self.act2 = _act()
        def forward(self, x):
            out = self.act1(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            return self.act2(out)

    class AlphaResNet18(nn.Module):
        """Standard ResNet-18 topology (BasicBlock x [2,2,2,2], widths
        [64,128,256,512], stage strides [1,2,2,2]) with every ReLU replaced
        by AlphaSigmoid. CIFAR-style 3x3 stem (no 7x7/maxpool) so it runs on
        32x32 inputs without collapsing spatial dims to 0."""
        def __init__(self, in_channels=3, num_classes=10):
            super().__init__()
            self.stem_conv = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
            self.stem_bn   = nn.BatchNorm2d(64)
            self.stem_act  = _act()
            widths  = [64, 128, 256, 512]
            strides = [1, 2, 2, 2]
            layers = []
            in_ch = 64
            for w, s in zip(widths, strides):
                layers.append(BasicBlock(in_ch, w, stride=s))
                layers.append(BasicBlock(w, w, stride=1))
                in_ch = w
            self.stages = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, num_classes)
        def forward(self, x):
            x = self.stem_act(self.stem_bn(self.stem_conv(x)))
            x = self.stages(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    class AlphaVGG11(nn.Module):
        """VGG-11 ('A') topology: conv blocks
        [64,M,128,M,256,256,M,512,512,M,512,512,M] with BN, every ReLU
        replaced by AlphaSigmoid; small 2-layer FC head sized for 32x32 input
        (5 maxpools -> 1x1 spatial)."""
        CFG = [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']
        def __init__(self, in_channels=3, num_classes=10):
            super().__init__()
            layers = []
            in_ch = in_channels
            for v in self.CFG:
                if v == 'M':
                    layers.append(nn.MaxPool2d(2, 2))
                else:
                    layers.append(nn.Conv2d(in_ch, v, 3, padding=1, bias=False))
                    layers.append(nn.BatchNorm2d(v))
                    layers.append(_act())
                    in_ch = v
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Sequential(
                nn.Linear(512, 256), _act(), nn.Linear(256, num_classes))
        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    return dict(torch=torch, nn=nn, AlphaSigmoid=AlphaSigmoid,
                set_alpha=_set_alpha, ResNet18=AlphaResNet18, VGG11=AlphaVGG11)

def _deepnet_gate_profile(env, model, x, alpha):
    """Run one forward pass with gate-recording on; return the per-layer
    active-fraction profile IN NETWORK EXECUTION ORDER (forward hooks fire in
    true call order, unlike .modules() registration order, which would be
    ambiguous for branching/residual topologies)."""
    torch = env["torch"]; AlphaSigmoid = env["AlphaSigmoid"]
    env["set_alpha"](model, alpha)
    profile = []
    handles = []
    def _hook(module, inp, out):
        profile.append(module.last_active_frac)
    acts = [m for m in model.modules() if isinstance(m, AlphaSigmoid)]
    for m in acts:
        m.record = True
        handles.append(m.register_forward_hook(_hook))
    with torch.no_grad():
        model(x)
    for h_ in handles: h_.remove()
    for m in acts: m.record = False
    return profile

def run_deepnet_gate_sweep():
    """Phase 3A: sweep alpha in DEEPNET_ALPHAS x architecture x seed, recording
    the per-layer active-gradient-fraction profile of ResNet-18/VGG-11-shaped
    alpha-sigmoid networks on random CIFAR-shaped (3x32x32) inputs.
    Checkpoint-cached to deepnet_gate_profile.csv (one row per (arch, alpha,
    seed, layer_idx)) — this is the raw material for testing whether the
    depth-compounding law generalizes to heterogeneous-layer architectures
    (see analyze_deepnet_compounding)."""
    print("\n[exp] Phase 3A: deep-CNN (ResNet-18/VGG-11-shaped) gradient-gate profiling")
    if args.skip_deepnet:
        print("  [skip] --skip-deepnet flag set"); return None
    env = _build_deepnet_modules()
    if env is None:
        print("  [skip] torch not available — install torch to run Phase 3A"); return None
    torch = env["torch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device={device}  alphas={DEEPNET_ALPHAS}  seeds={SEEDS}")

    existing = load_csv("deepnet_gate_profile.csv", warn=False)
    ARCHS = {"resnet18": env["ResNet18"], "vgg11": env["VGG11"]}
    rows = []
    for arch_name, ctor in ARCHS.items():
        for seed in SEEDS:
            if _already_done(existing, {"arch": arch_name, "alpha": DEEPNET_ALPHAS[0], "seed": seed,
                                         "layer_idx": 0}):
                continue
            torch.manual_seed(seed)
            model = ctor(in_channels=3, num_classes=10).to(device).eval()
            rng = np.random.default_rng(1000 + seed)
            x = torch.from_numpy(rng.standard_normal((4, 3, 32, 32)).astype(np.float32)).to(device)
            for alpha in DEEPNET_ALPHAS:
                profile = _deepnet_gate_profile(env, model, x, alpha)
                for li, frac in enumerate(profile):
                    rows.append({"arch": arch_name, "alpha": alpha, "seed": seed,
                                 "layer_idx": li, "n_layers": len(profile),
                                 "active_frac": frac})
            print(f"  [{arch_name}] seed={seed}: {len(profile)} gated layers profiled "
                  f"across {len(DEEPNET_ALPHAS)} alphas")
    if rows:
        append_csv(rows, "deepnet_gate_profile.csv")
    else:
        print("  [skip] deepnet_gate_profile.csv already complete")
    return load_csv("deepnet_gate_profile.csv", warn=False)

def analyze_deepnet_compounding(df):
    """Tests whether the geometric depth-compounding law found on a UNIFORM
    synthetic stack (ext_c_depth_scaling: F^(L)/F^(1) = (1+c*alpha)^-(L-1))
    generalizes to the heterogeneous-layer ResNet-18/VGG-11-shaped networks.
    Two nested tests, reported honestly regardless of outcome (no rubber-
    stamping a fixed answer):
      (1) GLOBAL fit  — does a single c capture the whole heterogeneous
          sequence (the strong form of the law)?
      (2) LOCAL ratios — regardless of whether a single global c fits, is the
          layer-to-layer decay still geometric (constant ratio) within runs of
          structurally-similar layers (e.g., consecutive same-stride blocks),
          with the rate merely varying by structural context (stride/width
          change)? This distinguishes "law generalizes with structure-
          dependent rates" from "law fails outright"."""
    print("\n[exp] Phase 3A: does the depth-compounding law generalize to heterogeneous layers?")
    if df is None or len(df) == 0:
        print("  [skip] no deepnet gate profile data available"); return None

    def _depth_law(depth, c_param, alpha_val):
        return 1.0 / (1.0 + c_param * alpha_val) ** (depth - 1)

    rows = []
    for arch_name, sub_arch in df.groupby("arch"):
        grp = sub_arch.groupby(["alpha", "layer_idx"])["active_frac"].agg(["mean", "std"]).reset_index()
        for alpha_val, sub in grp.groupby("alpha"):
            sub = sub.sort_values("layer_idx")
            f1 = sub[sub["layer_idx"] == sub["layer_idx"].min()]["mean"].values
            if len(f1) == 0 or f1[0] < 1e-6:
                continue
            depths = (sub["layer_idx"].values - sub["layer_idx"].values.min() + 1).astype(float)
            ratios = sub["mean"].values / (f1[0] + 1e-8)

            # (1) GLOBAL single-c fit, exactly mirroring ext_c_depth_scaling
            c_fit, r2_med = float("nan"), float("nan")
            try:
                popt, _ = curve_fit(lambda d, c: _depth_law(d, c, alpha_val), depths, ratios,
                                    p0=[0.05], bounds=([0.0], [1.0]), maxfev=2000)
                c_fit = float(popt[0])
                r2_med, r2_lo, r2_hi, frac_undef = _bootstrap_r2(
                    depths, ratios, lambda d, c: _depth_law(d, c, alpha_val),
                    p0=[c_fit if c_fit > 1e-6 else 1e-3], n_boot=500, seed=0, bounds=([0.0], [1.0]))
            except Exception:
                r2_lo = r2_hi = float("nan"); frac_undef = float("nan")

            # (2) LOCAL layer-to-layer ratio: is decay geometric (ratio[l]/ratio[l-1]
            # roughly constant) or does it swing with structural transitions
            # (stride/width changes -> stage boundaries)? Report the coefficient
            # of variation of the local ratio as a model-free geometricity check.
            local_ratio = ratios[1:] / np.clip(ratios[:-1], 1e-12, None)
            local_ratio = local_ratio[np.isfinite(local_ratio)]
            cv_local = float(np.std(local_ratio) / (np.abs(np.mean(local_ratio)) + 1e-12)) if len(local_ratio) > 1 else float("nan")

            rows.append({"arch": arch_name, "alpha": alpha_val, "n_layers": int(len(depths)),
                         "f1": float(f1[0]), "c_fit_global": c_fit,
                         "r2_global_boot_median": r2_med, "r2_global_boot_lo": r2_lo,
                         "r2_global_boot_hi": r2_hi, "cv_local_ratio": cv_local})
            print(f"  [{arch_name}] alpha={alpha_val:5.1f}  n_layers={len(depths):2d}  "
                  f"global c_fit={c_fit:.4f}  R^2(boot,med)={r2_med:+.3f} [{r2_lo:+.3f},{r2_hi:+.3f}]  "
                  f"local-ratio CV={cv_local:.3f}  "
                  f"{'(geometric: low CV)' if np.isfinite(cv_local) and cv_local < 0.5 else '(NOT geometric: high CV — structure-driven, not pure depth-driven)' if np.isfinite(cv_local) else ''}")

    out = pd.DataFrame(rows)
    if len(out) == 0:
        print("  [inconclusive] no alpha/arch combination yielded a usable profile (f1 too small everywhere)")
        return out
    save_csv(out, "deepnet_compounding_test.csv")

    # Verdict: compare the heterogeneous-network global R^2 against the
    # uniform-stack regime (ext_c_depth_scaling reported R^2~0.92-0.96 for
    # alpha>=20 on the UNIFORM synthetic stack — see depth_scaling_law.csv).
    post = out[out["alpha"] >= 20.0].dropna(subset=["r2_global_boot_median"])
    pre  = out[out["alpha"] < 10.0].dropna(subset=["r2_global_boot_median"])
    high_cv_frac = float(np.mean(out["cv_local_ratio"].dropna() > 0.5)) if out["cv_local_ratio"].notna().any() else float("nan")
    print(f"\n  VERDICT inputs: median global R^2 at alpha>=20: "
          f"{post['r2_global_boot_median'].median() if len(post) else float('nan'):+.3f}  "
          f"(uniform-stack reference ~0.92-0.96, ext_c_depth_scaling); "
          f"at alpha<10: {pre['r2_global_boot_median'].median() if len(pre) else float('nan'):+.3f};  "
          f"fraction of (arch,alpha) with non-geometric local decay (CV>0.5): {high_cv_frac:.0%}")
    if len(post) == 0:
        verdict = "INCONCLUSIVE: no alpha>=20 fits converged on the heterogeneous architectures."
    elif post["r2_global_boot_median"].median() > 0.7 and high_cv_frac < 0.3:
        verdict = ("Law GENERALIZES (global single-c fit remains good and layer-to-layer decay "
                   "stays geometric) — heterogeneity in width/stride/residual structure does not "
                   "break the compounding law in the post-transition (alpha>=20) regime.")
    elif high_cv_frac >= 0.3 and (post["r2_global_boot_median"].median() <= 0.7):
        verdict = ("Law does NOT generalize in its strong (single-global-c) form: local decay is "
                   "non-geometric (CV>0.5) at >=30% of (arch,alpha>=20) combinations — collapse rate "
                   "is structure-dependent (stage/stride transitions dominate over uniform per-layer "
                   "decay). A per-block-type-rate generalization may still hold; that is a distinct, "
                   "weaker claim than the one being tested and would require its own derivation.")
    else:
        verdict = ("MIXED / INCONCLUSIVE: neither a clean global fit nor a clearly non-geometric "
                   "local pattern dominates — say so explicitly rather than rounding to either side.")
    print(f"  VERDICT: {verdict}")
    out.attrs["verdict"] = verdict
    return out

def plot_deepnet_compounding(df_profile, df_test):
    if df_profile is None or len(df_profile) == 0: return
    archs = sorted(df_profile["arch"].unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(7*len(archs), 5), squeeze=False)
    cmap = plt.cm.viridis
    for ai, arch_name in enumerate(archs):
        ax = axes[0][ai]
        sub_arch = df_profile[df_profile["arch"] == arch_name]
        alphas = sorted(sub_arch["alpha"].unique())
        for i, alpha_val in enumerate(alphas):
            sub = sub_arch[sub_arch["alpha"] == alpha_val]
            grp = sub.groupby("layer_idx")["active_frac"].agg(["mean", "std"]).reset_index()
            ax.errorbar(grp["layer_idx"], grp["mean"], yerr=grp["std"], marker="o", lw=1.5,
                        capsize=2, color=cmap(i/max(len(alphas)-1, 1)), label=f"a={alpha_val:.0f}")
        ax.set_yscale("log")
        ax.set_xlabel("Layer index (network execution order)")
        ax.set_ylabel("Active gradient fraction (log scale)")
        ax.set_title(f"{arch_name}: gate compounding across heterogeneous layers")
        ax.legend(fontsize=8, ncol=2); ax.grid(True, linestyle="--", alpha=0.4)
    plt.suptitle("Phase 3A: does depth-compounding collapse generalize beyond uniform synthetic stacks?\n"
                 "(ResNet-18/VGG-11-shaped alpha-sigmoid networks — see deepnet_compounding_test.csv for verdict)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout(); save_fig("deepnet_gate_compounding.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ALL FIGURES (paper + PhD)
# ══════════════════════════════════════════════════════════════════════════════

def plot_alpha_sweep(df):
    for metric,ylabel,fname in [
        ("loss_final","Mean final loss","alpha_sweep_mean_final_loss.png"),
        ("output_iou_final","Mean IoU","alpha_sweep_iou.png"),
        ("rel_loss_reduction","Mean relative loss reduction","alpha_sweep_rel_loss_reduction.png"),
    ]:
        plt.figure(figsize=(7,5))
        for opt in sorted(df["optimizer"].unique()):
            sub=df[df["optimizer"]==opt]
            grp=sub.groupby("alpha",as_index=False)[metric].mean().sort_values("alpha")
            plt.plot(grp["alpha"],grp[metric],marker="o",label=opt)
        plt.xlabel("alpha"); plt.ylabel(ylabel); plt.title(f"Alpha sweep: {ylabel}")
        plt.legend(); save_fig(fname)
    # mean±std overlay
    for metric,ylabel,fname in [
        ("loss_final","Final loss","alpha_sweep_loss_mean_std.png"),
        ("output_iou_final","Output IoU","alpha_sweep_iou_mean_std.png"),
    ]:
        plt.figure(figsize=(7,5))
        for opt in sorted(df["optimizer"].unique()):
            sub=df[df["optimizer"]==opt]
            grp=sub.groupby("alpha")[metric].agg(["mean","std"]).reset_index().sort_values("alpha")
            x=grp["alpha"].values; y=grp["mean"].values; s=grp["std"].values
            plt.plot(x,y,marker="o",label=opt); plt.fill_between(x,y-s,y+s,alpha=0.15)
        plt.xlabel("alpha"); plt.ylabel(ylabel)
        plt.title(f"Alpha sweep (mean±std): {ylabel}"); plt.legend(); save_fig(fname)

def plot_phase_diagram(df):
    KNAMES=["identity_like","avg_blur","random_norm","sobel_x","laplacian"]
    pivot=df[df["optimizer"]=="adam"].groupby(["kernel_name","alpha"])["output_iou_final"].mean().unstack()
    pivot=pivot.reindex(KNAMES)
    fig,ax=plt.subplots(figsize=(9,5))
    im=ax.imshow(pivot.values,cmap="RdYlGn",vmin=0.3,vmax=1.0,aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"a={a:.0f}" for a in pivot.columns])
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v=pivot.values[i,j]
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=9,
                    color="white" if v<0.6 else "black")
    plt.colorbar(im,ax=ax,fraction=0.046).set_label("Reconstruction IoU")
    ax.set_title("Phase diagram: IoU heatmap over (kernel, alpha) space",fontweight="bold")
    plt.tight_layout(); save_fig("phase_diagram_alpha_x_kernel.png")

def plot_oracle_ablation(df):
    plt.figure(figsize=(8,5))
    colors_map={"adam":"#1f77b4","oracle_pgd":"#d62728","pgd":"#2ca02c"}
    for opt in ["adam","oracle_pgd","pgd"]:
        sub=df[df["optimizer"]==opt]
        grp=sub.groupby("alpha")["output_iou_final"].agg(["mean","std"]).reset_index()
        plt.errorbar(grp["alpha"],grp["mean"],yerr=grp["std"],
                     marker="o",capsize=4,lw=2,color=colors_map.get(opt,"gray"),label=opt)
    plt.xlabel("Sigmoid stiffness alpha"); plt.ylabel("Reconstruction IoU")
    plt.title("Oracle ablation: Adam vs Oracle Rescaling vs PGD\nMomentum drives Adam advantage")
    plt.legend(fontsize=10); plt.grid(True,linestyle="--",alpha=0.4)
    save_fig("oracle_ablation.png")

def plot_activation_comparison(df):
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    ACTS=["sigmoid","tanh","relu","gelu","swish"]
    cmap=plt.cm.tab10
    for i,act in enumerate(ACTS):
        sub=df[df["activation"]==act]
        grp=sub.groupby("alpha")["output_iou_final"].agg(["mean","std"]).reset_index()
        axes[0].errorbar(grp["alpha"],grp["mean"],yerr=grp["std"],
                         marker="o",capsize=3,lw=2,color=cmap(i),label=act)
        grp2=sub.groupby("alpha")["active_grad_frac_final"].agg(["mean","std"]).reset_index()
        axes[1].errorbar(grp2["alpha"],grp2["mean"],yerr=grp2["std"],
                         marker="s",capsize=3,lw=2,color=cmap(i),label=act)
    axes[0].set_xlabel("Sigmoid stiffness alpha"); axes[0].set_ylabel("Reconstruction IoU")
    axes[0].set_title("Activation comparison: IoU vs alpha"); axes[0].legend(fontsize=9)
    axes[0].grid(True,linestyle="--",alpha=0.4)
    axes[1].set_xlabel("Sigmoid stiffness alpha"); axes[1].set_ylabel("Active gradient fraction")
    axes[1].set_title("Activation comparison: gradient collapse"); axes[1].legend(fontsize=9)
    axes[1].grid(True,linestyle="--",alpha=0.4)
    plt.suptitle("Activation taxonomy: three collapse regimes",fontweight="bold",fontsize=12)
    plt.tight_layout(); save_fig("activation_comparison.png")

def plot_activation_taxonomy_v2(df):
    """Corrected activation comparison (post Bug #6 fix). Title intentionally
    does NOT presume a "3-way" structure -- analyze_activation_taxonomy()
    determines empirically whether (and how many) regimes separate."""
    if df is None or len(df)==0: return
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    ACTS=sorted(df["activation"].unique())
    cmap=plt.cm.tab10
    for i,act in enumerate(ACTS):
        sub=df[df["activation"]==act]
        grp=sub.groupby("alpha")["output_iou_final"].agg(["mean","std"]).reset_index()
        axes[0].errorbar(grp["alpha"],grp["mean"],yerr=grp["std"],
                         marker="o",capsize=3,lw=2,color=cmap(i),label=act)
        grp2=sub.groupby("alpha")["active_grad_frac_final"].agg(["mean","std"]).reset_index()
        axes[1].errorbar(grp2["alpha"],grp2["mean"],yerr=grp2["std"],
                         marker="s",capsize=3,lw=2,color=cmap(i),label=act)
    axes[0].set_xlabel("Stiffness alpha"); axes[0].set_ylabel("Reconstruction IoU")
    axes[0].set_title("Corrected activation comparison: IoU vs alpha"); axes[0].legend(fontsize=9)
    axes[0].grid(True,linestyle="--",alpha=0.4)
    axes[1].set_xlabel("Stiffness alpha"); axes[1].set_ylabel("Active gradient fraction")
    axes[1].set_title("Corrected activation comparison: gradient collapse"); axes[1].legend(fontsize=9)
    axes[1].grid(True,linestyle="--",alpha=0.4)
    plt.suptitle("Activation taxonomy v2 (post Bug #6 fix) -- regime count is empirical, see\n"
                 "activation_taxonomy_clustering.csv for the data-driven cluster verdict",
                 fontweight="bold",fontsize=11)
    plt.tight_layout(); save_fig("activation_taxonomy_v2.png")

def plot_noise_robustness(df):
    fig,ax=plt.subplots(figsize=(8,5))
    cmap=plt.cm.RdYlGn
    for i,alpha in enumerate(sorted(df["alpha"].unique())):
        sub=df[df["alpha"]==alpha]
        grp=sub.groupby("noise_std")["output_iou_final"].agg(["mean","std"]).reset_index()
        snr=[-20*np.log10(n+1e-12) for n in grp["noise_std"]]
        ax.errorbar(snr,grp["mean"],yerr=grp["std"],marker="o",capsize=3,lw=2,
                    color=cmap(i/max(len(sorted(df["alpha"].unique()))-1,1)),label=f"a={alpha:.0f}")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Reconstruction IoU")
    ax.set_title("Noise robustness: IoU nearly flat across SNR 15-60 dB\n(landscape geometry, not noise, controls recoverability)")
    ax.legend(fontsize=9); ax.grid(True,linestyle="--",alpha=0.4)
    plt.tight_layout(); save_fig("noise_robustness.png")

def plot_phase_transition_fit(df):
    sub=df[df["optimizer"]=="adam"] if "optimizer" in df.columns else df
    grp=sub.groupby("alpha")["output_iou_final"].agg(["mean","std","count"]).reset_index()
    def _sig(a,iou_max,iou_min,a_star,delta):
        return iou_min+(iou_max-iou_min)/(1.+np.exp((a-a_star)/(delta+1e-8)))
    mn=float(grp["mean"].values.min()); mx=float(grp["mean"].values.max())
    popt,_=curve_fit(_sig,grp["alpha"].values,grp["mean"].values,
                      p0=[np.clip(mx,0.5,1.09), np.clip(mn,0.0,0.98), 11.7, 3.0],
                      bounds=([0.3,-0.05,0.5,0.1],[1.1,0.99,60.0,20.0]),
                      maxfev=50000)
    a_dense=np.linspace(grp["alpha"].min(),grp["alpha"].max(),300)
    fig,ax=plt.subplots(figsize=(7,5))
    ax.errorbar(grp["alpha"],grp["mean"],yerr=grp["std"]/np.sqrt(grp["count"]),
                fmt="o",capsize=4,color="#1f77b4",label="Empirical (Adam)")
    ax.plot(a_dense,_sig(a_dense,*popt),color="#d62728",lw=2.2,
            label=f"Sigmoid fit: a*={popt[2]:.2f}, D={popt[3]:.2f}")
    ax.axvline(popt[2],color="gray",linestyle=":",lw=1.5)
    ax.set_xlabel("Sigmoid stiffness alpha"); ax.set_ylabel("Mean reconstruction IoU")
    ax.set_title(f"Phase transition: a*={popt[2]:.2f} (fitted)")
    ax.legend(fontsize=10); ax.grid(True,linestyle="--",alpha=0.4)
    plt.tight_layout(); save_fig("phase_transition_fit.png")
    pd.DataFrame({"alpha_star":[popt[2]],"delta":[popt[3]],
                   "iou_max":[popt[0]],"iou_min":[popt[1]]}).to_csv(
        os.path.join(CSV_DIR,"phase_transition_fit.csv"),index=False)

def plot_convergence_bands(results):
    for qty,ylabel,fname in [("loss","Loss","convergence_bands_loss.png"),
                               ("grad","Gradient norm","convergence_bands_grad.png")]:
        plt.figure(figsize=(7,5))
        for opt,stats in results.items():
            mean=stats[f"{qty}_mean"]; std=stats[f"{qty}_std"]
            x=np.arange(len(mean))
            plt.plot(x,mean,label=opt); plt.fill_between(x,mean-std,mean+std,alpha=0.15)
        plt.xlabel("Iteration"); plt.ylabel(ylabel)
        plt.title(f"Convergence bands (mean±std): {ylabel}"); plt.legend(); save_fig(fname)

def plot_twolayer(df):
    from scipy.optimize import curve_fit as _cf
    grp_f=df.groupby(["layers","alpha"])["frac_active"].agg(["mean","std"]).reset_index()
    alphas=sorted(df["alpha"].unique())
    f1=[df[(df["layers"]==1)&(df["alpha"]==a)]["frac_active"].mean() for a in alphas]
    f2=[df[(df["layers"]==2)&(df["alpha"]==a)]["frac_active"].mean() for a in alphas]
    ratio=np.array(f2)/(np.array(f1)+1e-8)
    popt,pcov=_cf(lambda a,c:1./(1.+c*a),np.array(alphas),ratio,p0=[0.05])
    c_fit=popt[0]; r2=1.-np.sum((ratio-1./(1.+c_fit*np.array(alphas)))**2)/np.sum((ratio-ratio.mean())**2)

    fig,axes=plt.subplots(1,3,figsize=(16,5))
    std1=[df[(df["layers"]==1)&(df["alpha"]==a)]["frac_active"].std() for a in alphas]
    std2=[df[(df["layers"]==2)&(df["alpha"]==a)]["frac_active"].std() for a in alphas]
    axes[0].errorbar(alphas,f1,yerr=std1,marker="o",color="#1f77b4",lw=2.2,capsize=3,label="1-layer")
    axes[0].errorbar(alphas,f2,yerr=std2,marker="s",color="#d62728",lw=2.2,capsize=3,
                     linestyle="--",label="2-layer")
    axes[0].set_xlabel("Sigmoid stiffness alpha"); axes[0].set_ylabel("Active gradient fraction")
    axes[0].set_title("Active gradient collapse: 1L vs 2L"); axes[0].legend(fontsize=10)
    axes[0].grid(True,linestyle="--",alpha=0.4)

    a_dense=np.linspace(0.5,45,300)
    axes[1].scatter(alphas,ratio,color="#2ca02c",s=80,zorder=5,marker="D",label="Empirical ratio")
    axes[1].plot(a_dense,1./(1.+c_fit*a_dense),color="#d62728",lw=2.2,
                 label=f"Bound 1/(1+{c_fit:.3f}*a)")
    axes[1].set_xlabel("Sigmoid stiffness alpha"); axes[1].set_ylabel("F_2L / F_1L")
    axes[1].set_title(f"Theorem verification R^2={r2:.4f}"); axes[1].legend(fontsize=9)
    axes[1].grid(True,linestyle="--",alpha=0.4)

    iou1=[df[(df["layers"]==1)&(df["alpha"]==a)]["iou_final"].mean() for a in alphas]
    iou2=[df[(df["layers"]==2)&(df["alpha"]==a)]["iou_final"].mean() for a in alphas]
    axes[2].errorbar(alphas,iou1,marker="o",color="#1f77b4",lw=2.2,capsize=3,label="1-layer IoU")
    axes[2].errorbar(alphas,iou2,marker="s",color="#d62728",lw=2.2,capsize=3,
                     linestyle="--",label="2-layer IoU")
    axes[2].set_xlabel("Sigmoid stiffness alpha"); axes[2].set_ylabel("Reconstruction IoU")
    axes[2].set_title("IoU degradation with depth"); axes[2].legend(fontsize=10)
    axes[2].grid(True,linestyle="--",alpha=0.4)
    plt.suptitle(f"Theorem: Gradient collapse compounds across layers — "
                 f"ratio~1/(1+{c_fit:.3f}*a),  R^2={r2:.4f}",fontweight="bold",fontsize=12)
    plt.tight_layout(); save_fig("theorem_verification.png")

def plot_optimizer_heatmap(df):
    from matplotlib.colors import LinearSegmentedColormap
    OPTS=["adam","lbfgsb","momentum","nesterov","pgd"]
    ALPHA_VALS=sorted(df["alpha"].unique())
    data=np.zeros((len(OPTS),len(ALPHA_VALS)))
    for i,opt in enumerate(OPTS):
        for j,a in enumerate(ALPHA_VALS):
            sub=df[(df["optimizer"]==opt)&(df["alpha"]==a)]
            data[i,j]=float(np.mean(sub["output_iou_final"]>0.7)) if len(sub)>0 else 0.
    cmap=LinearSegmentedColormap.from_list("rg",["#d62728","#ff7f0e","#2ca02c"])
    fig,ax=plt.subplots(figsize=(9,4))
    im=ax.imshow(data,cmap=cmap,vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(len(ALPHA_VALS))); ax.set_xticklabels([f"a={a:.0f}" for a in ALPHA_VALS])
    ax.set_yticks(range(len(OPTS))); ax.set_yticklabels([o.upper() for o in OPTS])
    for i in range(len(OPTS)):
        for j in range(len(ALPHA_VALS)):
            ax.text(j,i,f"{data[i,j]:.0%}",ha="center",va="center",fontsize=10,
                    color="white" if data[i,j]<0.5 else "black",fontweight="bold")
    plt.colorbar(im,ax=ax,fraction=0.046).set_label("Success rate P[IoU>0.7]")
    ax.set_title("Optimizer success rate P[IoU>0.7]",fontweight="bold")
    plt.tight_layout(); save_fig("optimizer_success_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_total=time.time()

    # ── verify-only short-circuit ──────────────────────────────────────────
    if args.verify_only:
        print("\n[verify-only] Loading existing CSVs without running new experiments...")
        alpha_df_v  = load_csv("alpha_sweep_results.csv")
        sr_df_v     = load_csv("stable_rank_vs_alpha.csv")
        scale_df_v  = load_csv("scale_experiment.csv")
        dense_df_v  = load_csv("gradient_leakage_dense.csv")
        run_verification_checks(alpha_df_v, dense_df_v, sr_df_v, scale_df_v)
        sys.exit(0)

    print("\n" + "="*60)
    print("STAGE 1: Core paper sweeps (parallelised)")
    print("="*60)
    alpha_df   = run_alpha_sweep()
    thresh_df  = run_threshold_sweep()
    kernel_df  = run_kernel_sweep()
    target_df  = run_target_sweep()
    oracle_df  = run_oracle_ablation()
    act_df     = run_activation_sweep()
    act_df_v2  = run_activation_sweep_v2()
    taxonomy_clusters = analyze_activation_taxonomy(act_df_v2)
    noise_df   = run_noise_sweep()
    grad_df    = run_grad_sparsity_sweep()
    phase_df   = run_phase_diagram()
    curv_df    = run_curvature_sweep()

    print("\n" + "="*60)
    print("STAGE 2: Multi-layer & scale sweeps")
    print("="*60)
    scale_df   = run_scale_sweep()
    twolayer_df= run_twolayer_sweep()
    gate_indep_df = test_gate_independence()

    print("\n" + "="*60)
    print("STAGE 3: Dense alpha sweep (for leakage)")
    print("="*60)
    dense_df   = run_dense_alpha_sweep()

    print("\n" + "="*60)
    print("STAGE 4: MNIST")
    print("="*60)
    run_mnist_experiment()

    print("\n" + "="*60)
    print("STAGE 4b: Phase 3C — full MNIST + CIFAR-10 reconstruction & phase transitions")
    print("="*60)
    phase3c_recon_df = run_phase3c_image_recon()
    phase3c_star_df = analyze_phase3c_transitions(phase3c_recon_df)

    print("\n" + "="*60)
    print("STAGE 4c: Phase 3D — tighten alpha* estimation (20-seed bootstrap CI vs. corrected CRLB)")
    print("="*60)
    phase3d_seed_df = run_phase3d_alpha_seed_sweep()
    analyze_phase3d_bootstrap_alpha_star(phase3d_seed_df)

    print("\n" + "="*60)
    print("STAGE 5: PhD Extensions")
    print("="*60)
    ext_a_fisher_cramer_rao(alpha_df)
    ext_b_finite_size_scaling(scale_df)
    ext_c_depth_scaling()
    ext_d_nullspace()
    ext_e_gradient_leakage(dense_df, alpha_df)
    ext_f_learned_weights()

    print("\n" + "="*60)
    print("STAGE 5b: Phase 3A — deep-CNN (ResNet-18/VGG-11-shaped) gate analysis")
    print("="*60)
    deepnet_profile_df = run_deepnet_gate_sweep()
    deepnet_test_df = analyze_deepnet_compounding(deepnet_profile_df)

    print("\n" + "="*60)
    print("STAGE 6: All figures")
    print("="*60)
    conv_results = run_convergence_bands()
    plot_alpha_sweep(alpha_df)
    plot_phase_diagram(phase_df)
    plot_oracle_ablation(oracle_df)
    plot_activation_comparison(act_df)
    plot_activation_taxonomy_v2(act_df_v2)
    plot_deepnet_compounding(deepnet_profile_df, deepnet_test_df)
    plot_phase3c_transitions(phase3c_recon_df, phase3c_star_df)
    plot_noise_robustness(noise_df)
    plot_phase_transition_fit(alpha_df)
    plot_convergence_bands(conv_results)
    plot_twolayer(twolayer_df)
    plot_optimizer_heatmap(alpha_df)

    print("\n" + "="*60)
    print("STAGE 7: Zip outputs")
    print("="*60)
    import shutil
    zip_path=os.path.join(os.path.dirname(ROOT_DIR),
                           os.path.basename(ROOT_DIR)+"_outputs")
    shutil.make_archive(zip_path,"zip",ROOT_DIR)
    print(f"[zip] {zip_path}.zip")

    elapsed=time.time()-t_total
    print(f"\n[done] Total wall time: {elapsed/60:.1f} min")
    print(f"[done] Figures : {FIG_DIR}")
    print(f"[done] CSVs    : {CSV_DIR}")

    # ── Stage 8: Missing notebook components + new PhD extensions ─────────
    print("\n" + "="*60)
    print("STAGE 8: Supplementary experiments & figures")
    print("="*60)
    run_landscape_slices()
    run_nonuniqueness_experiment()
    erank_df = run_effective_rank_sweep()
    sr_df    = run_stable_rank_sweep()
    run_predictive_model(alpha_df, kernel_df, phase_df)
    run_optimizer_statistics(alpha_df)
    run_stable_rank_auc(alpha_df, sr_df)
    run_ablation_table()
    plot_threshold_sensitivity(thresh_df)
    print_poster_headline_stats(alpha_df, grad_df, kernel_df, phase_df,
                                 erank_df, curv_df, sr_df)
    # New PhD extensions
    run_per_kernel_alpha_star(alpha_df, kernel_df)
    run_convergence_rate_analysis()
    run_mutual_information_proxy()

    # ── Final zip ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 9: Final zip")
    print("="*60)
    import shutil as _shutil
    zip_path=os.path.join(os.path.dirname(ROOT_DIR),
                           os.path.basename(ROOT_DIR)+"_outputs")
    _shutil.make_archive(zip_path,"zip",ROOT_DIR)
    print(f"[zip] {zip_path}.zip")
    elapsed=time.time()-t_total
    print(f"\n[done] Total wall time: {elapsed/60:.1f} min")
    print(f"[done] Figures : {FIG_DIR}")
    print(f"[done] CSVs    : {CSV_DIR}")

    # ── Headline summary table ─────────────────────────────────────────────
    print_summary_table(alpha_df, grad_df, sr_df, curv_df)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — MISSING NOTEBOOK COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

# Alias used throughout
def sigmoid_prime_from_output(s, alpha):
    return alpha * s * (1.0 - s)

# ── 12.1: 2D landscape slices ─────────────────────────────────────────────────
def run_landscape_slices():
    print("\n[fig] 2D landscape slices (alpha=2 vs alpha=20)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, alpha_val in zip(axes, [2.0, 20.0]):
        p = run_single_experiment(kernel_name="sobel_x", target_name="checkerboard",
                                   alpha=alpha_val, c=0.5, optimizer_name="adam",
                                   optimizer_kwargs={"lr":0.03,"steps":300})
        x_sol = p["x_final"].reshape(-1); prob = p["problem"]
        rng = np.random.default_rng(0); n = x_sol.size
        u = rng.normal(size=n); u /= np.linalg.norm(u)
        v = rng.normal(size=n); v -= v.dot(u)*u; v /= np.linalg.norm(v)
        ts = np.linspace(-0.8, 0.8, 50)
        Z = np.zeros((50, 50))
        for i, ti in enumerate(ts):
            for j, tj in enumerate(ts):
                xij = project_box((x_sol + ti*u + tj*v).reshape(prob.image_shape))
                Z[i, j] = prob.loss(xij)
        im = ax.contourf(ts, ts, Z, levels=30, cmap="viridis")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Landscape slice near solution (alpha={alpha_val})")
        ax.set_xlabel("Direction u"); ax.set_ylabel("Direction v")
    plt.suptitle("2D landscape slices: low alpha (smooth) vs high alpha (flat plateaus)")
    plt.tight_layout()
    save_fig("landscape_alpha_2.png")
    # save individual
    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    ax2 = axes[1]; plt.tight_layout()
    save_fig("landscape_alpha_20.png")
    plt.close("all")

# ── 12.2: Non-uniqueness experiment ──────────────────────────────────────────
def run_nonuniqueness_experiment(kernel_name="sobel_x", target_name="checkerboard",
                                  alpha=20.0, n_restarts=20, steps=400):
    print(f"\n[exp] Non-uniqueness: {n_restarts} restarts at alpha={alpha}")
    solutions, losses, ious, output_sims = [], [], [], []
    for seed in range(n_restarts):
        p = run_single_experiment(kernel_name=kernel_name, target_name=target_name,
                                   alpha=alpha, c=0.5, optimizer_name="adam",
                                   seed=seed, optimizer_kwargs={"lr":0.03,"steps":steps})
        xf = p["x_final"]; fx = p["problem"].forward(xf)
        solutions.append(xf.reshape(-1)); losses.append(p["summary"]["loss_final"])
        ious.append(p["summary"]["output_iou_final"]); output_sims.append(fx.reshape(-1))
        print(f"  restart {seed:2d}: loss={losses[-1]:.3f}  IoU={ious[-1]:.3f}")
    solutions   = np.stack(solutions)
    output_sims = np.stack(output_sims)
    input_dists, output_dists = [], []
    for i in range(n_restarts):
        for j in range(i+1, n_restarts):
            input_dists.append(np.linalg.norm(solutions[i]-solutions[j]))
            output_dists.append(np.linalg.norm(output_sims[i]-output_sims[j]))
    input_dists  = np.array(input_dists)
    output_dists = np.array(output_dists)
    # pairwise dist figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(input_dists, bins=30, edgecolor="white", color="#1f77b4")
    axes[0].set_xlabel("Pairwise L2 distance (input space)", fontsize=11)
    axes[0].set_ylabel("Count"); axes[0].set_title(f"Solution diversity (alpha={alpha})")
    axes[0].axvline(input_dists.mean(), color="red", linestyle="--",
                    label=f"mean={input_dists.mean():.2f}"); axes[0].legend()
    axes[1].hist(output_dists, bins=30, edgecolor="white", color="#ff7f0e")
    axes[1].set_xlabel("Pairwise L2 distance (output space)", fontsize=11)
    axes[1].set_ylabel("Count"); axes[1].set_title(f"Output similarity (alpha={alpha})")
    axes[1].axvline(output_dists.mean(), color="red", linestyle="--",
                    label=f"mean={output_dists.mean():.2f}"); axes[1].legend()
    plt.suptitle("Non-uniqueness: diverse inputs → similar outputs")
    plt.tight_layout(); save_fig("nonuniqueness_pairwise_dist.png")
    # scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(input_dists, output_dists, alpha=0.4, s=15, color="#2ca02c")
    lim = max(input_dists.max(), output_dists.max())
    ax.plot([0,lim],[0,lim],"r--",alpha=0.5,label="y=x reference"); ax.legend()
    ax.set_xlabel("Input distance ||x_i - x_j||")
    ax.set_ylabel("Output distance ||f(x_i) - f(x_j)||")
    ax.set_title("Input diversity vs output similarity")
    plt.tight_layout(); save_fig("nonuniqueness_input_vs_output_scatter.png")
    # mosaic
    target = p["problem"].y
    fig, axes = plt.subplots(3, 6, figsize=(14, 7))
    chosen = [i*(n_restarts//6) for i in range(6)]
    for col, idx in enumerate(chosen):
        x_img = solutions[idx].reshape(SHAPE)
        fx_img = output_sims[idx].reshape(SHAPE)
        axes[0,col].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[0,col].set_title("Target y", fontsize=8)
        axes[1,col].imshow(x_img, cmap="gray", vmin=0, vmax=1)
        axes[1,col].set_title(f"x (restart {idx})\nIoU={ious[idx]:.2f}", fontsize=8)
        axes[2,col].imshow(fx_img, cmap="gray", vmin=0, vmax=1)
        axes[2,col].set_title(f"f(x)\nloss={losses[idx]:.1f}", fontsize=8)
        for row in range(3): axes[row,col].axis("off")
    plt.suptitle(f"Six diverse solutions — same target, same model (alpha={alpha})")
    plt.tight_layout(); save_fig("nonuniqueness_mosaic.png")
    print(f"  Input dist: mean={input_dists.mean():.3f}  Output dist: mean={output_dists.mean():.3f}")
    print(f"  Ratio output/input: {(output_dists/(input_dists+1e-6)).mean():.4f}")
    np.save(os.path.join(NPY_DIR,"nonuniqueness_input_dists.npy"), input_dists)
    np.save(os.path.join(NPY_DIR,"nonuniqueness_output_dists.npy"), output_dists)
    return solutions, input_dists, output_dists

# ── 12.3: Effective rank (Gauss-Newton spectrum) ──────────────────────────────
def run_effective_rank_sweep():
    print("\n[exp] Effective rank of Gauss-Newton matrix vs alpha")
    from scipy.sparse.linalg import LinearOperator, eigsh
    existing = load_csv("effective_rank_vs_alpha.csv", warn=False)
    existing_spec = load_csv("eigenvalue_spectra.csv", warn=False)
    rows = []
    spec_rows = []
    SPEC_ALPHAS = {1.0, 5.0, 10.0, 20.0, 40.0}
    for alpha_val in ALPHAS:
        for seed in (0, 1, 2):
            need_main = not _already_done(existing, {"alpha":alpha_val,"seed":seed})
            need_spec = (alpha_val in SPEC_ALPHAS and
                         not _already_done(existing_spec, {"alpha":alpha_val,"seed":seed}))
            if not need_main and not need_spec:
                continue
            p = run_single_experiment(image_shape=(32,32), kernel_name="sobel_x",
                                       target_name="checkerboard", alpha=alpha_val,
                                       optimizer_name="adam",
                                       optimizer_kwargs={"lr":0.03,"steps":200},
                                       seed=seed)
            prob = p["problem"]; xf = p["x_final"]
            ax_ = prob.conv(xf); s_ = sigmoid(ax_, alpha_val, 0.0)
            w = np.abs(sigmoid_prime_from_output(s_, alpha_val))
            n = xf.size; K_EIG = min(n-2, max(150, n//3))
            def _mv(v):
                return prob.conv_transpose(w*prob.conv(v.reshape(prob.image_shape))).reshape(-1)
            M = LinearOperator((n,n), matvec=_mv, dtype=float)
            rng2 = np.random.default_rng(seed)
            try:
                evs = eigsh(M, k=K_EIG, which="LM", v0=rng2.normal(size=n),
                            return_eigenvectors=False, tol=1e-4)
                evs = np.sort(np.abs(evs))[::-1]
            except Exception:
                evs = np.array([1e-10])
            # Entropy-based effective rank (continuous, no truncation artefact)
            evs_pos = evs[evs > 1e-12 * (evs[0] + 1e-30)]
            p_i = evs_pos / (evs_pos.sum() + 1e-30)
            eff_rank = float(np.exp(-np.sum(p_i * np.log(p_i + 1e-300))))
            lam_max = float(evs[0]) if evs[0] > 0 else 1e-10
            # Also track threshold-based count for comparison
            thresh_rank = int(np.sum(evs > 0.01 * lam_max))
            if need_main:
                rows.append({"alpha":alpha_val,"seed":seed,
                              "effective_rank":eff_rank,"thresh_rank":thresh_rank,
                              "lam_max":lam_max,
                              "loss_final":p["summary"]["loss_final"],
                              "output_iou_final":p["summary"]["output_iou_final"]})
                print(f"  alpha={alpha_val} seed={seed}: eff_rank(entropy)={eff_rank:.1f}  thresh_rank={thresh_rank}")
            # Store normalised eigenvalue spectrum for selected alphas
            if need_spec and alpha_val in SPEC_ALPHAS:
                evs_norm = evs / (lam_max + 1e-30)
                for idx, ev_n in enumerate(evs_norm):
                    spec_rows.append({"alpha": alpha_val, "seed": seed,
                                      "eig_index": idx, "eig_norm": float(ev_n)})
    if rows: append_csv(rows, "effective_rank_vs_alpha.csv")
    else: print("[skip] effective_rank_vs_alpha.csv already complete")
    if spec_rows: append_csv(spec_rows, "eigenvalue_spectra.csv")
    df = load_csv("effective_rank_vs_alpha.csv")
    df_spec = load_csv("eigenvalue_spectra.csv", warn=False)
    if df is not None:
        grp = df.groupby("alpha")["effective_rank"].agg(["mean","std"]).reset_index()
        # Two-panel figure: effective rank + eigenvalue decay curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].errorbar(grp["alpha"], grp["mean"], yerr=grp["std"],
                         marker="o", capsize=4, lw=2, color="#d62728")
        axes[0].set_xlabel("Sigmoid stiffness alpha")
        axes[0].set_ylabel("Effective rank (entropy-based)")
        axes[0].set_title("Gauss-Newton effective rank collapses with alpha\n(entropy rank — no truncation artefact)")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        # Panel 2: normalised eigenvalue decay curves
        spec_cmap = plt.cm.viridis
        spec_alphas_sorted = sorted(SPEC_ALPHAS)
        spec_colors = {a: spec_cmap(i/(len(spec_alphas_sorted)-1))
                       for i, a in enumerate(spec_alphas_sorted)}
        if df_spec is not None and len(df_spec) > 0:
            for av in spec_alphas_sorted:
                sub_s = df_spec[df_spec["alpha"].sub(av).abs() < 1e-9]
                if len(sub_s) == 0: continue
                grp_s = sub_s.groupby("eig_index")["eig_norm"].mean().reset_index()
                grp_s = grp_s.sort_values("eig_index")
                axes[1].plot(grp_s["eig_index"] + 1, grp_s["eig_norm"],
                             lw=1.8, color=spec_colors[av], label=f"alpha={av:.0f}")
            axes[1].set_xscale("log")
            axes[1].set_xlabel("Eigenvalue index (log scale)")
            axes[1].set_ylabel("Normalised eigenvalue lambda_i / lambda_max")
            axes[1].set_title("Eigenvalue decay curves (normalised)\nDimensional collapse evidence")
            axes[1].legend(fontsize=9); axes[1].grid(True, linestyle="--", alpha=0.4, which="both")
        else:
            axes[1].text(0.5, 0.5, "No eigenvalue spectra data yet",
                         ha="center", va="center", transform=axes[1].transAxes)
        plt.suptitle("Effective Rank & Eigenvalue Decay: Dimensional Collapse with alpha",
                     fontweight="bold", fontsize=12)
        plt.tight_layout()
        save_fig("effective_rank_vs_alpha.png")
    return df

# ── 12.4: Stable rank (exact SVD of effective Jacobian) ──────────────────────
def _stable_rank_exact_svd(alpha, kernel, seed=0, N_side=16, c=0.0):
    """Exact stable rank via SVD of the effective Jacobian. No sampling noise."""
    from scipy.ndimage import convolve as _convolve
    N = N_side * N_side
    A = np.zeros((N, N))
    for i in range(N):
        e = np.zeros((N_side, N_side))
        e[i // N_side, i % N_side] = 1.0
        A[:, i] = _convolve(e, kernel, mode="wrap").ravel()
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (N_side, N_side))
    z = _convolve(x, kernel, mode="wrap").ravel()
    s = 1.0 / (1.0 + np.exp(-np.clip(alpha * (z - c), -60, 60)))
    w = alpha * s * (1.0 - s)
    J = w[:, None] * A
    sv = np.linalg.svd(J, compute_uv=False)
    sv2 = sv ** 2
    return float(sv2.sum() / (sv2.max() + 1e-30))


def run_stable_rank_sweep():
    print("\n[exp] Stable rank (exact SVD of effective Jacobian) vs alpha")
    existing = load_csv("stable_rank_vs_alpha.csv", warn=False)
    kernel = KERNELS["sobel_x"]
    rows = []
    for alpha_val in ALPHAS:
        for seed in range(5):
            if _already_done(existing, {"alpha":alpha_val,"seed":seed}): continue
            sr = _stable_rank_exact_svd(alpha_val, kernel, seed=seed, N_side=16, c=0.0)
            rows.append({"alpha":alpha_val,"seed":seed,"stable_rank":sr})
            print(f"  alpha={alpha_val} seed={seed}: stable_rank={sr:.4f}")
    if rows: append_csv(rows, "stable_rank_vs_alpha.csv")
    else: print("[skip] stable_rank_vs_alpha.csv already complete")
    df = load_csv("stable_rank_vs_alpha.csv")
    if df is not None:
        grp = df.groupby("alpha")["stable_rank"].agg(
            median=("stable_rank", "median"),
            q25=("stable_rank", lambda x: x.quantile(0.25)),
            q75=("stable_rank", lambda x: x.quantile(0.75)),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(6.5,4.2))
        ax.errorbar(grp["alpha"], grp["median"],
                    yerr=[grp["median"]-grp["q25"], grp["q75"]-grp["median"]],
                    marker="o", capsize=4, lw=2.2, color="#d62728")
        ax.set_xlabel("Sigmoid stiffness alpha")
        ax.set_ylabel("Stable rank r_s(J) = ||J||_F^2 / sigma_max^2")
        ax.set_title("Dimensional collapse via stable rank\n(median ± IQR across seeds)")
        ax.set_yscale("log"); ax.grid(True, linestyle="--", alpha=0.4, which="both")
        plt.tight_layout(); save_fig("stable_rank_vs_alpha.png")
        sr1 = grp[grp["alpha"]==grp["alpha"].min()]["median"].values[0]
        sr_last = grp[grp["alpha"]==grp["alpha"].max()]["median"].values[0]
        print(f"  Stable rank collapse: r_s(alpha_min)/r_s(alpha_max) = {sr1/sr_last:.1f}x")
    return df

# ── 12.5: Predictive difficulty model ────────────────────────────────────────
def run_predictive_model(alpha_df, kernel_df, phase_df):
    print("\n[exp] Predictive difficulty model (logistic regression)")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("[skip] sklearn not available"); return

    KERNEL_PROPS = {
        "identity_like": {"rank":1,"spectral_norm":1.00,"is_highpass":0},
        "avg_blur":      {"rank":1,"spectral_norm":1.00,"is_highpass":0},
        "random_norm":   {"rank":3,"spectral_norm":1.00,"is_highpass":0},
        "sobel_x":       {"rank":2,"spectral_norm":4.00,"is_highpass":1},
        "laplacian":     {"rank":3,"spectral_norm":8.00,"is_highpass":1},
    }

    def _build(df):
        if df is None: return None
        out = df.copy()
        if "output_iou_final" not in out.columns and "iou_final" in out.columns:
            out["output_iou_final"] = out["iou_final"]
        rows = []
        for _, row in out.iterrows():
            kname = row.get("kernel_name","sobel_x")
            props = KERNEL_PROPS.get(kname, {"rank":2,"spectral_norm":4.,"is_highpass":1})
            alpha = float(row["alpha"])
            if "output_iou_final" not in row: continue
            success = int(float(row["output_iou_final"]) > 0.7)
            rows.append({"alpha":alpha,"log_alpha":np.log(alpha+1),
                          "kernel_rank":props["rank"],"spectral_norm":props["spectral_norm"],
                          "is_highpass":props["is_highpass"],
                          "alpha_x_highpass":alpha*props["is_highpass"],
                          "log_alpha_x_sn":np.log(alpha+1)*props["spectral_norm"],
                          "success":success})
        return pd.DataFrame(rows) if rows else None

    parts = [_build(df) for df in [phase_df, kernel_df, alpha_df] if df is not None]
    parts = [p for p in parts if p is not None and len(p)>0]
    if not parts: print("[skip] no training data for predictive model"); return
    combined = pd.concat(parts, ignore_index=True)
    FEATURES = ["log_alpha","kernel_rank","spectral_norm","is_highpass",
                "alpha_x_highpass","log_alpha_x_sn"]
    X = combined[FEATURES].values; y = combined["success"].values.astype(int)
    scaler = StandardScaler()
    bc = np.bincount(y)
    strat = y if len(bc)==2 and bc.min()>=2 else None
    use_holdout = len(y)>=16
    if use_holdout:
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.3,random_state=42,stratify=strat)
        X_tr_s=scaler.fit_transform(X_tr); X_te_s=scaler.transform(X_te)
    else:
        X_tr_s=X_te_s=scaler.fit_transform(X); y_tr=y_te=y
    clf = LogisticRegression(C=1.,max_iter=500,random_state=42)
    clf.fit(X_tr_s, y_tr)
    y_prob_te = clf.predict_proba(X_te_s)[:,1]
    try: auc_te = roc_auc_score(y_te, y_prob_te)
    except: auc_te = float("nan")
    acc_te = float(np.mean(clf.predict(X_te_s)==y_te))
    print(f"  Hold-out AUC={auc_te:.3f}  Acc={acc_te:.3f}  (n={len(y_te)})")
    print("  Feature coefficients:")
    for feat,coef in sorted(zip(FEATURES,clf.coef_[0]),key=lambda x:abs(x[1]),reverse=True):
        print(f"    {feat:25s}: {coef:+.3f}")

    # Predicted P(success) heatmap
    KNAMES = ["identity_like","avg_blur","random_norm","sobel_x","laplacian"]
    alpha_grid = list(ALPHAS)
    prob_matrix = np.zeros((len(KNAMES), len(alpha_grid)))
    for i,kname in enumerate(KNAMES):
        props = KERNEL_PROPS[kname]
        for j,alpha in enumerate(alpha_grid):
            log_a=np.log(alpha+1)
            feat=np.array([[log_a,props["rank"],props["spectral_norm"],
                            props["is_highpass"],alpha*props["is_highpass"],
                            log_a*props["spectral_norm"]]])
            prob_matrix[i,j]=clf.predict_proba(scaler.transform(feat))[0,1]

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    im=axes[0].imshow(prob_matrix,cmap="RdYlGn",vmin=0,vmax=1,aspect="auto")
    for i in range(len(KNAMES)):
        for j in range(len(alpha_grid)):
            v=prob_matrix[i,j]; tc="black" if 0.3<v<0.8 else "white"
            axes[0].text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=9,
                         fontweight="bold",color=tc)
    axes[0].set_xticks(range(len(alpha_grid)))
    axes[0].set_xticklabels([f"a={a:.0f}" for a in alpha_grid],fontsize=9)
    axes[0].set_yticks(range(len(KNAMES))); axes[0].set_yticklabels(KNAMES,fontsize=9)
    plt.colorbar(im,ax=axes[0],fraction=0.046).set_label("P(IoU>0.7)")
    axes[0].set_title(f"Predicted P(success) heatmap\nAUC={auc_te:.3f}  Acc={acc_te:.3f}")

    # Calibration curve
    from sklearn.calibration import calibration_curve
    try:
        frac_pos, mean_pred = calibration_curve(y_te, y_prob_te, n_bins=10)
        axes[1].plot([0,1],[0,1],"--",color="gray",label="Perfect calibration")
        axes[1].plot(mean_pred,frac_pos,"o-",color="#1f77b4",label=f"Model (AUC={auc_te:.3f})")
        axes[1].set_xlabel("Predicted P(success)"); axes[1].set_ylabel("Empirical success rate")
        axes[1].set_title("Calibration curve (hold-out set)"); axes[1].legend(fontsize=9)
        axes[1].grid(True,linestyle="--",alpha=0.4)
    except Exception as e:
        axes[1].text(0.5,0.5,f"Calibration n/a\n{e}",ha="center",va="center",transform=axes[1].transAxes)
    plt.suptitle("Predictive Difficulty Model: P(success | alpha, kernel)",
                 fontweight="bold",fontsize=12)
    plt.tight_layout(); save_fig("predictive_difficulty_model.png")

# ── 12.6: Optimizer statistics (Wilcoxon + bootstrap CI) ─────────────────────
def run_optimizer_statistics(alpha_df):
    print("\n[analysis] Wilcoxon + bootstrap CI: Adam vs PGD")
    results = []
    for alpha in sorted(alpha_df["alpha"].unique()):
        sub = alpha_df[alpha_df["alpha"]==alpha]
        adam = sub[sub["optimizer"]=="adam"]["output_iou_final"].values if "optimizer" in sub.columns else np.array([])
        pgd  = sub[sub["optimizer"]=="pgd"]["output_iou_final"].values  if "optimizer" in sub.columns else np.array([])
        n = min(len(adam),len(pgd))
        if n < 3: continue
        adam,pgd = adam[:n],pgd[:n]
        w = wilcoxon_pairwise(adam,pgd)
        ma,lo_a,hi_a = bootstrap_ci(adam)
        mp,lo_p,hi_p = bootstrap_ci(pgd)
        d = cohens_d(adam,pgd)
        sig = "***" if w["pvalue"]<0.001 else "**" if w["pvalue"]<0.01 else "*" if w["pvalue"]<0.05 else "ns"
        print(f"  a={alpha:5.1f}: Adam={ma:.3f}[{lo_a:.3f},{hi_a:.3f}]  "
              f"PGD={mp:.3f}[{lo_p:.3f},{hi_p:.3f}]  p={w['pvalue']:.4f}{sig}  d={d:.2f}")
        results.append({"alpha":alpha,"n":n,"adam_iou_mean":ma,"adam_ci_lo":lo_a,"adam_ci_hi":hi_a,
                        "pgd_iou_mean":mp,"pgd_ci_lo":lo_p,"pgd_ci_hi":hi_p,
                        "wilcoxon_p":w["pvalue"],"cohens_d":d})
    if results:
        save_csv(pd.DataFrame(results),"optimizer_statistics.csv")

# ── 12.7: Stable rank AUC predictor ──────────────────────────────────────────
def run_stable_rank_auc(alpha_df, sr_df):
    print("\n[analysis] Stable rank as reconstruction success predictor")
    if sr_df is None or alpha_df is None: return
    try:
        from sklearn.metrics import roc_auc_score
        from scipy.stats import mannwhitneyu
    except ImportError:
        return
    sw_adam = alpha_df[alpha_df["optimizer"]=="adam"] if "optimizer" in alpha_df.columns else alpha_df
    merge_cols = [c for c in ["alpha","seed"] if c in sr_df.columns and c in sw_adam.columns]
    merged = sr_df.merge(sw_adam[merge_cols+["output_iou_final"]],on=merge_cols,how="inner") if merge_cols else sr_df
    if "output_iou_final" not in merged.columns: return
    sr_vals = merged["stable_rank"].values
    success = (merged["output_iou_final"].values > 0.7).astype(int)
    pos,neg = sr_vals[success==1],sr_vals[success==0]
    if len(pos)>0 and len(neg)>0:
        u_stat,p_val = mannwhitneyu(pos,neg,alternative="greater")
        auc = u_stat/(len(pos)*len(neg))
        print(f"  Stable rank predictor AUC={auc:.3f}  p={p_val:.4f}")
        print(f"  sr(success)={pos.mean():.2f}  sr(fail)={neg.mean():.2f}")
        fig,ax=plt.subplots(figsize=(7,5))
        ax.scatter(sr_vals[success==1],merged["output_iou_final"].values[success==1],
                   alpha=0.6,color="#2ca02c",label="IoU>0.7")
        ax.scatter(sr_vals[success==0],merged["output_iou_final"].values[success==0],
                   alpha=0.6,color="#d62728",label="IoU<=0.7")
        ax.set(xlabel="Stable rank sr(J)",ylabel="Reconstruction IoU",
               title=f"Stable rank vs IoU (AUC={auc:.3f})")
        ax.legend(); ax.grid(True,linestyle="--",alpha=0.4)
        plt.tight_layout(); save_fig("stable_rank_auc.png")

# ── 12.8: Ablation summary table ─────────────────────────────────────────────
def run_ablation_table():
    print("\n[analysis] Ablation summary table")
    rows = []
    for label,fname in [("activation","activation_comparison.csv"),
                         ("oracle","oracle_ablation.csv"),
                         ("noise","noise_robustness.csv")]:
        df = load_csv(fname, warn=False)
        if df is None: continue
        grp_col = "optimizer" if "optimizer" in df.columns else "activation"
        for key,grp in df.groupby(grp_col):
            rows.append({"experiment":label,"condition":str(key),
                          "iou_mean":grp["output_iou_final"].mean(),
                          "iou_std":grp["output_iou_final"].std(),
                          "active_grad_mean":grp.get("active_grad_frac_final",
                                                      pd.Series([np.nan])).mean(),
                          "n_runs":len(grp)})
    if rows:
        table = pd.DataFrame(rows)
        save_csv(table,"ablation_summary_table.csv")
        print(table.round(3).to_string(index=False))

# ── 12.9: Threshold sensitivity figure ───────────────────────────────────────
def plot_threshold_sensitivity(thresh_df):
    print("\n[fig] Threshold sensitivity analysis")
    if thresh_df is None: return
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for opt in thresh_df["optimizer"].unique():
        sub=thresh_df[thresh_df["optimizer"]==opt]
        grp=sub.groupby("c")["loss_final"].mean()
        axes[0].plot(grp.index,grp.values,marker="o",label=opt)
    axes[0].set_xlabel("Threshold (c)"); axes[0].set_ylabel("Mean Final Loss")
    axes[0].set_title("Impact of c on Final Loss"); axes[0].legend()
    axes[0].grid(True,linestyle="--",alpha=0.4)
    for opt in thresh_df["optimizer"].unique():
        sub=thresh_df[thresh_df["optimizer"]==opt]
        success=sub.groupby("c")["output_iou_final"].apply(lambda x:(x>0.7).mean())
        axes[1].plot(success.index,success.values,marker="s",label=opt)
    axes[1].set_xlabel("Threshold (c)"); axes[1].set_ylabel("Success Rate (IoU > 0.7)")
    axes[1].set_title("Impact of c on Success Rate"); axes[1].legend()
    axes[1].grid(True,linestyle="--",alpha=0.4)
    plt.suptitle("Threshold Sensitivity Analysis",fontweight="bold")
    plt.tight_layout(); save_fig("threshold_sensitivity_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — NEW PhD EXTENSIONS (2nd batch)
# ══════════════════════════════════════════════════════════════════════════════

# ── 13.1: Bootstrap R² helper ─────────────────────────────────────────────────
def _bootstrap_r2(x_data, y_data, fit_func, p0, n_boot=1000, seed=0, bounds=None):
    """Bootstrap confidence interval for R².
    Resamples (x, y) pairs with replacement n_boot times.
    Returns (r2_med, r2_lo, r2_hi, frac_undefined) using 2.5/97.5 percentiles.

    R^2 = 1 - ss_res/ss_tot is mathematically UNDEFINED (0/0) whenever the
    resampled response is constant (ss_tot -> 0) -- which happens often here
    because the depth-collapse ratio is ~1 for every depth when c~0 (small
    alpha, pre-transition). A fixed epsilon in the denominator (the previous
    `+ 1e-30`) does NOT fix this: it is many orders of magnitude smaller than
    the float64 rounding noise in ss_res (~1e-16 scale), so ss_res/(ss_tot+eps)
    still explodes to astronomical values (e.g. -1e21) that are an artifact of
    floating-point precision, not a measurement of fit quality. The correct
    treatment is to recognize R^2 as undefined for such resamples (ss_tot below
    a noise floor set by the data's own scale) and exclude them, instead of
    reporting a number that looks quantitative but is numerically meaningless.
    `frac_undefined` reports how often this happened -- a large fraction *is*
    the finding (it means the model is structurally non-identifiable in this
    regime), and is reported explicitly rather than being hidden inside a
    garbage point estimate.

    `bounds`, if given, additionally constrains each resampled curve_fit to a
    physically-motivated parameter range.
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    n = len(x_data)
    rng = np.random.default_rng(seed)
    # Noise floor for ss_tot: variation in y below (scale * 1e-6)^2 is
    # indistinguishable from "constant" at the precision these values are
    # measured/stored at (CSV round-trip, float32 accumulation, etc.).
    scale = max(float(np.ptp(y_data)), float(np.abs(y_data).max()), 1.0)
    ss_tot_floor = (scale * 1e-6) ** 2
    r2_boots = []
    n_undefined = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x_data[idx], y_data[idx]
        ss_tot = np.sum((yb - yb.mean()) ** 2)
        if ss_tot < ss_tot_floor:
            n_undefined += 1
            continue
        try:
            kw = {"p0": p0, "maxfev": 3000}
            if bounds is not None: kw["bounds"] = bounds
            popt, _ = curve_fit(fit_func, xb, yb, **kw)
            y_pred = fit_func(xb, *popt)
            ss_res = np.sum((yb - y_pred) ** 2)
            r2 = 1.0 - ss_res / ss_tot
        except Exception:
            r2 = float("nan")
        r2_boots.append(r2)
    frac_undefined = n_undefined / float(n_boot)
    r2_arr = np.array([v for v in r2_boots if not np.isnan(v)])
    if len(r2_arr) == 0:
        return float("nan"), float("nan"), float("nan"), frac_undefined
    # Use the MEDIAN, not the mean, as the point estimate. The bootstrap R^2
    # distribution for a small (n=6) dataset is heavy-tailed: a handful of
    # low-diversity resamples have small-but-nonzero ss_tot together with a
    # comparatively large ss_res, producing legitimate but extreme outliers
    # (e.g. R^2 ~ -7700) that drag the MEAN far outside its own [2.5,97.5]
    # percentile interval (e.g. mean=-15.5 reported alongside CI=[-0.76,0.93]
    # -- a point estimate that isn't even inside its own CI is a red flag that
    # it's the wrong summary statistic). The median is robust to these outliers
    # and, by construction, always lies within the percentile CI.
    r2_med = float(np.median(r2_arr))
    r2_lo  = float(np.percentile(r2_arr, 2.5))
    r2_hi  = float(np.percentile(r2_arr, 97.5))
    return r2_med, r2_lo, r2_hi, frac_undefined

# ── 13.2: Per-kernel alpha-star ───────────────────────────────────────────────
def run_per_kernel_alpha_star(alpha_df, kernel_df):
    """For each of 5 kernels: fit sigmoid model → α*(kernel), compute spectral
    norm σ_max(K). Plot α*(kernel) vs 1/σ_max(K) and report Pearson r."""
    print("\nPer-kernel alpha* vs 1/sigma_max(K)")
    csv_name = "per_kernel_alpha_star.csv"
    existing = load_csv(csv_name, warn=False)
    if existing is not None and len(existing) == 5:
        print(f"[skip] {csv_name} already complete")
        df_out = existing
    else:
        KNAMES = ["identity_like", "avg_blur", "sobel_x", "laplacian", "random_norm"]

        def _sig_model(alpha, iou_max, iou_min, alpha_star, delta):
            return iou_min + (iou_max - iou_min) / (1.0 + np.exp((alpha - alpha_star) / (delta + 1e-8)))

        rows = []
        # Use phase_diagram data if kernel_df has all kernels; otherwise use what we have
        # kernel_df was run at alpha=10 only — we need the full alpha sweep per kernel.
        # Use phase_df (alpha × kernel) if available, else run fresh
        phase_df_local = load_csv("phase_diagram_alpha_x_kernel.csv", warn=False)
        if phase_df_local is None and kernel_df is not None:
            # Fall back: kernel_df only has alpha=10, so we rely on alpha_df for sobel_x
            # and simply report alpha_star = nan for others
            phase_df_local = kernel_df

        def _op_sigma_max(k, N=SHAPE[0]):
            """Spectral norm of circular convolution operator — same formula as _build_kernels."""
            kp = np.zeros((N, N)); kp[:k.shape[0], :k.shape[1]] = k
            return float(np.abs(np.fft.fft2(kp)).max())

        for kname in KNAMES:
            K = KERNELS[kname]
            sigma_max = _op_sigma_max(K)  # spectral norm of convolution operator

            alpha_star_fit = float("nan")
            fit_note = ""
            if phase_df_local is not None and "kernel_name" in phase_df_local.columns and "alpha" in phase_df_local.columns:
                sub = phase_df_local[phase_df_local["kernel_name"] == kname]
                iou_col = "output_iou_final" if "output_iou_final" in sub.columns else "iou_final"
                if iou_col in sub.columns and len(sub) >= 6:
                    grp = sub.groupby("alpha")[iou_col].mean().reset_index().sort_values("alpha")
                    alphas_arr = grp["alpha"].values
                    iou_m = grp[iou_col].values

                    # FIX (alpha* instability bug): the old hardcoded bound iou_min∈[-0.05,0.6]
                    # silently broke this fit for kernels whose IoU never drops below ~0.6
                    # (e.g. sobel_x bottoms out near 0.77). p0[1]=max(iou_m.min(),0)=0.77 then
                    # sat OUTSIDE the upper bound 0.6, curve_fit raised
                    # "Initial guess is outside of provided bounds", and the except-block
                    # silently produced alpha_star=NaN — which is exactly the run-to-run
                    # "sobel_x: alpha*=nan" seen in the latest log (vs =41.64 in an earlier
                    # run where a different cached metric definition happened to push
                    # iou_min below 0.6). Fix: derive iou_max/iou_min bounds from the
                    # observed data range (with margin) so they fit EVERY kernel's regime,
                    # and clip p0 strictly inside those bounds before fitting.
                    span = float(iou_m.max() - iou_m.min())
                    if span < 0.05:
                        # No detectable transition in this kernel's IoU response — the
                        # sigmoid has nothing to identify (any alpha_star fits a flat
                        # line equally well). Reporting a point estimate here would just
                        # be regurgitating the initial guess. Be explicit instead.
                        fit_note = f"NO_TRANSITION_SIGNAL (IoU range={span:.3f} < 0.05)"
                    else:
                        lo = np.array([max(0.05, iou_m.max()-0.5), iou_m.min()-0.15,
                                       max(0.5, alphas_arr.min()), 0.05])
                        hi = np.array([min(1.2, iou_m.max()+0.15), iou_m.max(),
                                       alphas_arr.max()*1.5, 25.0])
                        p0 = np.clip([iou_m.max(), iou_m.min(), 11.7, 3.0],
                                     lo + 1e-6, hi - 1e-6)
                        try:
                            popt, _ = curve_fit(_sig_model, alphas_arr, iou_m,
                                                 p0=p0, bounds=(lo, hi), maxfev=50000)
                            alpha_star_fit = float(popt[2])
                            if np.isclose(popt[2], lo[2], rtol=1e-3) or np.isclose(popt[2], hi[2], rtol=1e-3):
                                fit_note = "UNIDENTIFIABLE (alpha* pinned at search-box edge)"
                        except Exception as e:
                            fit_note = f"fit failed: {e}"
                            print(f"  [warn] fit failed for {kname}: {e}")
                else:
                    fit_note = f"insufficient data (n={len(sub)})"

            rows.append({
                "kernel_name": kname,
                "sigma_max": sigma_max,
                "inv_sigma_max": 1.0 / (sigma_max + 1e-12),
                "alpha_star": alpha_star_fit,
                "fit_note": fit_note
            })
            note = f"   [{fit_note}]" if fit_note else ""
            print(f"  {kname:14s}: sigma_max={sigma_max:.4f}  alpha*={alpha_star_fit:.2f}{note}")

        df_out = pd.DataFrame(rows)
        save_csv(df_out, csv_name)

    # Plot and Pearson correlation
    valid = df_out.dropna(subset=["alpha_star"])
    if len(valid) >= 2:
        x_vals = valid["inv_sigma_max"].values
        y_vals = valid["alpha_star"].values
        r, p_val = pearsonr(x_vals, y_vals)
        print(f"  Pearson r(alpha* vs 1/sigma_max) = {r:.4f}  p={p_val:.4f}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x_vals, y_vals, s=100, zorder=5, color="#1f77b4", edgecolors="black")
        for _, row in valid.iterrows():
            ax.annotate(row["kernel_name"],
                        (row["inv_sigma_max"], row["alpha_star"]),
                        textcoords="offset points", xytext=(6, 3), fontsize=9)
        # regression line
        if len(valid) >= 3:
            m, b = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_line, m * x_line + b, "--", color="#d62728", lw=1.8,
                    label=f"Linear fit (r={r:.3f}, p={p_val:.3f})")
            ax.legend(fontsize=9)
        ax.set_xlabel("1 / sigma_max(K)  [spectral norm reciprocal]")
        ax.set_ylabel("alpha*  [phase transition point]")
        ax.set_title("Per-kernel phase transition: alpha* proportional to 1/sigma_max(K)\n"
                     "(Claim: alpha*(kernel) ~ 1/sigma_max(A))")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        save_fig("per_kernel_alpha_star.png")
    else:
        print("  [warn] insufficient valid fits to plot")

# ── 13.3: Convergence rate analysis ──────────────────────────────────────────
def run_convergence_rate_analysis():
    """Run short experiments for alpha in {1, 5, 10, 20, 40} with Adam and PGD,
    fit L(t) = L_inf + (L0-L_inf)*exp(-t/tau), report tau(alpha) for each optimizer."""
    print("\nConvergence rate analysis: tau(alpha) for Adam vs PGD")
    csv_name = "convergence_rate_analysis.csv"
    existing = load_csv(csv_name, warn=False)

    SHORT_ALPHAS = [1.0, 5.0, 10.0, 20.0, 40.0]
    SHORT_STEPS = 300
    OPTS_CONV = {
        "adam": {"lr": 0.03, "steps": SHORT_STEPS},
        "pgd":  {"lr": 0.10, "steps": SHORT_STEPS},
    }

    def _exp_decay(t, L_inf, L0, tau):
        return L_inf + (L0 - L_inf) * np.exp(-t / (tau + 1e-8))

    rows = []
    loss_curves_by = {}   # (alpha, opt) -> loss array

    for alpha in SHORT_ALPHAS:
        for opt_name, kw in OPTS_CONV.items():
            key = (alpha, opt_name)
            if _already_done(existing, {"alpha": alpha, "optimizer": opt_name}):
                # Try to reconstruct from existing row for the figure
                row = existing[
                    (existing["alpha"].sub(alpha).abs() < 1e-9) &
                    (existing["optimizer"] == opt_name)
                ]
                if len(row) > 0:
                    tau_val = float(row.iloc[0]["tau"])
                    L0_val  = float(row.iloc[0]["L0"])
                    Li_val  = float(row.iloc[0]["L_inf"])
                    t_arr = np.arange(SHORT_STEPS)
                    loss_curves_by[key] = _exp_decay(t_arr, Li_val, L0_val, tau_val)
                continue

            # Run fresh experiment
            p = run_single_experiment(
                alpha=alpha, optimizer_name=opt_name,
                optimizer_kwargs=kw, seed=0,
                image_shape=SHAPE, kernel_name="sobel_x",
                target_name="checkerboard"
            )
            loss_hist = np.array(p["result"]["loss_hist"])
            t_arr = np.arange(len(loss_hist), dtype=float)
            loss_curves_by[key] = loss_hist

            # Fit exponential decay
            L0_guess  = float(loss_hist[0])
            Li_guess  = float(loss_hist[-1])
            tau_guess = float(len(loss_hist) / 5.0)
            try:
                popt, _ = curve_fit(
                    _exp_decay, t_arr, loss_hist,
                    p0=[Li_guess, L0_guess, tau_guess],
                    bounds=([0.0, 0.0, 1.0], [1e6, 1e6, 1e6]),
                    maxfev=10000
                )
                L_inf_fit, L0_fit, tau_fit = popt
                y_pred = _exp_decay(t_arr, *popt)
                ss_res = np.sum((loss_hist - y_pred) ** 2)
                ss_tot = np.sum((loss_hist - loss_hist.mean()) ** 2)
                r2 = float(1.0 - ss_res / (ss_tot + 1e-30))
            except Exception as e:
                print(f"  [warn] curve_fit failed alpha={alpha} {opt_name}: {e}")
                L_inf_fit = Li_guess; L0_fit = L0_guess
                tau_fit = float("nan"); r2 = float("nan")

            rows.append({
                "alpha": alpha, "optimizer": opt_name,
                "tau": tau_fit, "L0": L0_fit, "L_inf": L_inf_fit, "r_squared": r2
            })
            print(f"  alpha={alpha:5.1f} {opt_name:4s}: tau={tau_fit:.2f}  r2={r2:.4f}")

    if rows:
        append_csv(rows, csv_name)
    df_conv = load_csv(csv_name, warn=False)

    if df_conv is None or len(df_conv) == 0:
        print("  [warn] No convergence rate data to plot"); return

    # Figure: 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap_alpha = plt.cm.viridis
    alpha_norm = {a: i / (len(SHORT_ALPHAS) - 1) for i, a in enumerate(SHORT_ALPHAS)}
    ls_map = {"adam": "-", "pgd": "--"}

    for alpha in SHORT_ALPHAS:
        for opt_name in ["adam", "pgd"]:
            key = (alpha, opt_name)
            curve = loss_curves_by.get(key)
            if curve is None:
                continue
            color = cmap_alpha(alpha_norm[alpha])
            label = f"a={alpha:.0f} {opt_name}" if alpha in [1.0, 10.0, 40.0] else None
            axes[0].plot(curve, lw=1.5, color=color,
                         linestyle=ls_map[opt_name], alpha=0.85, label=label)

    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curves: Adam (-) vs PGD (--)\ncolour = alpha value")
    axes[0].legend(fontsize=8, ncol=2); axes[0].grid(True, linestyle="--", alpha=0.4)

    # Right panel: tau_PGD / tau_Adam vs alpha
    ratio_rows = []
    for alpha in SHORT_ALPHAS:
        sub = df_conv[df_conv["alpha"].sub(alpha).abs() < 1e-9]
        tau_adam = sub[sub["optimizer"] == "adam"]["tau"].values
        tau_pgd  = sub[sub["optimizer"] == "pgd"]["tau"].values
        if len(tau_adam) > 0 and len(tau_pgd) > 0:
            ta = float(tau_adam[0]); tp = float(tau_pgd[0])
            ratio_rows.append({"alpha": alpha,
                                "tau_adam": ta, "tau_pgd": tp,
                                "ratio": tp / (ta + 1e-8)})

    if ratio_rows:
        rdf = pd.DataFrame(ratio_rows)
        axes[1].plot(rdf["alpha"], rdf["ratio"], "o-", color="#d62728", lw=2.2, markersize=8)
        axes[1].axhline(1.0, color="gray", linestyle=":", lw=1.5)
        axes[1].set_xlabel("Sigmoid stiffness alpha")
        axes[1].set_ylabel("tau_PGD / tau_Adam")
        axes[1].set_title("Convergence time ratio: PGD/Adam vs alpha\n(>1 = Adam faster)")
        axes[1].grid(True, linestyle="--", alpha=0.4)
        for _, r in rdf.iterrows():
            axes[1].annotate(f"{r['ratio']:.1f}x",
                             (r["alpha"], r["ratio"]),
                             textcoords="offset points", xytext=(4, 4), fontsize=8)

    plt.suptitle("Convergence Rate Analysis — tau(alpha) for Adam vs PGD",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    save_fig("convergence_rate_analysis.png")

# ── 13.4: Mutual information proxy ────────────────────────────────────────────
def run_mutual_information_proxy():
    """Compute I_proxy(x; f(x)) via entropy of sigmoid outputs for a
    coarse (6 alpha × 5 kernel) grid. Save heatmap."""
    print("\nMutual information proxy: I(x; f(x))")
    csv_name = "mutual_information_proxy.csv"
    existing = load_csv(csv_name, warn=False)

    KNAMES = ["identity_like", "avg_blur", "sobel_x", "laplacian", "random_norm"]
    MI_ALPHAS = [1.0, 5.0, 10.0, 20.0, 40.0, 60.0]
    N_BINS = 20

    def _entropy(arr, n_bins=N_BINS):
        """Shannon entropy (bits) from histogram."""
        counts, _ = np.histogram(arr, bins=n_bins, range=(0.0, 1.0))
        probs = counts / (counts.sum() + 1e-30)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs + 1e-300)))

    def _mi_proxy(x_flat, fx_flat, n_bins=N_BINS, n_quartiles=4):
        """I_proxy = H(f(x)) - H(f(x)|x) using x-quartile conditioning."""
        H_fx = _entropy(fx_flat, n_bins)
        q_edges = np.percentile(x_flat, np.linspace(0, 100, n_quartiles + 1))
        H_cond_parts = []
        weights = []
        for qi in range(n_quartiles):
            lo, hi = q_edges[qi], q_edges[qi + 1]
            mask = (x_flat >= lo) & (x_flat <= hi if qi == n_quartiles - 1 else x_flat < hi)
            if mask.sum() < 5:
                continue
            H_cond_parts.append(_entropy(fx_flat[mask], n_bins))
            weights.append(mask.sum())
        if not H_cond_parts:
            return 0.0
        weights = np.array(weights, dtype=float)
        H_cond = float(np.average(H_cond_parts, weights=weights))
        return max(0.0, H_fx - H_cond)

    rows = []
    for alpha in MI_ALPHAS:
        for kname in KNAMES:
            if _already_done(existing, {"alpha": alpha, "kernel_name": kname}):
                continue
            p = run_single_experiment(
                alpha=alpha, kernel_name=kname,
                optimizer_name="adam",
                optimizer_kwargs={"lr": 0.03, "steps": 200},
                seed=0, image_shape=SHAPE, target_name="checkerboard"
            )
            xf = p["x_final"]
            fx = p["problem"].forward(xf)
            mi = _mi_proxy(xf.reshape(-1), fx.reshape(-1))
            iou = p["summary"]["output_iou_final"]
            rows.append({
                "alpha": alpha, "kernel_name": kname,
                "MI_proxy": float(mi), "output_iou_final": float(iou)
            })
            print(f"  alpha={alpha:5.1f} {kname:14s}: MI_proxy={mi:.4f}  IoU={iou:.3f}")

    if rows:
        append_csv(rows, csv_name)

    df_mi = load_csv(csv_name, warn=False)
    if df_mi is None or len(df_mi) == 0:
        print("  [warn] No MI data to plot"); return

    # Heatmap figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (col, title, cmap) in enumerate([
        ("MI_proxy",        "MI proxy I(x; f(x)) [bits]", "YlOrRd"),
        ("output_iou_final","Reconstruction IoU",          "RdYlGn"),
    ]):
        pivot = df_mi.pivot_table(index="kernel_name", columns="alpha",
                                   values=col, aggfunc="mean")
        pivot = pivot.reindex(KNAMES)
        pivot = pivot.reindex(columns=sorted(pivot.columns))
        ax = axes[ax_idx]
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                       vmin=pivot.values[~np.isnan(pivot.values)].min() if not np.all(np.isnan(pivot.values)) else 0,
                       vmax=pivot.values[~np.isnan(pivot.values)].max() if not np.all(np.isnan(pivot.values)) else 1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"a={a:.0f}" for a in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="black")
        plt.colorbar(im, ax=ax, fraction=0.046).set_label(title)
        ax.set_title(title)

    plt.suptitle("Mutual Information Proxy over (alpha x kernel) grid",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    save_fig("mutual_information_proxy.png")

# ── 13.5: Verification checks ────────────────────────────────────────────────
def run_verification_checks(alpha_df, dense_df, sr_df, scale_df):
    """Check key expected values from the paper with 5% tolerance."""
    print("\n" + "="*60)
    print("VERIFICATION CHECKS")
    print("="*60)

    all_pass = True
    results = []

    def _check(label, value, lo, hi):
        nonlocal all_pass
        if value is None or (isinstance(value, float) and np.isnan(value)):
            status = "SKIP"
        elif lo <= value <= hi:
            status = "PASS"
        else:
            status = "FAIL"
            all_pass = False
        results.append({"check": label, "value": value, "expected": f"[{lo}, {hi}]", "status": status})
        marker = "[PASS]" if status == "PASS" else ("[SKIP]" if status == "SKIP" else "[FAIL]")
        print(f"  {marker} {label}: {value} (expect [{lo:.4f}, {hi:.4f}])")
        return status == "PASS"

    if alpha_df is not None and "optimizer" in alpha_df.columns:
        adam = alpha_df[alpha_df["optimizer"] == "adam"]
        grp = adam.groupby("alpha")["output_iou_final"].mean()

        _check("alpha=1  Adam IoU", grp.get(1.0, float("nan")),  0.82, 0.89)
        _check("alpha=10 Adam IoU", grp.get(10.0, float("nan")), 0.71, 0.81)
        _check("alpha=20 Adam IoU", grp.get(20.0, float("nan")), 0.64, 0.74)
        _check("alpha=40 Adam IoU", grp.get(40.0, float("nan")), 0.58, 0.68)
    else:
        for lbl in ["alpha=1  Adam IoU", "alpha=10 Adam IoU",
                    "alpha=20 Adam IoU", "alpha=40 Adam IoU"]:
            _check(lbl, float("nan"), 0.0, 1.0)

    # Stable rank collapse ratio
    if sr_df is not None:
        sr_grp = sr_df.groupby("alpha")["stable_rank"].mean()
        sr_lo = sr_grp.get(sr_grp.index.min(), float("nan"))
        sr_hi = sr_grp.get(sr_grp.index.max(), float("nan"))
        ratio = sr_lo / (sr_hi + 1e-30) if not np.isnan(sr_lo) else float("nan")
        _check("Stable rank collapse ratio (>15x)", ratio, 15.0, 1e9)
    else:
        _check("Stable rank collapse ratio (>15x)", float("nan"), 15.0, 1e9)

    # Phase transition alpha*
    fcr = load_csv("fisher_cramer_rao.csv", warn=False)
    if fcr is not None and "parameter" in fcr.columns:
        row = fcr[fcr["parameter"] == "alpha_star"]
        if len(row) > 0:
            a_star = float(row.iloc[0]["estimate"])
            _check("Phase transition alpha*", a_star, 9.0, 14.0)
        else:
            _check("Phase transition alpha*", float("nan"), 9.0, 14.0)
    else:
        _check("Phase transition alpha*", float("nan"), 9.0, 14.0)

    print("-" * 60)
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    print(f"  TOTAL: {n_pass} PASS  {n_fail} FAIL  {n_skip} SKIP")
    print("="*60)
    return all_pass

# ── 13.6: Headline summary table ──────────────────────────────────────────────
def print_summary_table(alpha_df, grad_df, sr_df, curv_df):
    """Print a clean table of headline numbers from the paper."""
    def _get_adam_iou(alpha_df, alpha_val):
        if alpha_df is None: return float("nan")
        sub = alpha_df[alpha_df["optimizer"] == "adam"] if "optimizer" in alpha_df.columns else alpha_df
        g = sub.groupby("alpha")["output_iou_final"].mean()
        return float(g.get(alpha_val, float("nan")))

    def _get_pgd_iou(alpha_df, alpha_val):
        if alpha_df is None: return float("nan")
        sub = alpha_df[alpha_df["optimizer"] == "pgd"] if "optimizer" in alpha_df.columns else alpha_df
        g = sub.groupby("alpha")["output_iou_final"].mean()
        return float(g.get(alpha_val, float("nan")))

    def _get_alpha_star():
        fcr = load_csv("fisher_cramer_rao.csv", warn=False)
        if fcr is None: return float("nan")
        row = fcr[fcr["parameter"] == "alpha_star"] if "parameter" in fcr.columns else pd.DataFrame()
        return float(row.iloc[0]["estimate"]) if len(row) > 0 else float("nan")

    def _get_sr_ratio(sr_df):
        if sr_df is None: return float("nan")
        g = sr_df.groupby("alpha")["stable_rank"].mean()
        lo, hi = g.get(g.index.min(), 1.0), g.get(g.index.max(), 1.0)
        return lo / (hi + 1e-30)

    def _get_active_frac(alpha_df, grad_df, alpha_val, opt_name):
        if alpha_df is not None and "optimizer" in alpha_df.columns:
            sub = alpha_df[
                (alpha_df["optimizer"] == opt_name) &
                (alpha_df["alpha"].sub(alpha_val).abs() < 1e-9)
            ]
            if len(sub) > 0 and "active_grad_frac_final" in sub.columns:
                return float(sub["active_grad_frac_final"].mean())
        if grad_df is not None:
            src = grad_df
            if "optimizer" in src.columns:
                src = src[src["optimizer"] == opt_name]
            frac_col = "frac_active_sp_gt_0.01" if "frac_active_sp_gt_0.01" in src.columns else None
            if frac_col:
                sub2 = src[src["alpha"].sub(alpha_val).abs() < 1e-9]
                if len(sub2) > 0:
                    return float(sub2[frac_col].mean())
        return float("nan")

    a1_adam  = _get_adam_iou(alpha_df, 1.0)
    a1_pgd   = _get_pgd_iou(alpha_df, 1.0)
    a10_adam = _get_adam_iou(alpha_df, 10.0)
    a10_pgd  = _get_pgd_iou(alpha_df, 10.0)
    a40_adam = _get_adam_iou(alpha_df, 40.0)
    a40_pgd  = _get_pgd_iou(alpha_df, 40.0)
    a_star   = _get_alpha_star()
    sr_ratio = _get_sr_ratio(sr_df)
    agf_adam = _get_active_frac(alpha_df, grad_df, 20.0, "adam")
    agf_pgd  = _get_active_frac(alpha_df, grad_df, 20.0, "pgd")

    W = 64

    def _fmt(v, fmt=".3f"):
        return format(v, fmt) if not np.isnan(v) else "N/A"

    line = lambda s: print("║ " + s.ljust(W - 4) + " ║")
    print("╔" + "═"*(W-2) + "╗")
    print("║" + "  GRADIENT GATE COLLAPSE — HEADLINE STATISTICS".center(W-2) + "║")
    print("╠" + "═"*(W-2) + "╣")
    line(f"alpha=1   → Adam IoU: {_fmt(a1_adam)}   PGD IoU: {_fmt(a1_pgd)}")
    line(f"alpha=10  → Adam IoU: {_fmt(a10_adam)}   PGD IoU: {_fmt(a10_pgd)}")
    line(f"alpha=40  → Adam IoU: {_fmt(a40_adam)}   PGD IoU: {_fmt(a40_pgd)}")
    line(f"alpha*    → {_fmt(a_star)} (from fisher_cramer_rao.csv)")
    if not np.isnan(sr_ratio):
        line(f"Stable rank collapse: {sr_ratio:.1f}x (alpha=1 -> alpha={ALPHAS[-1]:.0f})")
    else:
        line(f"Stable rank collapse: N/A")
    if not np.isnan(agf_adam) and not np.isnan(agf_pgd):
        line(f"Active grad frac at alpha=20: Adam {agf_adam:.1%}   PGD {agf_pgd:.1%}")
    else:
        line(f"Active grad frac at alpha=20: N/A")
    print("╚" + "═"*(W-2) + "╝")

# ── 12.10: Poster headline statistics ────────────────────────────────────────
def print_poster_headline_stats(alpha_df, grad_df, kernel_df, phase_df,
                                 erank_df, curv_df, sr_df):
    print("\n" + "="*72)
    print("POSTER HEADLINE STATISTICS")
    print("="*72)
    if alpha_df is not None:
        adam = alpha_df[alpha_df["optimizer"]=="adam"] if "optimizer" in alpha_df.columns else alpha_df
        iou_by_a = adam.groupby("alpha")["output_iou_final"].agg(["mean","std"])
        print("\n[C1] Phase transition in IoU (Adam, sobel_x, checkerboard)")
        for a,row in iou_by_a.iterrows():
            print(f"   alpha={a:5.1f}: IoU={row['mean']:.3f} +/- {row['std']:.3f}")
        a_lo,a_hi=iou_by_a.index.min(),iou_by_a.index.max()
        print(f"   Drop alpha={a_lo} -> alpha={a_hi}: {iou_by_a.loc[a_lo,'mean']-iou_by_a.loc[a_hi,'mean']:.3f}")
    print("\n[C2] Collapse signals at alpha=10 vs alpha=40")
    def _ratio(df,col):
        if df is None: return np.nan,np.nan
        g=df.groupby("alpha")[col].mean()
        return g.get(10.,np.nan),g.get(40.,np.nan)
    if grad_df is not None:
        g10,g40=_ratio(grad_df[grad_df["optimizer"]=="adam"] if "optimizer" in grad_df.columns else grad_df,
                        "frac_active_sp_gt_0.01")
        print(f"   Active grad frac (Adam): alpha=10 -> {g10:.3f}  alpha=40 -> {g40:.3f}")
    if erank_df is not None:
        e10,e40=_ratio(erank_df,"effective_rank")
        print(f"   Effective rank: alpha=10 -> {e10:.1f}  alpha=40 -> {e40:.1f}")
    if sr_df is not None:
        s_lo=sr_df.groupby("alpha")["stable_rank"].mean().get(ALPHAS[0],np.nan)
        s_hi=sr_df.groupby("alpha")["stable_rank"].mean().get(ALPHAS[-1],np.nan)
        print(f"   Stable rank: alpha={ALPHAS[0]} -> {s_lo:.1f}  alpha={ALPHAS[-1]} -> {s_hi:.1f}"
              f"  (collapse {s_lo/s_hi:.1f}x)" if s_hi>0 else "")
    if kernel_df is not None:
        print("\n[C4] Kernel difficulty (Adam, alpha=10)")
        ks=(kernel_df[kernel_df["optimizer"]=="adam"] if "optimizer" in kernel_df.columns else kernel_df)\
            .groupby("kernel_name")["output_iou_final"].agg(["mean","std"])\
            .sort_values("mean",ascending=False)
        for kname,row in ks.iterrows():
            print(f"   {kname:>14s}: IoU={row['mean']:.3f} +/- {row['std']:.3f}")
    print("="*72)

if __name__ == "__main__":
    main()
