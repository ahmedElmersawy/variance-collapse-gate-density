"""Generalized gate / gradient-flow / rank instrumentation for arbitrary
nn.Module architectures.

run_experiments.py's Phase 3A (AlphaResNet18 / AlphaVGG11) could only profile
the gradient gate Gamma = |f'(z)| because every ReLU had been replaced with a
custom AlphaSigmoid whose analytic derivative is known (see that file's
"MODELING DECISION" comment, lines ~2222-2241) — the conclusion only applies
to "ResNet-18/VGG-11-shaped sigmoid networks", not the canonical architectures.

This module measures the same quantity on REAL, off-the-shelf activations
(ReLU, GELU, SiLU, Sigmoid, Tanh, ELU, ...) with no architecture modification,
using one fact: for any elementwise activation y = f(x), a backward hook sees

    grad_input = grad_output * f'(x)        (elementwise chain rule)

so f'(x) = grad_input / grad_output recovers the exact pointwise gate
Gamma(x) = |f'(x)| without knowing f's analytic derivative. That is the same
quantity (active-gradient-fraction / gate density) used throughout the rest
of the gate-collapse theory, now measurable on a real ResNet/ViT/ConvNeXt.

Caveat: this trick requires f to act elementwise with a single input and
single output tensor. It does not apply to attention softmax, LayerNorm, or
other non-elementwise ops — those are out of scope for "gate" by this
project's own definition (Gamma is defined per-unit, not per-layer-mixing).
"""
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

GATE_EPS = 0.01          # matches the active_grad_fraction threshold used elsewhere in the project
GRAD_RATIO_EPS = 1e-12   # numerical floor for the grad_input / grad_output division

ELEMENTWISE_ACT_TYPES = (
    nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.ELU,
    nn.LeakyReLU, nn.Hardtanh, nn.Mish, nn.Softplus, nn.PReLU,
)


@dataclass
class LayerStats:
    name: str
    order: int
    active_frac: Optional[float] = None
    gate_mean: Optional[float] = None
    grad_norm: Optional[float] = None
    grad_sparsity: Optional[float] = None
    grad_entropy: Optional[float] = None
    effective_rank: Optional[float] = None
    stable_rank: Optional[float] = None
    singular_values: Optional[List[float]] = None


def effective_rank(matrix: torch.Tensor) -> float:
    """Entropy-based effective rank exp(-sum p_i log p_i) over the normalized
    singular-value distribution — the same continuous (threshold-free)
    definition used for the Jacobian effective-rank sweep elsewhere in the
    project (run_experiments.py run_effective_rank_sweep), applied here to a
    per-layer activation matrix instead of the full problem Jacobian."""
    s = torch.linalg.svdvals(matrix.float())
    s = s[s > 1e-12 * (s[0] + 1e-30)]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-300)).sum()))


def stable_rank(matrix: torch.Tensor) -> float:
    """||A||_F^2 / ||A||_2^2 (Rudelson & Vershynin) — a scale-invariant,
    perturbation-robust rank surrogate; cheaper and smoother than a hard
    threshold-rank and complementary to the entropy-based effective rank."""
    s = torch.linalg.svdvals(matrix.float())
    if s.numel() == 0 or s[0] <= 0:
        return 0.0
    return float((s ** 2).sum() / (s[0] ** 2))


def gradient_entropy(grad: torch.Tensor) -> float:
    """Shannon entropy of the normalized |gradient| distribution across
    elements. Complementary to active_frac: active_frac only counts elements
    above GATE_EPS, entropy captures how unevenly the surviving mass is
    spread across the units that DO carry signal."""
    g = grad.float().abs().flatten()
    total = g.sum()
    if total <= 0:
        return 0.0
    p = g / total
    return float(-(p * torch.log(p + 1e-300)).sum())


class GateInstrumentor:
    """Attaches forward+backward hooks to every elementwise-activation module
    in `model` and records, after one forward+backward pass:

      - active_frac / gate_mean    gradient gate Gamma = |f'(x)|, recovered
                                    via the grad_input/grad_output trick
      - grad_norm / grad_sparsity / grad_entropy   gradient-flow metrics
      - effective_rank / stable_rank / singular_values   of the layer's
        output activation matrix (rank-collapse metrics)

    Usage:
        instr = GateInstrumentor(model)
        loss = criterion(model(x), y)
        loss.backward()
        stats = instr.collect()   # List[LayerStats], forward execution order
        instr.remove()

    Inplace activations (torchvision's default, e.g. ReLU(inplace=True))
    corrupt the grad_input/grad_output trick because the input tensor is
    overwritten before autograd can attribute a clean grad_input to it — see
    gradient_gate.architectures._disable_inplace, which every build_model() call uses.
    """

    def __init__(self, model: nn.Module, act_types=ELEMENTWISE_ACT_TYPES,
                 compute_rank: bool = True, rank_max_cols: int = 512):
        self.model = model
        self.compute_rank = compute_rank
        self.rank_max_cols = rank_max_cols
        self._order = 0
        self._stats = {}
        self._handles = []

        for name, module in model.named_modules():
            if isinstance(module, act_types):
                self._stats[name] = LayerStats(name=name, order=-1)
                self._handles.append(module.register_forward_hook(self._make_fwd_hook(name)))
                self._handles.append(module.register_full_backward_hook(self._make_bwd_hook(name)))

    def _make_fwd_hook(self, name):
        def hook(module, inputs, output):
            st = self._stats[name]
            if st.order == -1:
                st.order = self._order
                self._order += 1
            if self.compute_rank and isinstance(output, torch.Tensor) and output.dim() >= 2:
                with torch.no_grad():
                    mat = output.detach().reshape(output.shape[0], -1)
                    if mat.shape[0] < 2:
                        return
                    if mat.shape[1] > self.rank_max_cols:
                        idx = torch.randperm(mat.shape[1], device=mat.device)[:self.rank_max_cols]
                        mat = mat[:, idx]
                    try:
                        s = torch.linalg.svdvals(mat.float())
                        st.singular_values = s.detach().cpu().tolist()
                        p = s[s > 1e-12 * (s[0] + 1e-30)]
                        if p.numel() > 0:
                            pn = p / p.sum()
                            st.effective_rank = float(torch.exp(-(pn * torch.log(pn + 1e-300)).sum()))
                            st.stable_rank = float((s ** 2).sum() / (s[0] ** 2 + 1e-30))
                    except Exception:
                        pass
        return hook

    def _make_bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            if not grad_input or grad_input[0] is None or not grad_output or grad_output[0] is None:
                return
            gi, go = grad_input[0].detach(), grad_output[0].detach()
            valid = go.abs() > GRAD_RATIO_EPS
            st = self._stats[name]
            if valid.any():
                gate = (gi[valid].abs() / go[valid].abs()).clamp(max=1e6)
                st.active_frac = float((gate > GATE_EPS).float().mean().item())
                st.gate_mean = float(gate.mean().item())
            st.grad_norm = float(go.norm().item())
            go_max = go.abs().max().clamp(min=1e-12)
            st.grad_sparsity = float((go.abs() < GATE_EPS * go_max).float().mean().item())
            st.grad_entropy = gradient_entropy(go)
        return hook

    def collect(self) -> List[LayerStats]:
        """Per-layer stats in true forward-execution order (hook firing
        order) — .named_modules() registration order is ambiguous for
        branching/residual topologies, same reasoning as
        run_experiments.py's _deepnet_gate_profile."""
        return sorted(self._stats.values(), key=lambda s: (s.order if s.order >= 0 else 10 ** 9))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
