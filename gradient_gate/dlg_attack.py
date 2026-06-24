"""Phase 5: real gradient-inversion privacy attacks (DLG / iDLG), and a
victim model whose gate collapse can be swept continuously so the central
Phase 5 question — does gate collapse PREDICT gradient-leakage attack
success? — can be tested directly, rather than via the IoU-threshold proxy
in run_experiments.py's ext_e_gradient_leakage().

References:
  Zhu, Liu, Han (2019) "Deep Leakage from Gradients" (DLG) — NeurIPS.
  Zhao, Mopuri, Bilen (2020) "iDLG: Improved Deep Leakage from Gradients" —
    fixes the private label analytically instead of also optimizing a soft
    label, which is both more accurate and removes a confound when
    correlating reconstruction quality against gate density.
"""
import math

import torch
import torch.nn as nn


class AlphaGateClassifier(nn.Module):
    """Small classifier with ONE tunable-alpha sigmoid gate layer
    (sigma(z;alpha,c) = sigmoid(alpha*(z-c)), the exact parameterization used
    throughout run_experiments.py) sandwiched between two conv layers. Lets
    the privacy experiment sweep gate collapse (alpha) on an otherwise fixed
    architecture/task and ask whether collapse predicts DLG/iDLG attack
    success."""

    def __init__(self, alpha: float = 1.0, c: float = 0.5, in_ch: int = 1,
                 num_classes: int = 10, img: int = 28):
        super().__init__()
        self.alpha = float(alpha)
        self.c = float(c)
        self.conv1 = nn.Conv2d(in_ch, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(12 * 4 * 4, num_classes)
        self.last_active_frac = None
        self.last_gate_mean = None

    def gate(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.sigmoid(self.alpha * (z - self.c))
        gamma = self.alpha * h * (1.0 - h)
        self.last_active_frac = float((gamma.abs() > 0.01).float().mean().item())
        self.last_gate_mean = float(gamma.abs().mean().item())
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.gate(self.conv1(x))
        h2 = torch.relu(self.conv2(h1))
        feat = self.pool(h2).flatten(1)
        return self.fc(feat)


def dlg_attack(model: nn.Module, x_true: torch.Tensor, y_true: torch.Tensor,
               n_classes: int = 10, iters: int = 150, lr: float = 1.0,
               use_idlg: bool = True, device: str = "cpu"):
    """Given the TRUE gradient of model's loss w.r.t. its parameters for one
    private (x_true, y_true) example, optimizes a dummy (x, y) pair so that
    its gradient matches the true gradient — recovering x_true without ever
    observing it (only its gradient contribution, as in federated learning).

    Returns (reconstructed_x, grad_distance_history).
    """
    model = model.to(device).train()
    x_true = x_true.to(device)
    y_true = y_true.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad()
    loss = nn.functional.cross_entropy(model(x_true), y_true)
    true_grads = [g.detach() for g in torch.autograd.grad(loss, params)]

    dummy_x = torch.randn_like(x_true, requires_grad=True, device=device)
    if use_idlg:
        # iDLG (Zhao et al., Eq. 5): under cross-entropy with a single
        # example, the true label is exactly recoverable as argmin of the
        # final Linear layer's weight-gradient row sums — no label
        # optimization needed, and it is exact (not approximate).
        last_w_grad = true_grads[-2]
        dummy_y = torch.tensor([int(torch.argmin(last_w_grad.sum(dim=1)).item())], device=device)
        opt_params = [dummy_x]
    else:
        dummy_y_logits = torch.randn(1, n_classes, requires_grad=True, device=device)
        opt_params = [dummy_x, dummy_y_logits]

    opt = torch.optim.LBFGS(opt_params, lr=lr)
    history = []

    def closure():
        opt.zero_grad()
        out = model(dummy_x)
        if use_idlg:
            d_loss = nn.functional.cross_entropy(out, dummy_y)
        else:
            d_loss = -(torch.log_softmax(dummy_y_logits, dim=1) *
                       torch.softmax(out, dim=1)).sum()
        d_grads = torch.autograd.grad(d_loss, params, create_graph=True)
        grad_diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(d_grads, true_grads))
        grad_diff.backward()
        return grad_diff

    for _ in range(iters):
        gd = opt.step(closure)
        history.append(float(gd.item()))

    return dummy_x.detach(), history


def inverting_gradients_attack(model: nn.Module, x_true: torch.Tensor, y_true: torch.Tensor,
                                iters: int = 150, lr: float = 0.1, tv_weight: float = 1e-4,
                                device: str = "cpu"):
    """Geiping et al. 2020 "Inverting Gradients — How easy is it to break
    privacy in federated learning?": a stronger gradient-inversion baseline
    than DLG/iDLG. Two key differences from dlg_attack():
      1. Cosine-similarity gradient-matching loss (scale-invariant across
         layers) instead of raw L2 distance — the paper's central claim is
         that L2 (DLG) is brittle to per-layer gradient-magnitude
         differences, while a single global cosine similarity over the
         flattened gradient vector is far more robust, especially for
         deeper networks.
      2. Adam + cosine LR schedule + a total-variation image prior, instead
         of LBFGS — matches the optimizer the paper found necessary for the
         cosine objective to converge reliably.
    Label recovered analytically via the same iDLG trick (exact for
    cross-entropy + a single example).
    """
    model = model.to(device).train()
    x_true = x_true.to(device)
    y_true = y_true.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad()
    loss = nn.functional.cross_entropy(model(x_true), y_true)
    true_grads = [g.detach() for g in torch.autograd.grad(loss, params)]
    true_flat = torch.cat([g.flatten() for g in true_grads])

    dummy_x = torch.randn_like(x_true, requires_grad=True, device=device)
    last_w_grad = true_grads[-2]
    dummy_y = torch.tensor([int(torch.argmin(last_w_grad.sum(dim=1)).item())], device=device)

    opt = torch.optim.Adam([dummy_x], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    history = []
    for _ in range(iters):
        opt.zero_grad()
        d_loss = nn.functional.cross_entropy(model(dummy_x), dummy_y)
        d_grads = torch.autograd.grad(d_loss, params, create_graph=True)
        d_flat = torch.cat([g.flatten() for g in d_grads])
        cos_sim = (d_flat * true_flat).sum() / (d_flat.norm() * true_flat.norm() + 1e-12)
        tv = ((dummy_x[:, :, 1:, :] - dummy_x[:, :, :-1, :]).abs().mean() +
              (dummy_x[:, :, :, 1:] - dummy_x[:, :, :, :-1]).abs().mean())
        total = (1.0 - cos_sim) + tv_weight * tv
        total.backward()
        opt.step()
        sched.step()
        history.append(float(total.item()))

    return dummy_x.detach(), history


def reconstruction_quality(x_true: torch.Tensor, x_recon: torch.Tensor) -> dict:
    """PSNR + Pearson correlation between the recovered and private image.
    DLG reconstructions are unconstrained (no [0,1] clipping enforced during
    optimization), so correlation is reported alongside PSNR since it is
    invariant to the affine scale/offset drift DLG reconstructions exhibit."""
    a = x_recon.detach().cpu().flatten()
    b = x_true.detach().cpu().flatten()
    mse = float(((a - b) ** 2).mean())
    ref_power = float((b ** 2).mean()) + 1e-12
    psnr = 10.0 * math.log10(max(ref_power, 1e-12) / max(mse, 1e-12))
    corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    return dict(mse=mse, psnr=psnr, corr=corr)
