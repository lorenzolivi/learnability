#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic: first-order approximation distortion (Eq. 10 vs Eq. 11)

Measures how well the diagonal first-order expansion of the gradient
contribution g_{t,ℓ} (Eq. 11) approximates the exact BPTT expression
(Eq. 10) that uses the full transport matrix M_{t,ℓ}.

For each architecture, we compute:
  - g_exact   = δ_t^T  M_{t,ℓ}  B_ℓ     (Eq. 10, full matrix product)
  - g_approx  = Σ_q μ^(q)_{t,ℓ} δ^(q)_t B^(q)_ℓ   (Eq. 11, diagonal approx)

and report the relative error  ‖g_exact − g_approx‖ / ‖g_exact‖  as a
function of lag ℓ.

DiagGate serves as a sanity check: its Jacobian is NOT exactly diagonal
(it has off-diagonal terms from Wh), so the first-order approximation
captures the diagonal part only.  The distortion quantifies the role of
off-diagonal recurrent mixing.

Designed to run on a MacBook (CPU, small H, moderate T).

For the full Appendix C sweep used in the paper, use the helper script:
  diagnostics/run_envelope_validation_batch.sh

Usage:
  python diagnose_first_order_distortion.py [--H 32] [--T 200] [--B 16] ...
"""

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def set_seed(seed: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def layernorm_if(enabled: bool, dim: int):
    return nn.LayerNorm(dim) if enabled else nn.Identity()


# ============================================================
# Data generation — temporally structured sinusoidal task
#
# Each input channel is a sum of sinusoids with random phases,
# giving smooth temporal structure the RNN must exploit.
# The target is a delayed, nonlinear combination of inputs:
#   y_t = Σ_k c_k tanh(u^T x_{t-ℓ_k}) + noise
# The tanh forces the model to form nonlinear temporal features.
# ============================================================

def make_dataset_cpu(Nseq: int, T: int, D: int,
                     task_lags: List[int],
                     task_coeffs: List[float],
                     noise_std: float = 0.01,
                     u_vec=None,
                     n_harmonics: int = 5,
                     base_period: float = 50.0):
    """
    Generate sinusoidal input sequences with a multi-lag regression target.

    Each input channel d is:  x_{t,d} = Σ_k A_k sin(2π t / P_k + φ_{k,d})
    where P_k are harmonically-related periods and φ are random phases
    (per sequence, per channel).

    Target: y_t = Σ_k c_k tanh(u^T x_{t-ℓ_k}) + ε_t
    """
    if u_vec is None:
        u = np.random.randn(D).astype(np.float32)
        u = u / (np.linalg.norm(u) + 1e-12)
    else:
        u = u_vec.astype(np.float32)

    # Time axis
    t_ax = np.arange(T, dtype=np.float32).reshape(1, T, 1)  # (1, T, 1)

    # Periods: base_period / k for k = 1..n_harmonics
    periods = np.array([base_period / k for k in range(1, n_harmonics + 1)],
                       dtype=np.float32)
    # Amplitudes: 1/k decay
    amplitudes = np.array([1.0 / k for k in range(1, n_harmonics + 1)],
                          dtype=np.float32)

    # Random phases per (sequence, harmonic, channel)
    phases = np.random.uniform(0, 2 * np.pi,
                               size=(Nseq, n_harmonics, D)).astype(np.float32)

    # Build input: X[n, t, d] = Σ_k A_k sin(2π t / P_k + φ_{n,k,d})
    X = np.zeros((Nseq, T, D), dtype=np.float32)
    for k in range(n_harmonics):
        # (Nseq, T, D) = A_k * sin(2π t / P_k + φ)
        arg = 2 * np.pi * t_ax / periods[k] + phases[:, k:k+1, :]  # broadcast
        X += amplitudes[k] * np.sin(arg)

    # Target: nonlinear delayed regression
    Y = np.zeros((Nseq, T, 1), dtype=np.float32)
    for k, lag in enumerate(task_lags):
        c = float(task_coeffs[k])
        if lag < T:
            proj = np.einsum("ntd,d->nt", X[:, :T - lag, :], u)
            Y[:, lag:, 0] += c * np.tanh(proj)

    Y += noise_std * np.random.randn(Nseq, T, 1).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y), u


# ============================================================
# Model definitions (copied from pipeline for self-containment)
# ============================================================

class ConstGateRNN(nn.Module):
    def __init__(self, D, H, s=0.1, ln=False):
        super().__init__()
        self.D, self.H = D, H
        self.Wx = nn.Linear(D, H)
        self.Wh = nn.Linear(H, H, bias=False)
        self.ln = layernorm_if(ln, H)
        self.out = nn.Linear(H, 1)
        s = float(np.clip(s, 1e-6, 1 - 1e-6))
        self.register_buffer("s_const", torch.tensor(s))
        nn.init.zeros_(self.Wx.bias)
        nn.init.zeros_(self.out.bias)

    def forward_with_intermediates(self, x, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        wh_diag = torch.diagonal(self.Wh.weight, 0) if return_intermediates else None
        s = self.s_const
        ys, hs, leaks, rdiags = [], [], [], []
        for t in range(T):
            h_prev = h
            pre = self.Wx(x[:, t]) + self.Wh(h_prev)
            pre = self.ln(pre)
            h_tilde = torch.tanh(pre)
            h = (1 - s) * h_prev + s * h_tilde
            ys.append(self.out(h))
            if return_intermediates:
                sH = s.expand(B, self.H)
                hs.append(h)
                leaks.append(1 - sH)
                rdiags.append((sH * (1 - h_tilde**2)) * wh_diag.view(1, -1))
        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        return y, torch.stack(hs, 1), {"leak": torch.stack(leaks, 1), "rdiag": torch.stack(rdiags, 1)}


class SharedGateRNN(nn.Module):
    def __init__(self, D, H, ln=False, init_s=0.005):
        super().__init__()
        self.D, self.H = D, H
        self.Wx = nn.Linear(D, H)
        self.Wh = nn.Linear(H, H, bias=False)
        self.ln_h = layernorm_if(ln, H)
        self.Ws = nn.Linear(D, 1, bias=True)
        self.Us = nn.Linear(H, 1, bias=False)
        self.Ws._skip_orth = True
        self.Us._skip_orth = True
        self.out = nn.Linear(H, 1)
        nn.init.zeros_(self.Wx.bias); nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.Ws.weight); nn.init.zeros_(self.Us.weight)
        init_s = float(np.clip(init_s, 1e-6, 1 - 1e-6))
        nn.init.constant_(self.Ws.bias, float(np.log(init_s / (1 - init_s))))

    def forward_with_intermediates(self, x, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        if return_intermediates:
            wh_diag = torch.diagonal(self.Wh.weight, 0)
            us_vec = self.Us.weight.view(-1)
        ys, hs, leaks, rdiags = [], [], [], []
        for t in range(T):
            h_prev = h
            s = torch.sigmoid(self.Ws(x[:, t]) + self.Us(h_prev))
            pre = self.Wx(x[:, t]) + self.Wh(h_prev)
            pre = self.ln_h(pre)
            h_tilde = torch.tanh(pre)
            sH = s.expand(B, self.H)
            h = (1 - sH) * h_prev + sH * h_tilde
            ys.append(self.out(h))
            if return_intermediates:
                leak = 1 - sH
                tp = 1 - h_tilde**2
                sp = (s * (1 - s)).expand(B, self.H)
                rdiag_gate = (h_tilde - h_prev) * (sp * us_vec.view(1, -1))
                rdiag_rec = (sH * tp) * wh_diag.view(1, -1)
                hs.append(h); leaks.append(leak); rdiags.append(rdiag_gate + rdiag_rec)
        y = torch.stack(ys, 1)
        if not return_intermediates:
            return y, None, None
        return y, torch.stack(hs, 1), {"leak": torch.stack(leaks, 1), "rdiag": torch.stack(rdiags, 1)}


class DiagGateRNN(nn.Module):
    def __init__(self, D, H, ln=False, init_s=0.005):
        super().__init__()
        self.D, self.H = D, H
        self.Wx = nn.Linear(D, H)
        self.Wh = nn.Linear(H, H, bias=False)
        self.ln_h = layernorm_if(ln, H)
        self.Ws = nn.Linear(D, H, bias=True)
        self.Us = nn.Linear(H, H, bias=False)
        self.Ws._skip_orth = True
        self.Us._skip_orth = True
        self.out = nn.Linear(H, 1)
        nn.init.zeros_(self.Wx.bias); nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.Ws.weight); nn.init.zeros_(self.Us.weight)
        init_s = float(np.clip(init_s, 1e-6, 1 - 1e-6))
        nn.init.constant_(self.Ws.bias, float(np.log(init_s / (1 - init_s))))

    def forward_with_intermediates(self, x, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        if return_intermediates:
            wh_diag = torch.diagonal(self.Wh.weight, 0)
            us_diag = torch.diagonal(self.Us.weight, 0)
        ys, hs, leaks, rdiags = [], [], [], []
        for t in range(T):
            h_prev = h
            s = torch.sigmoid(self.Ws(x[:, t]) + self.Us(h_prev))
            pre = self.Wx(x[:, t]) + self.Wh(h_prev)
            pre = self.ln_h(pre)
            h_tilde = torch.tanh(pre)
            h = (1 - s) * h_prev + s * h_tilde
            ys.append(self.out(h))
            if return_intermediates:
                leak = 1 - s
                tp = 1 - h_tilde**2
                sp = s * (1 - s)
                rdiag_gate = (h_tilde - h_prev) * (sp * us_diag.view(1, -1))
                rdiag_rec = (s * tp) * wh_diag.view(1, -1)
                hs.append(h); leaks.append(leak); rdiags.append(rdiag_gate + rdiag_rec)
        y = torch.stack(ys, 1)
        if not return_intermediates:
            return y, None, None
        return y, torch.stack(hs, 1), {"leak": torch.stack(leaks, 1), "rdiag": torch.stack(rdiags, 1)}


class GRUModel(nn.Module):
    def __init__(self, D, H, ln=False):
        super().__init__()
        self.D, self.H = D, H
        self.Wz, self.Uz = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wr, self.Ur = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wh, self.Uh = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.ln_h = layernorm_if(ln, H)
        self.out = nn.Linear(H, 1)
        nn.init.zeros_(self.out.bias)

    def forward_with_intermediates(self, x, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        if return_intermediates:
            uz_diag = torch.diagonal(self.Uz.weight)
            ur_diag = torch.diagonal(self.Ur.weight)
            uh_diag = torch.diagonal(self.Uh.weight)
        ys, hs = [], []
        z_l, r_l, leak_l, rdiag_l = [], [], [], []
        for t in range(T):
            h_prev = h
            z = torch.sigmoid(self.Wz(x[:, t]) + self.Uz(h_prev))
            r = torch.sigmoid(self.Wr(x[:, t]) + self.Ur(h_prev))
            g = torch.tanh(self.ln_h(self.Wh(x[:, t]) + self.Uh(r * h_prev)))
            h = (1 - z) * h_prev + z * g
            ys.append(self.out(h))
            if return_intermediates:
                zp = z * (1 - z); rp = r * (1 - r); gp = 1 - g**2
                rdiag = (g - h_prev) * zp * uz_diag + z * gp * uh_diag * (r + h_prev * rp * ur_diag)
                hs.append(h); z_l.append(z); r_l.append(r)
                leak_l.append(1 - z); rdiag_l.append(rdiag)
        y = torch.stack(ys, 1)
        if not return_intermediates:
            return y, None, None
        return y, torch.stack(hs, 1), {
            "z": torch.stack(z_l, 1), "r": torch.stack(r_l, 1),
            "leak": torch.stack(leak_l, 1), "rdiag": torch.stack(rdiag_l, 1)}


class LSTMModel(nn.Module):
    def __init__(self, D, H, ln=False):
        super().__init__()
        self.D, self.H = D, H
        self.Wi, self.Ui = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wf, self.Uf = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wo, self.Uo = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wg, self.Ug = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.ln_cand = layernorm_if(ln, H)
        self.out = nn.Linear(H, 1)
        nn.init.zeros_(self.out.bias)

    def forward_with_intermediates(self, x, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        c = torch.zeros(B, self.H, device=x.device)
        if return_intermediates:
            uf_diag = torch.diagonal(self.Uf.weight)
            ui_diag = torch.diagonal(self.Ui.weight)
            ug_diag = torch.diagonal(self.Ug.weight)
        ys, hs = [], []
        f_l, e_l, cd_l = [], [], []
        for t in range(T):
            h_prev, c_prev = h, c
            i = torch.sigmoid(self.Wi(x[:, t]) + self.Ui(h_prev))
            f = torch.sigmoid(self.Wf(x[:, t]) + self.Uf(h_prev))
            o = torch.sigmoid(self.Wo(x[:, t]) + self.Uo(h_prev))
            g = torch.tanh(self.ln_cand(self.Wg(x[:, t]) + self.Ug(h_prev)))
            c = f * c_prev + i * g
            tanh_c = torch.tanh(c)
            h = o * tanh_c
            ys.append(self.out(h))
            if return_intermediates:
                e = o * (1 - tanh_c**2)
                cdiag = c_prev * (f * (1 - f)) * uf_diag + i * (1 - g**2) * ug_diag + g * (i * (1 - i)) * ui_diag
                hs.append(h); f_l.append(f); e_l.append(e); cd_l.append(cdiag)
        y = torch.stack(ys, 1)
        if not return_intermediates:
            return y, None, None
        return y, torch.stack(hs, 1), {
            "forget": torch.stack(f_l, 1), "expr": torch.stack(e_l, 1),
            "cdiag": torch.stack(cd_l, 1)}


def apply_orthogonal(model):
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.weight is not None and m.weight.ndim == 2:
            if getattr(m, '_skip_orth', False):
                continue
            nn.init.orthogonal_(m.weight)


# ============================================================
# Exact Jacobian computation  (full H×H matrix per step)
# ============================================================

def compute_exact_jacobians(model, x_single):
    """
    Compute exact per-step Jacobians J_j = ∂h_j/∂h_{j-1} via torch.autograd.

    Args:
        model: RNN model instance
        x_single: (1, T, D) single-sequence input

    Returns:
        jacobians: list of T tensors, each (H, H).  jacobians[0] is J_1.
        hseq: (T, H) hidden state sequence
        delta: (T, H) output-projected error gradient ∂E/∂h_t
        y_hat: (T, 1) predictions
    """
    B, T, D = x_single.shape
    assert B == 1
    H = model.H
    device = x_single.device

    # We need gradients w.r.t. intermediate h states, so we run step by step
    # with torch.enable_grad and record the graph.
    model.eval()

    jacobians = []
    h_list = []

    # Determine model type
    is_lstm = isinstance(model, LSTMModel)
    is_gru = isinstance(model, GRUModel)
    is_const = isinstance(model, ConstGateRNN)
    is_shared = isinstance(model, SharedGateRNN)
    is_diag = isinstance(model, DiagGateRNN)

    if is_lstm:
        # LSTM: compute full 2H×2H stacked Jacobian ∂[h_t;c_t]/∂[h_{t-1};c_{t-1}]
        # then multiply these to get the exact transport, and extract h→h block.
        # This is necessary because the pipeline's diagonal approximation traces
        # the cell pathway: h_{t-ℓ} → cdiag → c chain via forget → expr → h_t.
        # Computing ∂h_t/∂h_{t-1} at fixed c_{t-1} would miss this pathway.
        h = torch.zeros(1, H, device=device)
        c = torch.zeros(1, H, device=device)

        for t in range(T):
            h_prev = h.detach().requires_grad_(True)
            c_prev = c.detach().requires_grad_(True)

            i = torch.sigmoid(model.Wi(x_single[:, t]) + model.Ui(h_prev))
            f = torch.sigmoid(model.Wf(x_single[:, t]) + model.Uf(h_prev))
            o = torch.sigmoid(model.Wo(x_single[:, t]) + model.Uo(h_prev))
            g = torch.tanh(model.ln_cand(model.Wg(x_single[:, t]) + model.Ug(h_prev)))
            c_new = f * c_prev + i * g
            tanh_c = torch.tanh(c_new)
            h_new = o * tanh_c

            # Compute full 2H×2H Jacobian ∂[h_t;c_t]/∂[h_{t-1};c_{t-1}]
            # Layout: rows 0..H-1 = h_t, rows H..2H-1 = c_t
            #         cols 0..H-1 = h_{t-1}, cols H..2H-1 = c_{t-1}
            J_full = torch.zeros(2 * H, 2 * H, device=device)
            for q in range(H):
                # ∂h_t[q]/∂h_{t-1} and ∂h_t[q]/∂c_{t-1}
                gh, gc = torch.autograd.grad(h_new[0, q], (h_prev, c_prev), retain_graph=True)
                J_full[q, :H] = gh[0]
                J_full[q, H:] = gc[0]
                # ∂c_t[q]/∂h_{t-1} and ∂c_t[q]/∂c_{t-1}
                gh2, gc2 = torch.autograd.grad(c_new[0, q], (h_prev, c_prev), retain_graph=True)
                J_full[H + q, :H] = gh2[0]
                J_full[H + q, H:] = gc2[0]

            jacobians.append(J_full)
            h_list.append(h_new.detach())

            h = h_new.detach()
            c = c_new.detach()

    elif is_gru:
        h = torch.zeros(1, H, device=device)

        for t in range(T):
            h_prev = h.detach().requires_grad_(True)

            z = torch.sigmoid(model.Wz(x_single[:, t]) + model.Uz(h_prev))
            r = torch.sigmoid(model.Wr(x_single[:, t]) + model.Ur(h_prev))
            g = torch.tanh(model.ln_h(model.Wh(x_single[:, t]) + model.Uh(r * h_prev)))
            h_new = (1 - z) * h_prev + z * g

            J = torch.zeros(H, H, device=device)
            for q in range(H):
                grad = torch.autograd.grad(h_new[0, q], h_prev, retain_graph=True)[0]
                J[q, :] = grad[0]

            jacobians.append(J)
            h_list.append(h_new.detach())
            h = h_new.detach()

    else:
        # Baseline gated RNNs (Const, Shared, Diag)
        h = torch.zeros(1, H, device=device)

        for t in range(T):
            h_prev = h.detach().requires_grad_(True)

            if is_const:
                s = model.s_const
                pre = model.Wx(x_single[:, t]) + model.Wh(h_prev)
                pre = model.ln(pre)
            elif is_shared:
                s = torch.sigmoid(model.Ws(x_single[:, t]) + model.Us(h_prev))
                pre = model.Wx(x_single[:, t]) + model.Wh(h_prev)
                pre = model.ln_h(pre)
            elif is_diag:
                s = torch.sigmoid(model.Ws(x_single[:, t]) + model.Us(h_prev))
                pre = model.Wx(x_single[:, t]) + model.Wh(h_prev)
                pre = model.ln_h(pre)

            h_tilde = torch.tanh(pre)
            if is_const:
                sH = s.expand(1, H)
            elif is_shared:
                sH = s.expand(1, H)
            else:
                sH = s
            h_new = (1 - sH) * h_prev + sH * h_tilde

            J = torch.zeros(H, H, device=device)
            for q in range(H):
                grad = torch.autograd.grad(h_new[0, q], h_prev, retain_graph=True)[0]
                J[q, :] = grad[0]

            jacobians.append(J)
            h_list.append(h_new.detach())
            h = h_new.detach()

    # Now run a clean forward pass for predictions and delta
    with torch.no_grad():
        y_hat_full, _, _ = model.forward_with_intermediates(x_single, return_intermediates=False)

    hseq = torch.stack(h_list, dim=0).squeeze(1)  # (T, H)
    y_hat = y_hat_full.squeeze(0)  # (T, 1)

    return jacobians, hseq, y_hat


# ============================================================
# Transport matrix M_{t,ell} = prod_{j=ell+1}^{t} J_j  (exact)
# ============================================================

@torch.no_grad()
def compute_exact_transport(jacobians, t_idx, ell, is_lstm=False):
    """
    Compute M_{t,ℓ} = ∏_{j=ℓ+1}^{t} J_j  (exact full-matrix product).

    Args:
        jacobians: list of Jacobians (H×H for baselines/GRU, 2H×2H for LSTM).
                   jacobians[j] = J_{j+1} (0-indexed: jacobians[0] = ∂state_1/∂state_0)
        t_idx: 0-based time index t
        ell: lag (code convention)
        is_lstm: if True, Jacobians are 2H×2H; extract h→h block from product.

    Returns: (H, H) transport matrix (h→h block for LSTM)
    """
    dim = jacobians[0].shape[0]
    device = jacobians[0].device

    start = t_idx - ell  # 0-based index of first Jacobian
    end = t_idx          # exclusive upper bound

    if start < 0 or end > len(jacobians):
        H = dim // 2 if is_lstm else dim
        return torch.eye(H, device=device, dtype=torch.float64)

    M = torch.eye(dim, device=device, dtype=torch.float64)
    for j in range(start, end):
        M = jacobians[j].double() @ M

    if is_lstm:
        # Extract h→h block (top-left H×H) from the 2H×2H product
        H = dim // 2
        return M[:H, :H]
    return M


# ============================================================
# Approximate transport: diagonal first-order (Eq. 11)
# ============================================================

@torch.no_grad()
def compute_approx_mu(model, intermediates, t_idx, ell):
    """
    Compute the diagonal first-order transport factor μ^(q)_{t,ℓ} for
    each hidden unit q, using the prefix-sum method from the pipeline.

    Returns: (H,) tensor of per-unit transport factors (zeroth + first order).
    """
    H = model.H

    is_lstm = isinstance(model, LSTMModel)

    if is_lstm:
        forget = intermediates["forget"][0]   # (T, H)
        expr   = intermediates["expr"][0]     # (T, H)
        cdiag  = intermediates["cdiag"][0]    # (T, H)

        # Zeroth order: e_t * prod_{j=t-ell+1}^{t} f_j
        # But for LSTM the zeroth order is: expr[t] * prod(forget[t-ell+1..t])
        # Actually the code computes:  mu0 = expr_end * prod_f
        # where expr_end = expr[:, ell:Tdg, :] and prod_f is the windowed product
        # For a single (t, ell): prod_f = prod(forget[t-ell+1 .. t])  (0-based: t-ell .. t-1)
        # Wait — let me be precise about indexing.

        # In the pipeline code (run_learnability_lstm_gru_DGX.py):
        #   prod_f = _win_prod_from_cs(cs_log_f, ell, ...) which computes
        #   exp(cs_log[:, ell+1:T+1, :] - cs_log[:, 1:T-ell+1, :])
        # This is the product of forget[t-ell .. t-1] for each output position.
        # For output position indexed by m (0-based), this is
        #   prod(forget[m .. m+ell-1])
        # The matched-stat convention aligns this so that output index m
        # corresponds to step t = m + ell (0-based).

        # For our single (t_idx, ell) pair:
        # t_idx is 0-based. The product is over steps t-ell .. t-1 (0-based).
        start_j = t_idx - ell
        end_j = t_idx

        if start_j < 0:
            return torch.zeros(H)

        prod_f = torch.ones(H, dtype=torch.float64)
        for j in range(start_j, end_j):
            prod_f *= forget[j].double()

        e_t = expr[t_idx].double()
        mu0 = e_t * prod_f

        # First order: mu0 * sum_{j=t-ell+1}^{t} (cdiag_j * e_shift_j / forget_j)
        # where e_shift_j = expr[j-1] (shifted by 1)
        # In the pipeline: ratio = cdiag * e_shift / forget64
        # e_shift[:, 1:, :] = expr[:, :-1, :]  (so e_shift[j] = expr[j-1])
        sum_ratio = torch.zeros(H, dtype=torch.float64)
        for j in range(start_j, end_j):
            e_shift_j = expr[j - 1].double() if j > 0 else torch.zeros(H, dtype=torch.float64)
            f_j = torch.clamp(forget[j].double(), 1e-12)
            sum_ratio += cdiag[j].double() * e_shift_j / f_j

        mu1 = mu0 * sum_ratio
        mu = mu0 + mu1
        return mu.float()

    else:
        # Baseline or GRU — all use leak/rdiag
        leak = intermediates["leak"][0]    # (T, H)
        rdiag = intermediates["rdiag"][0]  # (T, H)

        start_j = t_idx - ell
        end_j = t_idx

        if start_j < 0:
            return torch.zeros(H)

        # For GRU, there are also reset and eta terms
        is_gru = isinstance(model, GRUModel)

        # Zeroth order: prod(leak[start_j .. end_j-1])
        prod_leak = torch.ones(H, dtype=torch.float64)
        for j in range(start_j, end_j):
            prod_leak *= torch.clamp(leak[j].double(), 1e-12)

        if is_gru:
            r_vals = intermediates["r"][0]  # (T, H)
            prod_reset = torch.ones(H, dtype=torch.float64)
            prod_eta = torch.ones(H, dtype=torch.float64)
            for j in range(start_j, end_j):
                prod_reset *= torch.clamp(r_vals[j].double(), 1e-12)
                prod_eta *= torch.clamp((leak[j] * r_vals[j]).double(), 1e-12)
            mu0 = prod_leak + prod_reset + prod_eta
        else:
            mu0 = prod_leak

        # First order: mu0_leak * sum(rdiag/leak)  [baseline]
        # For GRU: gamma0 * sum(rdiag/leak)
        sum_ratio = torch.zeros(H, dtype=torch.float64)
        for j in range(start_j, end_j):
            lk = torch.clamp(leak[j].double(), 1e-12)
            sum_ratio += rdiag[j].double() / lk

        if is_gru:
            mu1 = prod_leak * sum_ratio
        else:
            mu1 = prod_leak * sum_ratio

        mu = mu0 + mu1
        return mu.float()


# ============================================================
# Envelope computation for a single (t, ell) pair
#
# The learnability theory depends on the envelope:
#   f(ℓ) = Σ_q |μ^(q)_{t,ℓ}|  where  μ^(q) = Λ^(q) Γ^(q)
#
# We compare:
#   f_exact(ℓ)  = Σ_q |Λ^(q) · [M_{t,ℓ}]_{qq}|   (full transport diagonal)
#   f_approx(ℓ) = Σ_q |Λ^(q) · Γ^(q)_{t,ℓ}|       (first-order expansion)
#
# The question is whether the decay PROFILE of the envelope is
# preserved (rank ordering, log-space shape), not pointwise accuracy.
# ============================================================


@torch.no_grad()
def compute_envelope_pair(model, jacobians, intermediates, t_idx, ell, Lambda_q):
    """
    Compute exact and approximate GELR envelopes for a single (t, ℓ).

    Returns dict with:
        f_exact:      Σ_q |Λ^(q) [M]_{qq}|  (full transport diagonal)
        f_approx:     Σ_q |Λ^(q) Γ^(q)|     (first-order expansion)
        offdiag_frac: ‖M - diag(diag(M))‖_F / ‖M‖_F
    """
    H = model.H
    is_lstm = isinstance(model, LSTMModel)
    Lambda_t = torch.from_numpy(Lambda_q).double()  # (H,)

    # Exact: diagonal of the full transport product
    M = compute_exact_transport(jacobians, t_idx, ell, is_lstm=is_lstm)  # (H, H)
    diag_M = torch.diag(M)
    gelr_exact = torch.abs(Lambda_t * diag_M)
    f_exact = gelr_exact.sum().item()

    # Approximate: first-order diagonal expansion
    Gamma_q = compute_approx_mu(model, intermediates, t_idx, ell).double()
    gelr_approx = torch.abs(Lambda_t * Gamma_q)
    f_approx = gelr_approx.sum().item()

    # Off-diagonal energy fraction of M
    M_norm = torch.norm(M).item()
    diag_mat = torch.diag(diag_M)
    offdiag_norm = torch.norm(M - diag_mat).item()
    offdiag_frac = offdiag_norm / max(M_norm, 1e-30)

    return {
        "f_exact": f_exact,
        "f_approx": f_approx,
        "offdiag_frac": offdiag_frac,
    }


# ============================================================
# Main diagnostic loop — envelope-level comparison
# ============================================================

def run_distortion_diagnostic(model, model_name, X, Y, lags, args,
                               optimizer=None, lr=1e-3):
    """
    For each lag ℓ, compare exact vs approximate GELR envelopes:
        f_exact(ℓ)  = ⟨Σ_q |Λ^(q) [M_{t,ℓ}]_{qq}|⟩_t   (full transport diagonal)
        f_approx(ℓ) = ⟨Σ_q |Λ^(q) Γ^(q)_{t,ℓ}|⟩_t      (first-order expansion)

    Focus is on TREND PRESERVATION: does the approximation reproduce the
    decay profile of the envelope, not pointwise values.

    Each lag contributes one point to the trend metrics:
      log10(mean_{samples} f_exact(t, ell))
      log10(mean_{samples} f_approx(t, ell))
    This matches Appendix C's "12 points (one per lag)" protocol.

    Returns: list of dicts with per-lag envelope statistics + trend metrics.
    """
    device = torch.device("cpu")
    model = model.to(device).eval()

    B_total, T, D = X.shape
    H = model.H

    # Compute adaptive base rates Λ^(q) from optimizer state
    if optimizer is not None:
        Lambda_q = compute_adaptive_base_rates(model, optimizer, lr)
    else:
        Lambda_q = np.full(H, lr, dtype=np.float64)
    log(f"  Λ^(q) stats: mean={Lambda_q.mean():.4e}, "
        f"std={Lambda_q.std():.4e}, min={Lambda_q.min():.4e}, max={Lambda_q.max():.4e}")

    rng = np.random.RandomState(int(getattr(args, "seed", 0)))

    n_seq = min(args.n_diag_sequences, B_total)
    n_t_per_seq = args.n_t_per_sequence
    if n_seq >= B_total:
        seq_indices = list(range(B_total))
    else:
        seq_indices = np.sort(rng.choice(B_total, size=n_seq, replace=False)).tolist()

    results = []

    for ell in lags:
        if ell < 1 or ell >= T - 1:
            continue

        log(f"  [{model_name}] lag ℓ={ell}")

        f_exact_list = []
        f_approx_list = []
        offdiag_fracs = []
        jac_diag_fracs = []

        for seq_idx in seq_indices:
            x_single = X[seq_idx:seq_idx+1].to(device)

            # Compute exact Jacobians for this sequence
            jacobians, hseq, y_hat = compute_exact_jacobians(model, x_single)

            # Get intermediates for approximate computation
            with torch.no_grad():
                _, _, intermediates = model.forward_with_intermediates(x_single, return_intermediates=True)

            # Measure per-step Jacobian diagonality (h→h block for LSTM)
            is_lstm_model = isinstance(model, LSTMModel)
            for j_idx in range(min(T, 20)):
                J = jacobians[j_idx]
                J_hh = J[:H, :H] if is_lstm_model else J
                diag_norm = torch.norm(torch.diag(J_hh)).item()
                full_norm = torch.norm(J_hh).item()
                if full_norm > 1e-12:
                    jac_diag_fracs.append(diag_norm / full_norm)

            # Sample time points for this lag
            valid_t = np.arange(ell + 1, T, dtype=np.int64)
            if valid_t.size == 0:
                continue
            if valid_t.size <= n_t_per_seq:
                t_samples = valid_t.tolist()
            else:
                t_samples = np.sort(rng.choice(valid_t, size=n_t_per_seq, replace=False)).tolist()

            for t_idx in t_samples:
                env = compute_envelope_pair(
                    model, jacobians, intermediates, t_idx, ell, Lambda_q)

                f_exact_list.append(env["f_exact"])
                f_approx_list.append(env["f_approx"])
                offdiag_fracs.append(env["offdiag_frac"])

        if len(f_exact_list) > 0:
            fe = np.array(f_exact_list)
            fa = np.array(f_approx_list)
            od = np.array(offdiag_fracs)
            jdf = np.array(jac_diag_fracs) if jac_diag_fracs else np.array([np.nan])

            # One lag-level point per envelope: log10 of the sample mean.
            # Appendix C correlations are defined on these 12 lag-level points,
            # not on the mean of per-sample logs.
            f_exact_mean = float(np.mean(fe))
            f_approx_mean = float(np.mean(fa))
            log_fe = float(np.log10(max(f_exact_mean, 1e-30)))
            log_fa = float(np.log10(max(f_approx_mean, 1e-30)))

            # Δlog₁₀: vertical shift in log space
            delta_log = log_fe - log_fa

            results.append({
                "lag": ell,
                "f_exact_mean": f_exact_mean,
                "f_approx_mean": f_approx_mean,
                "log10_f_exact": float(log_fe),
                "log10_f_approx": float(log_fa),
                "delta_log": float(delta_log),
                "offdiag_frac_mean": float(np.mean(od)),
                "offdiag_frac_std": float(np.std(od)),
                "jacobian_diag_frac": float(np.mean(jdf)),
                "n_samples": len(f_exact_list),
            })
            log(f"    ℓ={ell}: log₁₀(exact)={log_fe:.2f}  log₁₀(approx)={log_fa:.2f}  "
                f"Δlog₁₀={delta_log:.3f}  offdiag={np.mean(od):.4f}  (n={len(f_exact_list)})")
        else:
            log(f"    ℓ={ell}: no valid samples")

    # ---- Trend-preservation metrics across lags ----
    if len(results) >= 3:
        from scipy import stats as sp_stats

        lags_arr = np.array([r["lag"] for r in results])
        log_lags = np.log10(lags_arr)
        log_exact = np.array([r["log10_f_exact"] for r in results])
        log_approx = np.array([r["log10_f_approx"] for r in results])

        # Log-log slopes (effective power-law exponent of envelope decay)
        slope_exact, _, _, _, _ = sp_stats.linregress(log_lags, log_exact)
        slope_approx, _, _, _, _ = sp_stats.linregress(log_lags, log_approx)

        # Spearman rank correlation (monotonic ordering preserved?)
        rho, _ = sp_stats.spearmanr(log_exact, log_approx)

        # Pearson correlation on log-envelopes (decay shape similarity)
        pearson, _ = sp_stats.pearsonr(log_exact, log_approx)

        # Δlog₁₀ stability across lags
        delta_vec = log_exact - log_approx

        trend_summary = {
            "slope_exact": float(slope_exact),
            "slope_approx": float(slope_approx),
            "slope_ratio": float(slope_approx / slope_exact) if abs(slope_exact) > 1e-10 else float("nan"),
            "spearman": float(rho),
            "pearson": float(pearson),
            "delta_log_mean": float(np.mean(delta_vec)),
            "delta_log_std": float(np.std(delta_vec)),
        }

        log(f"\n  --- TREND METRICS [{model_name}] ---")
        log(f"  Log-log slopes: exact={slope_exact:.3f}  approx={slope_approx:.3f}  "
            f"(ratio={trend_summary['slope_ratio']:.3f})")
        log(f"  Spearman ρ={rho:.4f}   Pearson r={pearson:.4f}")
        log(f"  Δlog₁₀: mean={np.mean(delta_vec):.3f}  std={np.std(delta_vec):.3f}")

        # Attach trend summary to results
        for r in results:
            r["_trend"] = trend_summary

    return results


# ============================================================
# Adaptive base rates (ported from main pipeline)
# ============================================================

def compute_adaptive_base_rates(model, optimizer, lr, eps=1e-8, beta2=0.999):
    """
    Compute per-neuron adaptive base rates Λ^(q)_r via the row-mean
    Rayleigh-quotient projection.

    For Adam/AdamW: Λ^(q) = (1/H) Σ_j lr / (sqrt(v̂_{qj}) + ε)
    For RMSprop:    same with square_avg (no bias correction)
    For SGD:        uniform Λ^(q) = lr

    Returns: (H,) numpy array of per-neuron adaptive base rates.
    """
    H = model.H

    recurrent_params = []
    for name, param in model.named_parameters():
        if param.shape == (H, H) and "weight" in name and "out" not in name:
            recurrent_params.append((name, param))

    if not recurrent_params:
        return np.full(H, lr, dtype=np.float64)

    state = optimizer.state
    has_v = False
    for _, p in recurrent_params:
        if p in state:
            if "exp_avg_sq" in state[p] or "square_avg" in state[p]:
                has_v = True
                break

    if not has_v:
        return np.full(H, lr, dtype=np.float64)

    Lambda_per_matrix = []
    for name, param in recurrent_params:
        pstate = state.get(param, {})
        if "exp_avg_sq" in pstate:
            v = pstate["exp_avg_sq"]
            step = pstate.get("step", 1)
            if isinstance(step, torch.Tensor):
                step = step.item()
            step = max(int(step), 1)
            v_hat = v / (1.0 - beta2 ** step)
        elif "square_avg" in pstate:
            v_hat = pstate["square_avg"]
        else:
            continue

        if not torch.isfinite(v_hat).all():
            continue

        lam = lr / (torch.sqrt(v_hat.float()) + eps)
        row_mean = lam.mean(dim=1)
        Lambda_per_matrix.append(row_mean.detach().cpu().numpy())

    if not Lambda_per_matrix:
        return np.full(H, lr, dtype=np.float64)

    Lambda_q = np.mean(Lambda_per_matrix, axis=0).astype(np.float64)
    log(f"  [Λ^(q)] range [{Lambda_q.min():.4e}, {Lambda_q.max():.4e}], "
        f"mean={Lambda_q.mean():.4e}, max/min={Lambda_q.max()/max(Lambda_q.min(),1e-30):.2f}")
    return Lambda_q


# ============================================================
# Training (simple, for MacBook)
# ============================================================

def train_model_simple(model, X, Y, epochs=30, lr=1e-3, bs=32,
                       optimizer_name="adamw", momentum=0.9,
                       weight_decay=1e-4, rmsprop_alpha=0.99):
    """Quick training loop. Returns (model, optimizer) so we can extract Λ."""
    device = torch.device("cpu")
    model = model.to(device).train()

    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0,
                              weight_decay=weight_decay)
    elif optimizer_name == "sgd_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=rmsprop_alpha,
                                  momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    B_total = X.shape[0]

    for ep in range(epochs):
        perm = torch.randperm(B_total)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, B_total, bs):
            idx = perm[i:i+bs]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)
            y_hat, _, _ = model.forward_with_intermediates(xb, return_intermediates=False)
            loss = F.mse_loss(y_hat, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"  epoch {ep+1}/{epochs}: loss={total_loss/n_batches:.6f}")

    model.eval()
    return model, opt


# ============================================================
# CLI and main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="First-order approximation distortion diagnostic")
    p.add_argument("--H", type=int, default=32, help="Hidden size")
    p.add_argument("--D", type=int, default=8, help="Input dimension")
    p.add_argument("--T", type=int, default=200, help="Sequence length")
    p.add_argument("--B", type=int, default=64, help="Number of sequences for training")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--const_s", type=float, default=0.1, help="Gate init for baselines")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--n_diag_sequences", type=int, default=4,
                   help="Sequences sampled for the diagnostic (Appendix C uses 16)")
    p.add_argument("--n_t_per_sequence", type=int, default=20,
                   help="Time samples drawn per sequence and lag (Appendix C uses 50)")
    p.add_argument("--models", type=str, default="diag,const,shared,gru,lstm",
                   help="Comma-separated model names")
    p.add_argument("--optimizers", type=str, default="sgd,adamw",
                   help="Comma-separated optimizers (sgd, sgd_momentum, adamw, rmsprop)")
    p.add_argument("--lags", type=str, default="1,2,3,5,8,13,21,34,55,80",
                   help="Comma-separated lag values to evaluate (Appendix C uses 1,2,3,5,8,13,21,34,55,89,144,245)")
    p.add_argument("--outdir", type=str, default="results/distortion_diagnostic",
                   help="Output directory")
    p.add_argument("--skip_training", action="store_true",
                   help="Use untrained (randomly initialized) models")
    p.add_argument("--ln", action="store_true", help="Enable layer normalization")
    p.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD/RMSProp")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--rmsprop_alpha", type=float, default=0.99, help="RMSProp smoothing")
    return p.parse_args()


def build_model_by_name(name, D, H, const_s, ln):
    name = name.lower().strip()
    if name == "const":
        return ConstGateRNN(D, H, s=const_s, ln=ln), "ConstGate"
    elif name == "shared":
        return SharedGateRNN(D, H, ln=ln, init_s=const_s), "SharedGate"
    elif name == "diag":
        return DiagGateRNN(D, H, ln=ln, init_s=const_s), "DiagGate"
    elif name == "gru":
        return GRUModel(D, H, ln=ln), "GRU"
    elif name == "lstm":
        return LSTMModel(D, H, ln=ln), "LSTM"
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    log(f"First-order distortion diagnostic (with GELR)")
    log(f"H={args.H}, D={args.D}, T={args.T}, B={args.B}, seed={args.seed}")

    # Generate data
    task_lags = [5, 20, 50]
    task_coeffs = [1.0, 0.5, 0.25]
    X, Y, u = make_dataset_cpu(args.B, args.T, args.D, task_lags, task_coeffs)
    log(f"Data: {X.shape}, task_lags={task_lags}")

    model_names = [m.strip() for m in args.models.split(",")]
    optimizer_names = [o.strip() for o in args.optimizers.split(",")]
    lags = [int(l) for l in args.lags.split(",")]
    lags = [l for l in lags if l < args.T - 1]

    all_results = {}

    for mname in model_names:
        for opt_name in optimizer_names:
            run_key = f"{mname}_{opt_name}"
            display_key = f"{build_model_by_name(mname, args.D, args.H, args.const_s, args.ln)[1]}_{opt_name}"

            log(f"\n{'='*60}")
            # Fresh model for each (model, optimizer) combo
            set_seed(args.seed)
            model, display_name = build_model_by_name(mname, args.D, args.H, args.const_s, args.ln)
            apply_orthogonal(model)
            log(f"Model: {display_name}  Optimizer: {opt_name}  "
                f"(params={sum(p.numel() for p in model.parameters()):,})")

            if not args.skip_training:
                log(f"Training {display_name} with {opt_name}...")
                model, opt = train_model_simple(
                    model, X, Y, epochs=args.epochs, lr=args.lr,
                    optimizer_name=opt_name, momentum=args.momentum,
                    weight_decay=args.weight_decay, rmsprop_alpha=args.rmsprop_alpha)
            else:
                log(f"Skipping training (using random init)")
                model.eval()
                opt = None

            log(f"Running distortion diagnostic...")
            results = run_distortion_diagnostic(
                model, display_key, X, Y, lags, args,
                optimizer=opt, lr=args.lr)
            all_results[display_key] = results

            # Save per-(model, optimizer) CSV — exclude _trend (nested dict)
            csv_path = os.path.join(args.outdir, f"distortion_{run_key}.csv")
            if results:
                csv_fields = [k for k in results[0].keys() if k != "_trend"]
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(results)
                log(f"Saved: {csv_path}")

            # Save trend summary JSON per run
            if results and "_trend" in results[0]:
                trend_path = os.path.join(args.outdir, f"trend_{run_key}.json")
                with open(trend_path, "w") as f:
                    json.dump(results[0]["_trend"], f, indent=2)
                log(f"Saved: {trend_path}")

    # Save combined JSON (strip _trend to avoid nesting issues)
    json_safe = {}
    for key, res_list in all_results.items():
        json_safe[key] = [{k: v for k, v in r.items() if k != "_trend"} for r in res_list]
    json_path = os.path.join(args.outdir, "distortion_all.json")
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    log(f"Saved: {json_path}")

    # Save all trend summaries in one file
    trend_all = {}
    for key, res_list in all_results.items():
        if res_list and "_trend" in res_list[0]:
            trend_all[key] = res_list[0]["_trend"]
    trend_all_path = os.path.join(args.outdir, "trend_all.json")
    with open(trend_all_path, "w") as f:
        json.dump(trend_all, f, indent=2)
    log(f"Saved: {trend_all_path}")

    # ---- Print summary tables ----
    log(f"\n{'='*60}")

    # Table 1: Envelope comparison per lag (exact vs first-order approx)
    for name, res_list in all_results.items():
        log(f"\n  [{name}] Envelope decay profile (log₁₀ scale)")
        log(f"  {'Lag':>6s}  {'exact':>10s}  {'approx':>10s}  {'Δlog₁₀':>8s}  {'offdiag':>8s}")
        for r in res_list:
            log(f"  {r['lag']:>6d}  {r['log10_f_exact']:>10.2f}  "
                f"{r['log10_f_approx']:>10.2f}  {r['delta_log']:>8.3f}  "
                f"{r['offdiag_frac_mean']:>8.4f}")

    # Table 2: Trend preservation summary across all runs
    log(f"\n{'='*60}")
    log(f"TREND PRESERVATION SUMMARY")
    log(f"  {'Run':>30s}  {'slope_ex':>8s}  {'slope_ap':>8s}  "
        f"{'ratio':>6s}  {'ρ':>7s}  {'r':>7s}  "
        f"{'Δlog_μ':>8s}  {'Δlog_σ':>7s}")
    log(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}")
    for name, res_list in all_results.items():
        if res_list and "_trend" in res_list[0]:
            t = res_list[0]["_trend"]
            log(f"  {name:>30s}  {t['slope_exact']:>8.3f}  {t['slope_approx']:>8.3f}  "
                f"{t['slope_ratio']:>6.3f}  {t['spearman']:>7.4f}  {t['pearson']:>7.4f}  "
                f"{t['delta_log_mean']:>8.3f}  {t['delta_log_std']:>7.3f}")

    log(f"\nDone. Results in {args.outdir}/")


if __name__ == "__main__":
    main()
