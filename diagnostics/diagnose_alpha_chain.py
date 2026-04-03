#!/usr/bin/env python3
"""Post-hoc alpha-chain diagnostic for projected gradients and matched statistics.

This script compares five scalar objects on the same frozen trained model:

1. projected stochastic gradient noise: <g_batch - g_full, w>
2. exact lag-specific projected instantaneous contribution
3. exact lag-specific projected sequence average
4. first-order instantaneous matched statistic samples: psi_{n,t}(ell)
5. first-order sequence-averaged matched statistic samples: mean_t psi_{n,t}(ell)

The first object is motivated by the heavy-tailed SGD literature. Objects (2)
and (3) isolate the exact lag-specific projected contribution without the
first-order envelope approximation. Objects (4) and (5) are the theorem-facing
matched-statistic objects used in the learnability pipeline.
"""

import argparse
import csv
import gc
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp

# Allow direct execution via `python diagnostics/diagnose_alpha_chain.py`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import run_learnability_baselines as base_runner
import run_learnability_lstm_gru as recurrent_runner
from seed_utils import bootstrap_mcculloch


BASELINE_MODELS = {"const", "shared", "diag"}
RECURRENT_MODELS = {"gru", "lstm"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--root",
        type=str,
        help=(
            "Run root containing baselines/seed_<S>/<model> and "
            "lstmgru/seed_<S>/<model> subtrees."
        ),
    )
    group.add_argument(
        "--modeldirs",
        nargs="+",
        help="Explicit model directories containing *_final_checkpoint.pt artifacts.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--models", type=str, default="diag,gru")
    p.add_argument("--lags", type=str, default="4,64,256,512")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--projection_w_seeds",
        type=str,
        default="",
        help=(
            "Comma-separated override list for validation-time projection seeds. "
            "If empty, the checkpoint's saved w_seed is used."
        ),
    )
    p.add_argument(
        "--gradient_dataset",
        type=str,
        default="train",
        choices=["train", "diag"],
        help="Dataset split used for the projected-gradient-noise diagnostic.",
    )
    p.add_argument(
        "--full_grad_batch_size",
        type=int,
        default=32,
        help="Streaming batch size for the full-gradient reference estimate.",
    )
    p.add_argument(
        "--grad_batch_size",
        type=int,
        default=32,
        help="Mini-batch size for projected gradient-noise samples.",
    )
    p.add_argument(
        "--grad_num_batches",
        type=int,
        default=1024,
        help="Number of random mini-batches used for projected gradient-noise samples.",
    )
    p.add_argument(
        "--matched_batch_size",
        type=int,
        default=32,
        help="Streaming batch size used to collect matched-statistic samples.",
    )
    p.add_argument(
        "--instantaneous_sample_cap",
        type=int,
        default=200000,
        help="Maximum instantaneous psi samples retained per lag.",
    )
    p.add_argument(
        "--alpha_min_samples",
        type=int,
        default=500,
        help="Minimum sample count required by the alpha estimators.",
    )
    p.add_argument(
        "--alpha_n_boot",
        type=int,
        default=300,
        help="Number of bootstrap replicates for McCulloch confidence intervals.",
    )
    p.add_argument(
        "--ecf_subsample_limit",
        type=int,
        default=100000,
        help="Maximum sample count passed into the ECF regression routine.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Optional output directory. Defaults to a shared alpha_chain_diagnostics folder.",
    )
    p.add_argument(
        "--save_raw_samples",
        type=int,
        default=0,
        help=(
            "Whether to save raw sample arrays. Default 0 disables NPZ output to reduce "
            "host-memory pressure during long runs."
        ),
    )
    return p.parse_args()


def _common_output_dir(args: argparse.Namespace, model_dirs: Sequence[str]) -> str:
    if args.outdir:
        return args.outdir
    if args.root:
        return os.path.join(args.root, "alpha_chain_diagnostics", f"seed_{args.seed}")
    if len(model_dirs) == 1:
        return os.path.join(model_dirs[0], "alpha_chain_diagnostics")
    return os.path.join(os.path.commonpath(model_dirs), "alpha_chain_diagnostics")


def _parse_csv_list(text: str, cast) -> List:
    return [cast(s) for s in text.split(",") if s.strip()]


def _resolve_model_dirs(args: argparse.Namespace) -> List[str]:
    if args.modeldirs:
        return [os.path.abspath(path) for path in args.modeldirs]

    model_dirs: List[str] = []
    for model in [m.strip().lower() for m in args.models.split(",") if m.strip()]:
        if model in BASELINE_MODELS:
            subdir = os.path.join(args.root, "baselines", f"seed_{args.seed}", model)
        elif model in RECURRENT_MODELS:
            subdir = os.path.join(args.root, "lstmgru", f"seed_{args.seed}", model)
        else:
            raise ValueError(f"Unknown model for auto-resolution: {model}")
        if not os.path.isdir(subdir):
            raise FileNotFoundError(f"Missing model directory: {subdir}")
        model_dirs.append(os.path.abspath(subdir))
    return model_dirs


def _resolve_projection_w_seeds(args: argparse.Namespace, run_args: SimpleNamespace) -> List[int]:
    if args.projection_w_seeds.strip():
        return _parse_csv_list(args.projection_w_seeds, int)
    return [int(run_args.w_seed)]


def _find_checkpoint(model_dir: str) -> str:
    candidates = sorted(
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.endswith("_final_checkpoint.pt")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No final checkpoint found in {model_dir}. "
            "Re-run the pilot with the updated runners that save final checkpoints."
        )
    if len(candidates) > 1:
        raise RuntimeError(f"Expected one final checkpoint in {model_dir}, found: {candidates}")
    return candidates[0]


def _namespace_from_payload(payload: Dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**dict(payload["args"]))


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _load_model_bundle(
    model_dir: str,
    device: torch.device,
) -> Tuple[object, str, SimpleNamespace, torch.nn.Module, torch.optim.Optimizer, np.ndarray]:
    ckpt_path = _find_checkpoint(model_dir)
    # These checkpoints intentionally store args/metadata in addition to weights.
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = _namespace_from_payload(payload)
    runner_type = str(payload["runner_type"])
    model_name = str(payload["model_name"]).lower()

    if runner_type == "baselines":
        runner = base_runner
        model = runner.build_model(
            model_name,
            args.D,
            args.H,
            const_s=args.const_s,
            ln=args.layernorm,
        ).to(device)
    elif runner_type == "lstmgru":
        runner = recurrent_runner
        model = runner.build_model(
            model_name,
            args.D,
            args.H,
            ln=args.layernorm,
            gru_init_update=args.gru_init_update,
            lstm_init_forget=args.lstm_init_forget,
        ).to(device)
    else:
        raise ValueError(f"Unknown runner type in checkpoint: {runner_type}")

    optimizer = runner.make_optimizer(args, model)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    model.eval()

    u_vec = payload.get("u_vec")
    if u_vec is None:
        raise ValueError(f"Checkpoint missing saved task direction u_vec: {ckpt_path}")
    return runner, model_name, args, model, optimizer, np.asarray(u_vec, dtype=np.float32)


def _rebuild_datasets(
    runner,
    args: SimpleNamespace,
    u_vec: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    runner.set_seed(int(args.seed))
    Xtr_cpu, Ytr_cpu, u_regen = runner.make_dataset_cpu(
        args.Nseq_train,
        args.T,
        args.D,
        args.task_lags,
        args.task_coeffs,
        args.noise_std,
        u_vec=None,
    )
    if not np.allclose(np.asarray(u_regen, dtype=np.float32), u_vec, atol=1e-6):
        raise ValueError("Saved task direction does not match deterministic regeneration from the run seed.")
    Xdg_cpu, Ydg_cpu, _ = runner.make_dataset_cpu(
        args.Nseq_diag,
        args.T,
        args.D,
        args.task_lags,
        args.task_coeffs,
        args.noise_std,
        u_vec=u_vec,
    )
    return Xtr_cpu, Ytr_cpu, Xdg_cpu, Ydg_cpu


def _project_current_gradients(model: torch.nn.Module, w_tangent: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if (not param.requires_grad) or param.grad is None:
            continue
        tangent = w_tangent[name].to(device=param.grad.device, dtype=param.grad.dtype)
        total += float(torch.sum(param.grad.detach() * tangent).item())
    return float(total)


def _compute_projected_full_gradient(
    model: torch.nn.Module,
    X_cpu: torch.Tensor,
    Y_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
    w_tangent: Dict[str, torch.Tensor],
) -> float:
    model.zero_grad(set_to_none=True)
    total_numel = int(Y_cpu.numel())
    Btot = int(X_cpu.shape[0])
    n_batches = max(1, int(np.ceil(Btot / batch_size)))
    for bi in range(n_batches):
        lo = bi * batch_size
        hi = min(Btot, (bi + 1) * batch_size)
        xb = X_cpu[lo:hi].to(device, non_blocking=True)
        yb = Y_cpu[lo:hi].to(device, non_blocking=True)
        yhat, _, _ = model.forward_with_intermediates(xb, return_intermediates=False)
        loss = F.mse_loss(yhat, yb, reduction="sum") / float(total_numel)
        loss.backward()
        del xb, yb, yhat, loss
    proj = _project_current_gradients(model, w_tangent)
    model.zero_grad(set_to_none=True)
    return proj


def _sample_projected_gradient_noise(
    model: torch.nn.Module,
    runner,
    X_cpu: torch.Tensor,
    Y_cpu: torch.Tensor,
    device: torch.device,
    full_grad_batch_size: int,
    grad_batch_size: int,
    n_batches: int,
    w_seed: int,
    sample_seed: int,
) -> np.ndarray:
    _, _, w_tangent = runner._make_random_unit_w_pytree(model, device=device, seed=w_seed)
    full_proj = _compute_projected_full_gradient(
        model,
        X_cpu,
        Y_cpu,
        device=device,
        batch_size=full_grad_batch_size,
        w_tangent=w_tangent,
    )

    rng = np.random.RandomState(int(sample_seed) + 20260326)
    Btot = int(X_cpu.shape[0])
    replace = grad_batch_size > Btot
    out = np.zeros(int(n_batches), dtype=np.float64)
    for i in range(int(n_batches)):
        idx = rng.choice(Btot, size=int(grad_batch_size), replace=replace)
        xb = X_cpu[idx].to(device, non_blocking=True)
        yb = Y_cpu[idx].to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        yhat, _, _ = model.forward_with_intermediates(xb, return_intermediates=False)
        loss = F.mse_loss(yhat, yb)
        loss.backward()
        proj_batch = _project_current_gradients(model, w_tangent)
        out[i] = proj_batch - full_proj
        model.zero_grad(set_to_none=True)
        del xb, yb, yhat, loss
    return out


def _orient_psi_if_requested(psi: torch.Tensor, orient_flag: bool) -> torch.Tensor:
    if not orient_flag:
        return psi
    mu_psi = psi.mean()
    if torch.isfinite(mu_psi):
        sgn = torch.sign(mu_psi)
        if float(sgn.item()) == 0.0:
            sgn = torch.tensor(1.0, device=psi.device)
    else:
        sgn = torch.tensor(1.0, device=psi.device)
    return sgn * psi


def _baseline_hidden_step(model: torch.nn.Module, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
    cls = model.__class__.__name__
    if cls == "ConstGateRNN":
        s = model.s_const
        pre = model.Wx(x_t) + model.Wh(h_prev)
        pre = model.ln(pre)
        h_tilde = torch.tanh(pre)
        return (1.0 - s) * h_prev + s * h_tilde
    if cls == "SharedGateRNN":
        s = torch.sigmoid(model.Ws(x_t) + model.Us(h_prev))
        pre = model.Wx(x_t) + model.Wh(h_prev)
        pre = model.ln_h(pre)
        h_tilde = torch.tanh(pre)
        sH = s.expand_as(h_prev)
        return (1.0 - sH) * h_prev + sH * h_tilde
    if cls == "DiagGateRNN":
        s = torch.sigmoid(model.Ws(x_t) + model.Us(h_prev))
        pre = model.Wx(x_t) + model.Wh(h_prev)
        pre = model.ln_h(pre)
        h_tilde = torch.tanh(pre)
        return (1.0 - s) * h_prev + s * h_tilde
    raise NotImplementedError(f"Exact lag transport not implemented for baseline model class {cls}")


def _gru_hidden_step(model: torch.nn.Module, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
    z = torch.sigmoid(model.Wz(x_t) + model.Uz(h_prev))
    r = torch.sigmoid(model.Wr(x_t) + model.Ur(h_prev))
    g = torch.tanh(model.ln_h(model.Wh(x_t) + model.Uh(r * h_prev)))
    return (1.0 - z) * h_prev + z * g


def _propagate_exact_baseline_lag(
    model: torch.nn.Module,
    xb: torch.Tensor,
    hseq: torch.Tensor,
    source_v: torch.Tensor,
    ell: int,
) -> torch.Tensor:
    if ell <= 0:
        return source_v
    B, width, H = source_v.shape
    u = source_v.reshape(B * width, H)
    for k in range(1, ell + 1):
        x_step = xb[:, k:width + k, :].reshape(B * width, xb.shape[2])
        h_prev = hseq[:, k - 1:width + k - 1, :].reshape(B * width, H)

        def step(h_prev_local: torch.Tensor) -> torch.Tensor:
            return _baseline_hidden_step(model, x_step, h_prev_local)

        _, u = jvp(step, (h_prev,), (u,))
        # We only need the numeric tangent value for the next transport step.
        # Detaching here prevents forward-mode/autograd metadata from
        # accumulating across long lag loops and batches.
        u = u.detach()
    return u.reshape(B, width, H).detach()


def _propagate_exact_gru_lag(
    model: torch.nn.Module,
    xb: torch.Tensor,
    hseq: torch.Tensor,
    source_v: torch.Tensor,
    ell: int,
) -> torch.Tensor:
    if ell <= 0:
        return source_v
    B, width, H = source_v.shape
    u = source_v.reshape(B * width, H)
    for k in range(1, ell + 1):
        x_step = xb[:, k:width + k, :].reshape(B * width, xb.shape[2])
        h_prev = hseq[:, k - 1:width + k - 1, :].reshape(B * width, H)

        def step(h_prev_local: torch.Tensor) -> torch.Tensor:
            return _gru_hidden_step(model, x_step, h_prev_local)

        _, u = jvp(step, (h_prev,), (u,))
        # We only need the numeric tangent value for the next transport step.
        # Detaching here prevents forward-mode/autograd metadata from
        # accumulating across long lag loops and batches.
        u = u.detach()
    return u.reshape(B, width, H).detach()


def _maybe_append_instantaneous(
    store: Dict[int, List[np.ndarray]],
    counts: Dict[int, int],
    cap: int,
    ell: int,
    values: np.ndarray,
    rng: np.random.RandomState,
) -> None:
    remaining = int(cap) - int(counts[ell])
    if remaining <= 0:
        return
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size > remaining:
        idx = rng.choice(flat.size, size=remaining, replace=False)
        flat = flat[idx]
    store[ell].append(flat)
    counts[ell] += int(flat.size)


def _collect_baseline_matched_samples(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: SimpleNamespace,
    Xdg_cpu: torch.Tensor,
    Ydg_cpu: torch.Tensor,
    device: torch.device,
    lags: Sequence[int],
    batch_size: int,
    instantaneous_cap: int,
    w_seed: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    Tdg = int(Xdg_cpu.shape[1])
    Bdg = int(Xdg_cpu.shape[0])
    lambda_matrix, lambda_rowmean = base_runner.extract_adaptive_rate_matrix(model, optimizer, lr=args.lr)
    use_lag_dependent = (lambda_matrix is not None) and bool(args.layernorm)
    lambda_fallback = torch.tensor(lambda_rowmean, dtype=torch.float64, device=device)
    Wout = model.out.weight.detach()

    seq_lists = {ell: [] for ell in lags}
    inst_lists = {ell: [] for ell in lags}
    inst_counts = {ell: 0 for ell in lags}
    exact_seq_lists = {ell: [] for ell in lags}
    exact_inst_lists = {ell: [] for ell in lags}
    exact_inst_counts = {ell: 0 for ell in lags}
    rng = np.random.RandomState(int(args.seed) + 424242)
    rng_exact = np.random.RandomState(int(args.seed) + 424243)

    n_batches = max(1, int(np.ceil(Bdg / batch_size)))
    for bi in range(n_batches):
        lo = bi * batch_size
        hi = min(Bdg, (bi + 1) * batch_size)
        xb = Xdg_cpu[lo:hi].to(device, non_blocking=True)
        yb = Ydg_cpu[lo:hi].to(device, non_blocking=True)

        with torch.no_grad():
            yhat, hseq, g = model.forward_with_intermediates(xb)
            leak = g["leak"]
            rdiag = g["rdiag"]
            err = yhat[..., 0] - yb[..., 0]
            delta = err.unsqueeze(-1) * Wout

        vseq = base_runner.compute_vseq_jvp(model, xb, w_seed=w_seed).detach()
        cs_log, cs_ratio = base_runner.precompute_prefix_sums(leak, rdiag)

        for ell in lags:
            mu0, _mu1, mu_all = base_runner.mu_for_matched_stat_from_prefix(
                cs_log, cs_ratio, int(ell), out_dtype=torch.float64
            )
            mu_used = mu_all if bool(args.include_first_order_diag) else mu0
            if mu_used.numel() == 0:
                continue
            if use_lag_dependent:
                lambda_ell = base_runner.compute_lag_dependent_rates(
                    lambda_matrix, hseq, mu_used.shape[1], lambda_fallback
                ).to(mu_used.dtype)
                mu_used = mu_used * lambda_ell
            else:
                mu_used = mu_used * lambda_fallback.unsqueeze(0).unsqueeze(0)

            psi = torch.sum(
                mu_used * delta[:, ell:Tdg, :] * vseq[:, 0:(Tdg - ell), :],
                dim=2,
            )
            psi = _orient_psi_if_requested(psi, bool(getattr(args, "orient_matched_statistic_sign", 0)))
            psi_np = psi.detach().cpu().numpy().astype(np.float64)
            seq_lists[ell].append(psi_np.mean(axis=1))
            _maybe_append_instantaneous(inst_lists, inst_counts, instantaneous_cap, ell, psi_np, rng)

            source_v = vseq[:, 0:(Tdg - ell), :]
            # Exact lag-specific projected contribution:
            #   delta_t^T M_{t,ell} (B_ell w)
            # This intentionally excludes the optimizer-weighted first-order
            # surrogate mu_{t,ell}^{(q)} so we can compare raw exact transport
            # against the first-order matched-statistic construction.
            exact_u = _propagate_exact_baseline_lag(model, xb, hseq, source_v, int(ell))
            exact_proj = torch.sum(delta[:, ell:Tdg, :] * exact_u, dim=2)
            exact_np = exact_proj.detach().cpu().numpy().astype(np.float64)
            exact_seq_lists[ell].append(exact_np.mean(axis=1))
            _maybe_append_instantaneous(
                exact_inst_lists, exact_inst_counts, instantaneous_cap, ell, exact_np, rng_exact
            )

        del xb, yb, yhat, hseq, g, leak, rdiag, err, delta, vseq, cs_log, cs_ratio

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for ell in lags:
        out[ell] = {
            "exact_instantaneous": (
                np.concatenate(exact_inst_lists[ell]) if exact_inst_lists[ell] else np.array([], dtype=np.float64)
            ),
            "exact_sequence_avg": (
                np.concatenate(exact_seq_lists[ell]) if exact_seq_lists[ell] else np.array([], dtype=np.float64)
            ),
            "instantaneous": (
                np.concatenate(inst_lists[ell]) if inst_lists[ell] else np.array([], dtype=np.float64)
            ),
            "sequence_avg": (
                np.concatenate(seq_lists[ell]) if seq_lists[ell] else np.array([], dtype=np.float64)
            ),
        }
    return out


def _collect_recurrent_matched_samples(
    model_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: SimpleNamespace,
    Xdg_cpu: torch.Tensor,
    Ydg_cpu: torch.Tensor,
    device: torch.device,
    lags: Sequence[int],
    batch_size: int,
    instantaneous_cap: int,
    w_seed: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    Tdg = int(Xdg_cpu.shape[1])
    Bdg = int(Xdg_cpu.shape[0])
    lambda_matrix, lambda_rowmean = recurrent_runner.extract_adaptive_rate_matrix(model, optimizer, lr=args.lr)
    use_lag_dependent = (lambda_matrix is not None) and bool(args.layernorm)
    lambda_fallback = torch.tensor(lambda_rowmean, dtype=torch.float64, device=device)
    Wout = model.out.weight.detach()

    seq_lists = {ell: [] for ell in lags}
    inst_lists = {ell: [] for ell in lags}
    inst_counts = {ell: 0 for ell in lags}
    exact_seq_lists = {ell: [] for ell in lags}
    exact_inst_lists = {ell: [] for ell in lags}
    exact_inst_counts = {ell: 0 for ell in lags}
    rng = np.random.RandomState(int(args.seed) + 424242)
    rng_exact = np.random.RandomState(int(args.seed) + 424243)

    n_batches = max(1, int(np.ceil(Bdg / batch_size)))
    for bi in range(n_batches):
        lo = bi * batch_size
        hi = min(Bdg, (bi + 1) * batch_size)
        xb = Xdg_cpu[lo:hi].to(device, non_blocking=True)
        yb = Ydg_cpu[lo:hi].to(device, non_blocking=True)

        with torch.no_grad():
            yhat, hseq, g = model.forward_with_intermediates(xb)
            err = yhat[..., 0] - yb[..., 0]
            delta = err.unsqueeze(-1) * Wout

        vseq = recurrent_runner.compute_vseq_jvp(model, xb, w_seed=w_seed).detach()

        if model_name == "gru":
            cs0, cs1, cs2, cs_ratio = recurrent_runner.precompute_prefixes_gru(
                g["leak"], g["r"], g["rdiag"]
            )
        else:
            cs0, cs_ratio, expr = recurrent_runner.precompute_prefixes_lstm(
                g["forget"], g["expr"], g["cdiag"]
            )

        for ell in lags:
            if model_name == "gru":
                mu0, _mu1, mu_all = recurrent_runner.mu_for_matched_stat_gru(
                    cs0,
                    cs1,
                    cs2,
                    cs_ratio,
                    int(ell),
                    include_first_order=bool(args.include_first_order_diag),
                    out_dtype=torch.float64,
                )
            else:
                mu0, _mu1, mu_all = recurrent_runner.mu_for_matched_stat_lstm(
                    cs0,
                    cs_ratio,
                    expr,
                    int(ell),
                    include_first_order=bool(args.include_first_order_diag),
                    out_dtype=torch.float64,
                )

            mu_used = mu_all if bool(args.include_first_order_diag) else mu0
            if mu_used.numel() == 0:
                continue
            if use_lag_dependent:
                lambda_ell = recurrent_runner.compute_lag_dependent_rates(
                    lambda_matrix, hseq, mu_used.shape[1], lambda_fallback
                ).to(mu_used.dtype)
                mu_used = mu_used * lambda_ell
            else:
                mu_used = mu_used * lambda_fallback.unsqueeze(0).unsqueeze(0)

            psi = torch.sum(
                mu_used * delta[:, ell:Tdg, :] * vseq[:, 0:(Tdg - ell), :],
                dim=2,
            )
            psi = _orient_psi_if_requested(psi, bool(getattr(args, "orient_matched_statistic_sign", 0)))
            psi_np = psi.detach().cpu().numpy().astype(np.float64)
            seq_lists[ell].append(psi_np.mean(axis=1))
            _maybe_append_instantaneous(inst_lists, inst_counts, instantaneous_cap, ell, psi_np, rng)

            if model_name != "gru":
                raise NotImplementedError(
                    f"Exact lag-specific projected contribution is currently implemented for GRU only, got {model_name}"
                )
            source_v = vseq[:, 0:(Tdg - ell), :]
            # Exact lag-specific projected contribution:
            #   delta_t^T M_{t,ell} (B_ell w)
            # This intentionally excludes the optimizer-weighted first-order
            # surrogate mu_{t,ell}^{(q)} so we can compare raw exact transport
            # against the first-order matched-statistic construction.
            exact_u = _propagate_exact_gru_lag(model, xb, hseq, source_v, int(ell))
            exact_proj = torch.sum(delta[:, ell:Tdg, :] * exact_u, dim=2)
            exact_np = exact_proj.detach().cpu().numpy().astype(np.float64)
            exact_seq_lists[ell].append(exact_np.mean(axis=1))
            _maybe_append_instantaneous(
                exact_inst_lists, exact_inst_counts, instantaneous_cap, ell, exact_np, rng_exact
            )

        del xb, yb, yhat, hseq, g, err, delta, vseq
        if model_name == "gru":
            del cs0, cs1, cs2, cs_ratio
        else:
            del cs0, cs_ratio, expr

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for ell in lags:
        out[ell] = {
            "exact_instantaneous": (
                np.concatenate(exact_inst_lists[ell]) if exact_inst_lists[ell] else np.array([], dtype=np.float64)
            ),
            "exact_sequence_avg": (
                np.concatenate(exact_seq_lists[ell]) if exact_seq_lists[ell] else np.array([], dtype=np.float64)
            ),
            "instantaneous": (
                np.concatenate(inst_lists[ell]) if inst_lists[ell] else np.array([], dtype=np.float64)
            ),
            "sequence_avg": (
                np.concatenate(seq_lists[ell]) if seq_lists[ell] else np.array([], dtype=np.float64)
            ),
        }
    return out


def _estimate_with_both_methods(
    runner,
    samples: np.ndarray,
    alpha_n_boot: int,
    ecf_subsample_limit: int,
) -> Dict[str, Dict[str, object]]:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return {
            "ecf": runner._default_alpha_meta("ecf", 0),
            "mcculloch": runner._default_alpha_meta("mcculloch", 0),
        }

    ecf = runner.estimate_alpha_sigma_with_meta(
        samples, method="ecf", n_samples_for_ecf=ecf_subsample_limit
    )
    mcc = runner.estimate_alpha_sigma_with_meta(samples, method="mcculloch")
    if samples.size >= 4 and np.isfinite(float(mcc["alpha_hat"])):
        median, ci_lo, ci_hi, _ = bootstrap_mcculloch(
            samples,
            runner.estimate_alpha_sigma_mcculloch_symmetric_from_quantiles,
            n_boot=alpha_n_boot,
            ci=0.95,
        )
    else:
        median, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")
    mcc = dict(mcc)
    mcc["bootstrap_median"] = float(median)
    mcc["ci_lo"] = float(ci_lo)
    mcc["ci_hi"] = float(ci_hi)
    mcc["ci_width"] = float(ci_hi - ci_lo) if np.isfinite(ci_lo) and np.isfinite(ci_hi) else float("inf")
    return {"ecf": ecf, "mcculloch": mcc}


def _samples_to_rows(
    model_name: str,
    lag: int,
    object_name: str,
    samples: np.ndarray,
    estimates: Dict[str, Dict[str, object]],
    extra_meta: Dict[str, object],
) -> List[Dict[str, object]]:
    base = {
        "model_name": model_name,
        "lag": int(lag),
        "object_name": object_name,
        "n_samples_total": int(samples.size),
        "sample_mean": float(np.mean(samples)) if samples.size else float("nan"),
        "sample_abs_mean": float(np.mean(np.abs(samples))) if samples.size else float("nan"),
        "sample_std": float(np.std(samples)) if samples.size else float("nan"),
    }
    base.update(extra_meta)

    rows: List[Dict[str, object]] = []
    for method_name, meta in estimates.items():
        row = dict(base)
        row.update({
            "estimation_method": method_name,
            "alpha_hat": float(meta["alpha_hat"]),
            "sigma_hat": float(meta["sigma_hat"]),
            "reliable": int(bool(meta["reliable"])),
            "method_origin": str(meta["method_origin"]),
            "method_reason": str(meta["method_reason"]),
            "reliability_reason": str(meta["reliability_reason"]),
            "n_samples_used": int(meta["n_samples_used"]),
            "used_subsample": int(meta["used_subsample"]),
            "iqr": float(meta["iqr"]),
            "quantile_ratio": float(meta["quantile_ratio"]),
            "boundary_hit": int(meta["boundary_hit"]),
            "ecf_n_grid": int(meta["ecf_n_grid"]),
            "ecf_n_points_strict": int(meta["ecf_n_points_strict"]),
            "ecf_n_points_relaxed": int(meta["ecf_n_points_relaxed"]),
            "ecf_n_points_used": int(meta["ecf_n_points_used"]),
            "ecf_filter_mode": str(meta["ecf_filter_mode"]),
            "mcc_bootstrap_median": float(meta.get("bootstrap_median", float("nan"))),
            "mcc_ci_lo": float(meta.get("ci_lo", float("nan"))),
            "mcc_ci_hi": float(meta.get("ci_hi", float("nan"))),
            "mcc_ci_width": float(meta.get("ci_width", float("nan"))),
        })
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    model_dirs = _resolve_model_dirs(args)
    outdir = _common_output_dir(args, model_dirs)
    os.makedirs(outdir, exist_ok=True)

    device = base_runner.resolve_device(args.device)
    summary: Dict[str, object] = {
        "config": {
            "lags": _parse_csv_list(args.lags, int),
            "requested_projection_w_seeds": _parse_csv_list(args.projection_w_seeds, int)
            if args.projection_w_seeds.strip()
            else [],
            "gradient_dataset": args.gradient_dataset,
            "full_grad_batch_size": int(args.full_grad_batch_size),
            "grad_batch_size": int(args.grad_batch_size),
            "grad_num_batches": int(args.grad_num_batches),
            "matched_batch_size": int(args.matched_batch_size),
            "instantaneous_sample_cap": int(args.instantaneous_sample_cap),
            "alpha_min_samples": int(args.alpha_min_samples),
            "alpha_n_boot": int(args.alpha_n_boot),
            "ecf_subsample_limit": int(args.ecf_subsample_limit),
            "save_raw_samples": int(args.save_raw_samples),
        },
        "models": {},
    }
    lags = _parse_csv_list(args.lags, int)

    csv_path = os.path.join(outdir, "alpha_chain_estimates.csv")
    fieldnames = [
        "model_name",
        "projection_w_seed",
        "lag",
        "object_name",
        "estimation_method",
        "n_samples_total",
        "n_samples_used",
        "used_subsample",
        "sample_mean",
        "sample_abs_mean",
        "sample_std",
        "alpha_hat",
        "sigma_hat",
        "reliable",
        "method_origin",
        "method_reason",
        "reliability_reason",
        "iqr",
        "quantile_ratio",
        "boundary_hit",
        "ecf_n_grid",
        "ecf_n_points_strict",
        "ecf_n_points_relaxed",
        "ecf_n_points_used",
        "ecf_filter_mode",
        "mcc_bootstrap_median",
        "mcc_ci_lo",
        "mcc_ci_hi",
        "mcc_ci_width",
        "gradient_dataset",
        "gradient_batch_size",
        "gradient_num_batches",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for model_dir in model_dirs:
        runner, model_name, run_args, model, optimizer, u_vec = _load_model_bundle(model_dir, device)
        projection_w_seeds = _resolve_projection_w_seeds(args, run_args)
        print(
            f"[alpha-chain] model={model_name} model_dir={model_dir} "
            f"projection_w_seeds={projection_w_seeds}"
        )
        runner._MIN_SAMPLES_ALPHA = int(args.alpha_min_samples)
        Xtr_cpu, Ytr_cpu, Xdg_cpu, Ydg_cpu = _rebuild_datasets(runner, run_args, u_vec)
        grad_X = Xtr_cpu if args.gradient_dataset == "train" else Xdg_cpu
        grad_Y = Ytr_cpu if args.gradient_dataset == "train" else Ydg_cpu
        summary["models"][model_name] = {
            "resolved_projection_w_seeds": projection_w_seeds,
            "by_projection_w_seed": {},
        }
        for w_seed in projection_w_seeds:
            print(f"[alpha-chain] start model={model_name} w_seed={w_seed}")
            rows_this_w: List[Dict[str, object]] = []
            sample_payload: Dict[str, np.ndarray] = {}
            projected_grad_noise = _sample_projected_gradient_noise(
                model,
                runner,
                grad_X,
                grad_Y,
                device=device,
                full_grad_batch_size=int(args.full_grad_batch_size),
                grad_batch_size=int(args.grad_batch_size),
                n_batches=int(args.grad_num_batches),
                w_seed=int(w_seed),
                sample_seed=int(args.seed),
            )
            if bool(args.save_raw_samples):
                sample_payload[
                    f"{model_name}__w_seed_{w_seed}__projected_gradient_noise"
                ] = projected_grad_noise.astype(np.float32, copy=False)
            grad_estimates = _estimate_with_both_methods(
                runner,
                projected_grad_noise,
                alpha_n_boot=int(args.alpha_n_boot),
                ecf_subsample_limit=int(args.ecf_subsample_limit),
            )

            if model_name in BASELINE_MODELS:
                matched_by_lag = _collect_baseline_matched_samples(
                    model,
                    optimizer,
                    run_args,
                    Xdg_cpu,
                    Ydg_cpu,
                    device=device,
                    lags=lags,
                    batch_size=int(args.matched_batch_size),
                    instantaneous_cap=int(args.instantaneous_sample_cap),
                    w_seed=int(w_seed),
                )
            else:
                matched_by_lag = _collect_recurrent_matched_samples(
                    model_name,
                    model,
                    optimizer,
                    run_args,
                    Xdg_cpu,
                    Ydg_cpu,
                    device=device,
                    lags=lags,
                    batch_size=int(args.matched_batch_size),
                    instantaneous_cap=int(args.instantaneous_sample_cap),
                    w_seed=int(w_seed),
                )

            w_summary: Dict[str, object] = {}
            for lag in lags:
                lag_summary = {}

                rows_this_w.extend(_samples_to_rows(
                    model_name,
                    lag,
                    "projected_gradient_noise",
                    projected_grad_noise,
                    grad_estimates,
                    {
                        "projection_w_seed": int(w_seed),
                        "gradient_dataset": args.gradient_dataset,
                        "gradient_batch_size": int(args.grad_batch_size),
                        "gradient_num_batches": int(args.grad_num_batches),
                    },
                ))
                lag_summary["projected_gradient_noise"] = {
                    method: {
                        "alpha_hat": float(meta["alpha_hat"]),
                        "reliable": bool(meta["reliable"]),
                    }
                    for method, meta in grad_estimates.items()
                }

                for object_name in ["exact_instantaneous", "exact_sequence_avg", "instantaneous", "sequence_avg"]:
                    samples = matched_by_lag[lag][object_name]
                    if bool(args.save_raw_samples):
                        sample_payload[
                            f"{model_name}__w_seed_{w_seed}__lag_{lag}__{object_name}"
                        ] = samples.astype(np.float32, copy=False)
                    estimates = _estimate_with_both_methods(
                        runner,
                        samples,
                        alpha_n_boot=int(args.alpha_n_boot),
                        ecf_subsample_limit=int(args.ecf_subsample_limit),
                    )
                    rows_this_w.extend(_samples_to_rows(
                        model_name,
                        lag,
                        (
                            f"exact_projected_lag_{object_name.split('_', 1)[1]}"
                            if object_name.startswith("exact_")
                            else f"matched_stat_{object_name}"
                        ),
                        samples,
                        estimates,
                        {"projection_w_seed": int(w_seed)},
                    ))
                    summary_key = (
                        f"exact_projected_lag_{object_name.split('_', 1)[1]}"
                        if object_name.startswith("exact_")
                        else f"matched_stat_{object_name}"
                    )
                    lag_summary[summary_key] = {
                        method: {
                            "alpha_hat": float(meta["alpha_hat"]),
                            "reliable": bool(meta["reliable"]),
                        }
                        for method, meta in estimates.items()
                    }

                for method_name in ["ecf", "mcculloch"]:
                    g_alpha = lag_summary["projected_gradient_noise"][method_name]["alpha_hat"]
                    ei_alpha = lag_summary["exact_projected_lag_instantaneous"][method_name]["alpha_hat"]
                    es_alpha = lag_summary["exact_projected_lag_sequence_avg"][method_name]["alpha_hat"]
                    i_alpha = lag_summary["matched_stat_instantaneous"][method_name]["alpha_hat"]
                    s_alpha = lag_summary["matched_stat_sequence_avg"][method_name]["alpha_hat"]
                    lag_summary.setdefault("chain_deltas", {})[method_name] = {
                        "full_grad_to_exact_instantaneous": float(ei_alpha - g_alpha) if np.isfinite(g_alpha) and np.isfinite(ei_alpha) else float("nan"),
                        "full_grad_to_exact_sequence_avg": float(es_alpha - g_alpha) if np.isfinite(g_alpha) and np.isfinite(es_alpha) else float("nan"),
                        "exact_instantaneous_to_exact_sequence_avg": float(es_alpha - ei_alpha) if np.isfinite(ei_alpha) and np.isfinite(es_alpha) else float("nan"),
                        "exact_instantaneous_to_matched_instantaneous": float(i_alpha - ei_alpha) if np.isfinite(ei_alpha) and np.isfinite(i_alpha) else float("nan"),
                        "exact_sequence_avg_to_matched_sequence_avg": float(s_alpha - es_alpha) if np.isfinite(es_alpha) and np.isfinite(s_alpha) else float("nan"),
                        "instantaneous_to_sequence_avg": float(s_alpha - i_alpha) if np.isfinite(i_alpha) and np.isfinite(s_alpha) else float("nan"),
                        "grad_to_sequence_avg": float(s_alpha - g_alpha) if np.isfinite(g_alpha) and np.isfinite(s_alpha) else float("nan"),
                    }

                w_summary[str(lag)] = lag_summary

            summary["models"][model_name]["by_projection_w_seed"][str(w_seed)] = w_summary
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                for row in rows_this_w:
                    writer.writerow(row)

            json_path = os.path.join(outdir, "alpha_chain_summary.json")
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            if bool(args.save_raw_samples):
                npz_path = os.path.join(outdir, f"alpha_chain_samples__{model_name}__w_seed_{w_seed}.npz")
                np.savez_compressed(npz_path, **sample_payload)
                print(f"[done] wrote {npz_path}")

            del matched_by_lag, projected_grad_noise, grad_estimates, rows_this_w, sample_payload, w_summary
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[alpha-chain] finished model={model_name} w_seed={w_seed}")
    json_path = os.path.join(outdir, "alpha_chain_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {json_path}")


if __name__ == "__main__":
    main()
