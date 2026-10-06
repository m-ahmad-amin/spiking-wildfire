import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def torch_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class LifConfig:
    tau_hours: float
    v_th: float
    v_reset: float
    refrac_hours: int
    kernel: int
    couple_sigma: float
    couple_weight: float

    @classmethod
    def from_dict(cls, raw: dict) -> "LifConfig":
        return cls(
            tau_hours=float(raw["tau_hours"]),
            v_th=float(raw["v_th"]),
            v_reset=float(raw["v_reset"]),
            refrac_hours=int(raw["refrac_hours"]),
            kernel=int(raw["kernel"]),
            couple_sigma=float(raw["couple_sigma"]),
            couple_weight=float(raw["couple_weight"]),
        )


def gaussian_kernel(cfg: LifConfig, device: torch.device | None = None) -> torch.Tensor:
    k = cfg.kernel
    if k % 2 == 0:
        raise ValueError("kernel size must be odd")
    axis = torch.arange(k, dtype=torch.float32, device=device) - (k // 2)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    sigma = max(cfg.couple_sigma, 1e-6)
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel[k // 2, k // 2] = 0
    total = float(kernel.sum())
    if total == 0 or cfg.couple_weight == 0:
        return torch.zeros_like(kernel)
    return kernel / total * cfg.couple_weight


def spatial_input(prev_spikes: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    pad = kernel.shape[0] // 2
    return F.conv2d(
        prev_spikes[None, None],
        kernel[None, None],
        padding=pad,
    )[0, 0]


def decay_factor(cfg: LifConfig) -> float:
    return math.exp(-1.0 / cfg.tau_hours)


def integrate(v, refrac, current, spatial, cfg: LifConfig):
    decay = decay_factor(cfg)
    active = refrac == 0
    updated = torch.where(active, decay * v + current + spatial, v)
    spike = active & (updated >= cfg.v_th)
    reset = torch.full_like(updated, cfg.v_reset)
    updated = torch.where(spike, reset, updated)
    held = torch.full_like(refrac, cfg.refrac_hours)
    refrac_out = torch.where(spike, held, torch.clamp(refrac - 1, min=0))
    return updated, refrac_out, spike


def hot_mask(v, refrac, current, prev_spikes, radius: int) -> torch.Tensor:
    occupied = prev_spikes > 0
    if bool(occupied.any().item()) and radius > 0:
        dilated = F.max_pool2d(
            occupied[None, None].float(),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )[0, 0] > 0
    else:
        dilated = occupied
    return (v.abs() > 1e-8) | (refrac > 0) | (current.abs() > 0) | dilated


def sparse_integrate(v, refrac, current, spatial, prev_spikes, cfg: LifConfig):
    radius = cfg.kernel // 2
    hot = hot_mask(v, refrac, current, prev_spikes, radius)
    updated = v.clone()
    refrac_out = refrac.clone()
    spike = torch.zeros_like(v, dtype=torch.bool)
    count = int(hot.sum().item())
    if count:
        decay = decay_factor(cfg)
        active = hot & (refrac == 0)
        updated[active] = decay * v[active] + current[active] + spatial[active]
        spike[active] = updated[active] >= cfg.v_th
        updated[spike] = cfg.v_reset
        refrac_out[hot] = torch.where(
            spike[hot],
            torch.full((count,), cfg.refrac_hours, dtype=refrac.dtype, device=v.device),
            torch.clamp(refrac[hot] - 1, min=0),
        )
    return updated, refrac_out, spike, count


def zeros_state(n_lat: int, n_lon: int, device: torch.device | None = None):
    device = device or torch.device("cpu")
    return (
        torch.zeros(n_lat, n_lon, device=device),
        torch.zeros(n_lat, n_lon, dtype=torch.int64, device=device),
        torch.zeros(n_lat, n_lon, device=device),
    )


def scatter_current(n_lat: int, n_lon: int, i, j, values, device: torch.device | None = None) -> torch.Tensor:
    device = device or torch.device("cpu")
    current = torch.zeros(n_lat, n_lon, device=device)
    if len(i):
        current[
            torch.as_tensor(i, dtype=torch.long, device=device),
            torch.as_tensor(j, dtype=torch.long, device=device),
        ] = torch.as_tensor(values, dtype=torch.float32, device=device)
    return current


def run_series(currents: list[torch.Tensor], cfg: LifConfig, sparse: bool = False):
    device = currents[0].device
    kernel = gaussian_kernel(cfg, device)
    v, refrac, prev = zeros_state(*currents[0].shape, device)
    membranes = []
    for current in currents:
        spatial = spatial_input(prev, kernel)
        if sparse:
            v, refrac, spike, _updates = sparse_integrate(
                v, refrac, current, spatial, prev, cfg
            )
        else:
            v, refrac, spike = integrate(v, refrac, current, spatial, cfg)
        prev = spike.float()
        membranes.append(v.clone())
    return membranes
