from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from spiking_wildfire.lif import (
    LifConfig,
    gaussian_kernel,
    integrate,
    scatter_current,
    sparse_integrate,
    spatial_input,
    torch_device,
    zeros_state,
)


@dataclass
class Step:
    t: int
    v: torch.Tensor
    spikes: torch.Tensor
    spatial: torch.Tensor
    rate24: torch.Tensor
    rate7: torch.Tensor
    neighbor: torch.Tensor
    active_days: torch.Tensor
    updates: int


def _neighbor(rate: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(3, 3, device=rate.device)
    kernel[1, 1] = 0
    return spatial_input(rate, kernel) / 8


def simulate(hourly, n_lat: int, n_lon: int, cfg: LifConfig, sparse: bool, device=None):
    if not isinstance(device, torch.device):
        device = torch_device(device)
    kernel = gaussian_kernel(cfg, device)
    v, refrac, prev = zeros_state(n_lat, n_lon, device)
    rate24 = torch.zeros(n_lat, n_lon, device=device)
    rate7 = torch.zeros(n_lat, n_lon, device=device)
    recent24: deque[torch.Tensor] = deque()
    recent7: deque[torch.Tensor] = deque()
    day_hits: deque[torch.Tensor] = deque()
    today = torch.zeros(n_lat, n_lon, device=device)
    today_index = 0
    for t, (ii, jj, values) in enumerate(hourly):
        current = scatter_current(n_lat, n_lon, ii, jj, values, device)
        spatial = spatial_input(prev, kernel)
        if sparse:
            v, refrac, spike, updates = sparse_integrate(
                v, refrac, current, spatial, prev, cfg
            )
        else:
            v, refrac, spike = integrate(v, refrac, current, spatial, cfg)
            updates = int(v.numel())
        fired = spike.float()
        prev = fired
        recent24.append(fired)
        rate24 = rate24 + fired
        if len(recent24) > 24:
            rate24 = rate24 - recent24.popleft()
        recent7.append(fired)
        rate7 = rate7 + fired
        if len(recent7) > 168:
            rate7 = rate7 - recent7.popleft()
        day = t // 24
        if day != today_index:
            day_hits.append(today)
            if len(day_hits) > 30:
                day_hits.popleft()
            today = torch.zeros(n_lat, n_lon, device=device)
            today_index = day
        today = torch.maximum(today, fired)
        active = today.clone()
        for item in day_hits:
            active = active + item
        yield Step(
            t=t,
            v=v,
            spikes=fired,
            spatial=spatial,
            rate24=rate24,
            rate7=rate7,
            neighbor=_neighbor(rate24),
            active_days=active,
            updates=updates,
        )


def recent_mask(hourly, t: int, n_lat: int, n_lon: int, window: int) -> np.ndarray:
    mask = np.zeros((n_lat, n_lon), dtype=bool)
    for step in range(max(0, t - window + 1), t + 1):
        ii, jj, _values = hourly[step]
        if len(ii):
            mask[ii, jj] = True
    return mask


def future_mask(hourly, t: int, horizon: int, n_lat: int, n_lon: int) -> np.ndarray:
    mask = np.zeros((n_lat, n_lon), dtype=bool)
    last = min(len(hourly) - 1, t + horizon)
    for step in range(t + 1, last + 1):
        ii, jj, _values = hourly[step]
        if len(ii):
            mask[ii, jj] = True
    return mask


def detection_matrix(step: Step, cells: np.ndarray) -> np.ndarray:
    v = step.v.detach().cpu().numpy()
    rate24 = step.rate24.detach().cpu().numpy() / 24
    rate7 = step.rate7.detach().cpu().numpy() / 168
    neighbor = step.neighbor.detach().cpu().numpy() / 24
    active = step.active_days.detach().cpu().numpy()
    ii = cells[:, 0]
    jj = cells[:, 1]
    return np.column_stack(
        [v[ii, jj], rate24[ii, jj], rate7[ii, jj], neighbor[ii, jj], active[ii, jj]]
    )


def forecast_matrix(step: Step, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            step.v.detach().cpu().numpy()[ii, jj],
            step.spatial.detach().cpu().numpy()[ii, jj],
            (step.rate24.detach().cpu().numpy() / 24)[ii, jj],
        ]
    )


def spike_window(steps_spikes: deque[np.ndarray]) -> np.ndarray:
    mask = np.zeros_like(steps_spikes[0], dtype=bool)
    for item in steps_spikes:
        mask |= item
    return mask
