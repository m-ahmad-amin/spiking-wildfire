import math

import torch

from spiking_wildfire.lif import (
    LifConfig,
    gaussian_kernel,
    integrate,
    run_series,
)


def _cfg(**kwargs) -> LifConfig:
    base = dict(
        tau_hours=24,
        v_th=1,
        v_reset=0,
        refrac_hours=3,
        kernel=3,
        couple_sigma=1,
        couple_weight=0.2,
    )
    base.update(kwargs)
    return LifConfig(**base)


def test_decay_without_spike():
    cfg = _cfg(tau_hours=1, v_th=10, refrac_hours=0, couple_weight=0)
    v = torch.ones(1, 1)
    refrac = torch.zeros(1, 1, dtype=torch.int64)
    zeros = torch.zeros(1, 1)
    updated, _refrac, spike = integrate(v, refrac, zeros, zeros, cfg)
    assert torch.allclose(updated, torch.full((1, 1), math.exp(-1)))
    assert not bool(spike)


def test_refractory_blocks_input():
    cfg = _cfg(tau_hours=1e9, v_th=0.5, refrac_hours=2, couple_weight=0)
    current = torch.ones(1, 1)
    zeros = torch.zeros(1, 1)
    v = torch.zeros(1, 1)
    refrac = torch.zeros(1, 1, dtype=torch.int64)
    v, refrac, spike = integrate(v, refrac, current, zeros, cfg)
    assert bool(spike)
    assert float(v) == 0
    for _ in range(2):
        v, refrac, spike = integrate(v, refrac, current, zeros, cfg)
        assert not bool(spike)
        assert float(v) == 0
    v, refrac, spike = integrate(v, refrac, current, zeros, cfg)
    assert bool(spike)


def test_neighbor_receives_spike():
    cfg = _cfg(tau_hours=1e9, v_th=0.5, refrac_hours=0, couple_weight=1)
    first = torch.zeros(3, 3)
    first[1, 1] = 1
    second = torch.zeros(3, 3)
    membranes = run_series([first, second], cfg)
    assert float(membranes[0][1, 1]) == 0
    assert float(membranes[1][1, 0]) > 0
    assert float(membranes[1][1, 1]) == 0
    assert float(gaussian_kernel(cfg)[1, 1]) == 0


def test_sparse_matches_dense_and_ignores_the_future():
    cfg = _cfg()
    torch.manual_seed(0)
    currents = [torch.rand(5, 5) for _ in range(8)]
    dense = run_series(currents, cfg, sparse=False)
    sparse = run_series(currents, cfg, sparse=True)
    for left, right in zip(dense, sparse):
        assert torch.allclose(left, right, atol=1e-5)
    later = currents[:4] + [torch.ones(5, 5) * 3]
    full = run_series(later, cfg)
    prefix = run_series(currents[:4], cfg)
    assert torch.allclose(full[3], prefix[-1])
