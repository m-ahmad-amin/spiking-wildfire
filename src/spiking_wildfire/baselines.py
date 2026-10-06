import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier

from spiking_wildfire.readout import average_precision


def _hours(events: pd.DataFrame) -> pd.Series:
    return events["time"].dt.floor("h")


def _cell_clocks(events: pd.DataFrame):
    frame = events.copy()
    frame["hour"] = _hours(frame)
    clocks = {}
    frp = {}
    for i, j, hour, value in zip(frame["i"], frame["j"], frame["hour"], frame["frp"].fillna(0)):
        cell = (int(i), int(j))
        stamp = np.datetime64(pd.Timestamp(hour), "h")
        clocks.setdefault(cell, set()).add(stamp)
        key = (cell, stamp)
        frp[key] = frp.get(key, 0.0) + float(value)
    ordered = {cell: np.array(sorted(stamps)) for cell, stamps in clocks.items()}
    return ordered, frp


def _quiet_future(times: np.ndarray, hour: np.datetime64, horizon_hours: int):
    recent_start = hour - np.timedelta64(24, "h")
    lo = int(np.searchsorted(times, recent_start, side="right"))
    hi = int(np.searchsorted(times, hour, side="right"))
    if lo != hi:
        return None
    future_end = hour + np.timedelta64(horizon_hours, "h")
    left = int(np.searchsorted(times, hour, side="right"))
    right = int(np.searchsorted(times, future_end, side="right"))
    return left < right


def _row(cell, hour, times, frp):
    i, j = cell
    neighbor = 0.0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbor += frp.get(((i + di, j + dj), hour), 0.0)
    prior = int(np.searchsorted(times, hour, side="left"))
    if prior == 0:
        since = 24.0 * 60
        past = 0
    else:
        since = float((hour - times[prior - 1]) / np.timedelta64(1, "h"))
        cutoff = hour - np.timedelta64(30, "D")
        past = prior - int(np.searchsorted(times, cutoff, side="left"))
    return [0.0, 0.0, neighbor, int(pd.Timestamp(hour).month), since, float(past)]


def _forecast_rows(events, start, end, horizon_hours, seed, cap):
    clocks, frp = _cell_clocks(events)
    if not clocks:
        return None
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for hour in pd.date_range(start, end, freq="24h"):
        stamp = np.datetime64(pd.Timestamp(hour), "h")
        positive = []
        negative = []
        for cell, times in clocks.items():
            flag = _quiet_future(times, stamp, horizon_hours)
            if flag is None:
                continue
            (positive if flag else negative).append(cell)
        if not positive:
            continue
        take = min(len(negative), 5 * len(positive))
        chosen = rng.choice(len(negative), size=take, replace=False) if take else []
        picked = positive + [negative[index] for index in np.atleast_1d(chosen)]
        marks = [1] * len(positive) + [0] * int(take)
        for cell, mark in zip(picked, marks):
            rows.append(_row(cell, stamp, clocks[cell], frp))
            labels.append(mark)
        if len(labels) >= cap:
            break
    if len(set(labels)) < 2:
        return None
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def tabular_forecast(events, train_start, train_end, test_start, test_end, horizon_hours, seed):
    train = _forecast_rows(events, train_start, train_end, horizon_hours, seed, 20000)
    test = _forecast_rows(events, test_start, test_end, horizon_hours, seed, 20000)
    if train is None or test is None:
        return None
    model = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    model.fit(train[0], train[1])
    scores = model.predict_proba(test[0])[:, 1]
    return average_precision(test[1], scores)


def cell_sequences(events: pd.DataFrame, cells: int, length: int, seed: int, start, end):
    frame = events.copy()
    frame["hour"] = _hours(frame)
    window = frame[(frame["hour"] >= start) & (frame["hour"] <= end)]
    activity = window.groupby(["i", "j"]).size().sort_values(ascending=False).head(cells)
    tensors = []
    labels = []
    rng = np.random.default_rng(seed)
    for (i, j), _count in activity.items():
        part = frame[(frame["i"] == i) & (frame["j"] == j)]
        hours = window.loc[(window["i"] == i) & (window["j"] == j), "hour"].drop_duplicates()
        if hours.empty:
            continue
        for hour in hours.sample(n=min(4, len(hours)), random_state=int(rng.integers(1_000_000))):
            window_values = []
            for step in range(length, 0, -1):
                stamp = hour - pd.Timedelta(hours=step)
                value = part.loc[part["hour"] == stamp, "frp"].sum()
                window_values.append(np.log1p(value))
            future = ((part["hour"] > hour) & (part["hour"] <= hour + pd.Timedelta(hours=24))).any()
            tensors.append(window_values)
            labels.append(int(future))
    if len(set(labels)) < 2:
        return None
    x = torch.tensor(tensors, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(labels, dtype=torch.float32)
    return x, y


class _LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, batch_first=True)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        _seq, (hidden, _cell) = self.lstm(x)
        return self.out(hidden[-1]).squeeze(-1)


def lstm_forecast(events, seed, train_start, train_end, test_start, test_end):
    train = cell_sequences(events, 200, 24, seed, train_start, train_end)
    test = cell_sequences(events, 200, 24, seed + 1, test_start, test_end)
    if train is None or test is None:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = train[0].to(device), train[1].to(device)
    torch.manual_seed(seed)
    model = _LSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = torch.sigmoid(model(test[0].to(device))).cpu().numpy()
    return average_precision(test[1].numpy(), scores)


class _ConvLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = 4
        self.conv = nn.Conv2d(1 + self.hidden, 4 * self.hidden, 3, padding=1)
        self.out = nn.Conv2d(self.hidden, 1, 1)

    def forward(self, x):
        batch, _steps, height, width = x.shape
        hidden = x.new_zeros(batch, self.hidden, height, width)
        cell = x.new_zeros(batch, self.hidden, height, width)
        for step in range(x.shape[1]):
            gates = self.conv(torch.cat([x[:, step : step + 1], hidden], dim=1))
            enter, forget, candidate, output = gates.chunk(4, dim=1)
            enter = torch.sigmoid(enter)
            forget = torch.sigmoid(forget)
            candidate = torch.tanh(candidate)
            output = torch.sigmoid(output)
            cell = forget * cell + enter * candidate
            hidden = output * torch.tanh(cell)
        return self.out(hidden).squeeze(1)


def convlstm_forecast(hourly, n_lat: int, n_lon: int, seed: int, train_end_index: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _ConvLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    usable = [t for t in range(24, len(hourly) - 1) if len(hourly[t][0])]
    train_pool = [t for t in usable if t <= train_end_index]
    test_pool = [t for t in usable if t > train_end_index]
    if len(train_pool) < 4 or len(test_pool) < 2:
        return None
    picks = rng.choice(train_pool, size=min(16, len(train_pool)), replace=False)
    checks = rng.choice(test_pool, size=min(12, len(test_pool)), replace=False)
    model.train()
    for _epoch in range(3):
        for t in picks:
            frames = []
            for step in range(t - 6, t):
                grid = np.zeros((n_lat, n_lon), dtype=np.float32)
                ii, jj, values = hourly[step]
                if len(ii):
                    grid[ii, jj] = values
                frames.append(grid)
            target = np.zeros((n_lat, n_lon), dtype=np.float32)
            ii, jj, _values = hourly[t]
            if len(ii):
                target[ii, jj] = 1
            batch = torch.tensor(np.stack(frames)[None], dtype=torch.float32, device=device)
            y = torch.tensor(target[None], dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = loss_fn(model(batch), y)
            loss.backward()
            opt.step()
    model.eval()
    scores = []
    truth = []
    with torch.no_grad():
        for t in checks:
            frames = []
            for step in range(t - 6, t):
                grid = np.zeros((n_lat, n_lon), dtype=np.float32)
                ii, jj, values = hourly[step]
                if len(ii):
                    grid[ii, jj] = values
                frames.append(grid)
            pred = torch.sigmoid(
                model(torch.tensor(np.stack(frames)[None], dtype=torch.float32, device=device))
            )
            scores.append(pred.detach().cpu().numpy().ravel())
            target = np.zeros((n_lat, n_lon), dtype=np.float32)
            ii, jj, _values = hourly[t]
            if len(ii):
                target[ii, jj] = 1
            truth.append(target.ravel())
    return average_precision(np.concatenate(truth), np.concatenate(scores))
