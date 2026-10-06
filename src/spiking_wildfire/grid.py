from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    south: float
    north: float
    west: float
    east: float
    step: float

    @property
    def n_lat(self) -> int:
        return int(round((self.north - self.south) / self.step))

    @property
    def n_lon(self) -> int:
        return int(round((self.east - self.west) / self.step))

    def locate(self, lat, lon):
        lat_a = np.asarray(lat, dtype=float)
        lon_a = np.asarray(lon, dtype=float)
        scaled_lat = np.round((lat_a - self.south) / self.step, decimals=8)
        scaled_lon = np.round((lon_a - self.west) / self.step, decimals=8)
        i = np.floor(scaled_lat).astype(int)
        j = np.floor(scaled_lon).astype(int)
        i = np.where(np.isclose(lat_a, self.north), self.n_lat - 1, i)
        j = np.where(np.isclose(lon_a, self.east), self.n_lon - 1, j)
        valid = (
            (lat_a >= self.south)
            & (lat_a <= self.north)
            & (lon_a >= self.west)
            & (lon_a <= self.east)
            & (i >= 0)
            & (i < self.n_lat)
            & (j >= 0)
            & (j < self.n_lon)
        )
        return i, j, valid

    def center(self, i: int, j: int) -> tuple[float, float]:
        return (
            self.south + (i + 0.5) * self.step,
            self.west + (j + 0.5) * self.step,
        )


def grid_from_config(cfg: dict) -> Grid:
    box = cfg["bbox"]
    return Grid(
        south=float(box["south"]),
        north=float(box["north"]),
        west=float(box["west"]),
        east=float(box["east"]),
        step=float(cfg["grid_deg"]),
    )
