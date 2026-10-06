import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self, gate_cells: float, expire_hours: int):
        self.gate = float(gate_cells)
        self.expire = int(expire_hours)
        self.next_id = 1
        self.active: dict[int, dict] = {}

    def update(self, mask: np.ndarray) -> np.ndarray:
        labeled, count = ndimage.label(mask.astype(bool))
        centroids = []
        if count:
            centroids = ndimage.center_of_mass(mask.astype(float), labeled, range(1, count + 1))
        old_ids = list(self.active)
        matched_old = set()
        matched_new = set()
        pairs: list[tuple[int, int]] = []
        if old_ids and count:
            cost = np.zeros((len(old_ids), count), dtype=float)
            for row, track_id in enumerate(old_ids):
                oi = self.active[track_id]["i"]
                oj = self.active[track_id]["j"]
                for col in range(count):
                    ci, cj = centroids[col]
                    cost[row, col] = float(np.hypot(oi - ci, oj - cj))
            cost[cost > self.gate] = 1e6
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if cost[row, col] >= 1e6:
                    continue
                pairs.append((old_ids[row], col + 1))
                matched_old.add(old_ids[row])
                matched_new.add(col + 1)
        assignment = {}
        for track_id, component in pairs:
            ci, cj = centroids[component - 1]
            self.active[track_id] = {"i": float(ci), "j": float(cj), "gap": 0}
            assignment[component] = track_id
        for component in range(1, count + 1):
            if component in matched_new:
                continue
            ci, cj = centroids[component - 1]
            track_id = self.next_id
            self.next_id += 1
            self.active[track_id] = {"i": float(ci), "j": float(cj), "gap": 0}
            assignment[component] = track_id
        for track_id in old_ids:
            if track_id in matched_old:
                continue
            self.active[track_id]["gap"] += 1
            if self.active[track_id]["gap"] > self.expire:
                del self.active[track_id]
        id_map = np.zeros(mask.shape, dtype=np.int32)
        for component, track_id in assignment.items():
            id_map[labeled == component] = track_id
        return id_map


def region_overlap(id_map: np.ndarray, perimeter: np.ndarray, previous: dict[int, int]):
    pred = id_map > 0
    truth = perimeter.astype(bool)
    union = np.logical_or(pred, truth).sum()
    iou = float(np.logical_and(pred, truth).sum() / union) if union else 1.0
    gt, count = ndimage.label(truth)
    switches = 0
    current: dict[int, int] = {}
    if count and pred.any():
        for component in range(1, count + 1):
            region = gt == component
            ids, counts = np.unique(id_map[region], return_counts=True)
            keep = ids > 0
            if not keep.any():
                continue
            track_id = int(ids[keep][np.argmax(counts[keep])])
            current[component] = track_id
            prior = previous.get(component)
            if prior is not None and prior != track_id:
                switches += 1
    return iou, switches, current


class RegionMemory:
    def __init__(self):
        self.ids = None
        self.next_id = 1
        self.track_of: dict[int, int] = {}

    def observe(self, perimeter: np.ndarray, id_map: np.ndarray) -> int:
        labeled, count = ndimage.label(perimeter.astype(bool))
        assigned = np.zeros_like(labeled)
        switches = 0
        next_track = {}
        for component in range(1, count + 1):
            region = labeled == component
            region_id = self._inherit(region)
            assigned[region] = region_id
            ids, counts = np.unique(id_map[region], return_counts=True)
            keep = ids > 0
            if not keep.any():
                continue
            track_id = int(ids[keep][np.argmax(counts[keep])])
            previous = self.track_of.get(region_id)
            if previous is not None and previous != track_id:
                switches += 1
            next_track[region_id] = track_id
        self.ids = assigned
        self.track_of = next_track
        return switches

    def _inherit(self, region: np.ndarray) -> int:
        if self.ids is None:
            region_id = self.next_id
            self.next_id += 1
            return region_id
        found, counts = np.unique(self.ids[region], return_counts=True)
        keep = found > 0
        if keep.any():
            return int(found[keep][np.argmax(counts[keep])])
        region_id = self.next_id
        self.next_id += 1
        return region_id
