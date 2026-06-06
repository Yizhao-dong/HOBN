from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .atlas import AtlasSpec, infer_atlas_specs


@dataclass
class HOBNData:
    timeseries: np.ndarray
    corr: np.ndarray
    labels: np.ndarray
    site: Optional[np.ndarray]
    atlas_specs: Tuple[AtlasSpec, AtlasSpec, AtlasSpec]


class BrainDataset(Dataset):
    def __init__(self, indices: np.ndarray, timeseries: np.ndarray, corr: np.ndarray, labels: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.timeseries = timeseries.astype(np.float32, copy=False)
        self.corr = corr.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        return (
            torch.tensor(idx, dtype=torch.long),
            torch.from_numpy(self.timeseries[idx]),
            torch.from_numpy(self.corr[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def _get_timeseries_key(data: Dict[str, Any]) -> str:
    if "timeseries" in data:
        return "timeseries"
    if "timeseires" in data:
        return "timeseires"
    raise KeyError("Npy dict must contain `timeseries` or the accepted alternate key `timeseires`.")


def load_hobn_data(path: str, aal_size: int = 116, cc200_size: int = 200, limit_subjects: Optional[int] = None) -> HOBNData:
    raw = np.load(path, allow_pickle=True).item()
    ts = np.asarray(raw[_get_timeseries_key(raw)], dtype=np.float32)
    corr = np.asarray(raw["corr"], dtype=np.float32)
    labels = np.asarray(raw["label"], dtype=np.int64)
    site = np.asarray(raw["site"]) if "site" in raw else None

    if limit_subjects is not None:
        ts = ts[:limit_subjects]
        corr = corr[:limit_subjects]
        labels = labels[:limit_subjects]
        if site is not None:
            site = site[:limit_subjects]

    if ts.ndim != 3:
        raise ValueError(f"Expected timeseries shape [subjects, rois, time], got {ts.shape}.")
    if corr.ndim != 3 or corr.shape[1] != corr.shape[2]:
        raise ValueError(f"Expected corr shape [subjects, rois, rois], got {corr.shape}.")
    if ts.shape[0] != corr.shape[0] or ts.shape[0] != labels.shape[0]:
        raise ValueError("timeseries, corr and labels subject counts do not match.")
    if ts.shape[1] != corr.shape[1]:
        raise ValueError("timeseries and corr ROI counts do not match.")

    specs = tuple(infer_atlas_specs(ts.shape[1], aal_size=aal_size, cc200_size=cc200_size))
    return HOBNData(ts, corr, labels, site, specs)


class FoldStandardizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, x: np.ndarray) -> "FoldStandardizer":
        self.mean = float(np.nanmean(x))
        self.std = float(np.nanstd(x))
        if self.std < 1e-6:
            self.std = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return ((x - self.mean) / self.std).astype(np.float32, copy=False)


def standardize_by_train(timeseries: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    scaler = FoldStandardizer().fit(timeseries[train_idx])
    return scaler.transform(timeseries)
