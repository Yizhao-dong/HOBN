from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class AtlasSpec:
    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


def infer_atlas_specs(total_rois: int, aal_size: int = 116, cc200_size: int = 200) -> List[AtlasSpec]:
    """Infer AAL/HO/CC200 slices from a concatenated ROI layout."""
    ho_size = total_rois - aal_size - cc200_size
    if ho_size <= 0:
        raise ValueError(
            f"Cannot infer HO size from total_rois={total_rois}, "
            f"aal_size={aal_size}, cc200_size={cc200_size}."
        )
    return [
        AtlasSpec("aal", 0, aal_size),
        AtlasSpec("ho", aal_size, aal_size + ho_size),
        AtlasSpec("cc200", aal_size + ho_size, total_rois),
    ]


def slice_atlas_tensor(x: torch.Tensor, specs: Sequence[AtlasSpec], atlas: AtlasSpec) -> torch.Tensor:
    if x.dim() == 3 and x.shape[1] == x.shape[2]:
        return x[:, atlas.start:atlas.end, atlas.start:atlas.end]
    if x.dim() == 3:
        return x[:, atlas.start:atlas.end, :]
    raise ValueError(f"Unsupported atlas tensor shape: {tuple(x.shape)}")


def contiguous_functional_groups(specs: Sequence[AtlasSpec], systems: Sequence[str]) -> Dict[str, np.ndarray]:
    """Create an eight-system ROI grouping for each atlas.

    Exact ROI-to-system labels can be supplied through label files in the config.
    When labels are unavailable, this function provides a deterministic grouping
    for the FGHM interface.
    """
    groups: Dict[str, np.ndarray] = {}
    n_systems = len(systems)
    for spec in specs:
        labels = np.zeros(spec.size, dtype=np.int64)
        splits = np.array_split(np.arange(spec.size), n_systems)
        for group_id, roi_ids in enumerate(splits):
            labels[roi_ids] = group_id
        groups[spec.name] = labels
    return groups


def make_cross_atlas_group_labels(
    specs: Sequence[AtlasSpec],
    systems: Sequence[str],
    atlas_groups: Optional[Dict[str, np.ndarray]] = None,
) -> torch.Tensor:
    if atlas_groups is None:
        atlas_groups = contiguous_functional_groups(specs, systems)

    labels = []
    n_systems = len(systems)
    for atlas_id, spec in enumerate(specs):
        atlas_label = np.asarray(atlas_groups[spec.name], dtype=np.int64)
        if atlas_label.shape[0] != spec.size:
            raise ValueError(
                f"Functional group size mismatch for {spec.name}: "
                f"expected {spec.size}, got {atlas_label.shape[0]}"
            )
        labels.append(atlas_id * n_systems + atlas_label)
    return torch.as_tensor(np.concatenate(labels), dtype=torch.long)
