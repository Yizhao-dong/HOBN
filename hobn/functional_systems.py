import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .atlas import AtlasSpec


SYSTEM_NAMES = [
    "default_mode",
    "frontoparietal_control",
    "dorsal_attention",
    "ventral_attention_salience",
    "sensorimotor",
    "visual",
    "auditory",
    "subcortical_limbic",
]


AAL116_NAMES = [
    "Precentral_L", "Precentral_R",
    "Frontal_Sup_L", "Frontal_Sup_R",
    "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R",
    "Frontal_Mid_L", "Frontal_Mid_R",
    "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R",
    "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R",
    "Frontal_Inf_Orb_L", "Frontal_Inf_Orb_R",
    "Rolandic_Oper_L", "Rolandic_Oper_R",
    "Supp_Motor_Area_L", "Supp_Motor_Area_R",
    "Olfactory_L", "Olfactory_R",
    "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
    "Frontal_Med_Orb_L", "Frontal_Med_Orb_R",
    "Rectus_L", "Rectus_R",
    "Cingulum_Ant_L", "Cingulum_Ant_R",
    "Cingulum_Mid_L", "Cingulum_Mid_R",
    "Cingulum_Post_L", "Cingulum_Post_R",
    "Insula_L", "Insula_R",
    "Calcarine_L", "Calcarine_R",
    "Cuneus_L", "Cuneus_R",
    "Lingual_L", "Lingual_R",
    "Occipital_Sup_L", "Occipital_Sup_R",
    "Occipital_Mid_L", "Occipital_Mid_R",
    "Occipital_Inf_L", "Occipital_Inf_R",
    "Fusiform_L", "Fusiform_R",
    "Postcentral_L", "Postcentral_R",
    "Parietal_Sup_L", "Parietal_Sup_R",
    "Parietal_Inf_L", "Parietal_Inf_R",
    "SupraMarginal_L", "SupraMarginal_R",
    "Angular_L", "Angular_R",
    "Precuneus_L", "Precuneus_R",
    "Paracentral_Lobule_L", "Paracentral_Lobule_R",
    "Caudate_L", "Caudate_R",
    "Putamen_L", "Putamen_R",
    "Pallidum_L", "Pallidum_R",
    "Thalamus_L", "Thalamus_R",
    "Hippocampus_L", "Hippocampus_R",
    "ParaHippocampal_L", "ParaHippocampal_R",
    "Amygdala_L", "Amygdala_R",
    "Heschl_L", "Heschl_R",
    "Temporal_Sup_L", "Temporal_Sup_R",
    "Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R",
    "Temporal_Mid_L", "Temporal_Mid_R",
    "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R",
    "Temporal_Inf_L", "Temporal_Inf_R",
    "Cerebelum_Crus1_L", "Cerebelum_Crus1_R",
    "Cerebelum_Crus2_L", "Cerebelum_Crus2_R",
    "Cerebelum_3_L", "Cerebelum_3_R",
    "Cerebelum_4_5_L", "Cerebelum_4_5_R",
    "Cerebelum_6_L", "Cerebelum_6_R",
    "Cerebelum_7b_L", "Cerebelum_7b_R",
    "Cerebelum_8_L", "Cerebelum_8_R",
    "Cerebelum_9_L", "Cerebelum_9_R",
    "Cerebelum_10_L", "Cerebelum_10_R",
    "Vermis_1_2", "Vermis_3", "Vermis_4_5", "Vermis_6",
    "Vermis_7", "Vermis_8", "Vermis_9", "Vermis_10",
]


HOA_BASE_NAMES = [
    "Frontal Pole",
    "Insular Cortex",
    "Superior Frontal Gyrus",
    "Middle Frontal Gyrus",
    "Inferior Frontal Gyrus pars triangularis",
    "Inferior Frontal Gyrus pars opercularis",
    "Precentral Gyrus",
    "Temporal Pole",
    "Superior Temporal Gyrus anterior division",
    "Superior Temporal Gyrus posterior division",
    "Middle Temporal Gyrus anterior division",
    "Middle Temporal Gyrus posterior division",
    "Middle Temporal Gyrus temporooccipital part",
    "Inferior Temporal Gyrus anterior division",
    "Inferior Temporal Gyrus posterior division",
    "Inferior Temporal Gyrus temporooccipital part",
    "Postcentral Gyrus",
    "Superior Parietal Lobule",
    "Inferior Parietal Lobule anterior division",
    "Inferior Parietal Lobule posterior division",
    "Supramarginal Gyrus anterior division",
    "Supramarginal Gyrus posterior division",
    "Angular Gyrus",
    "Precuneous Cortex",
    "Subcallosal Area",
    "Paracingulate Gyrus",
    "Cingulate Gyrus anterior division",
    "Cingulate Gyrus middle division",
    "Cingulate Gyrus posterior division",
    "Frontal Medial Cortex",
    "Frontal Orbital Cortex",
    "Juxtapositional Lobule Cortex formerly Supplementary Motor Area",
    "Parahippocampal Gyrus anterior division",
    "Parahippocampal Gyrus posterior division",
    "Lingual Gyrus",
    "Temporal Fusiform Cortex anterior division",
    "Temporal Fusiform Cortex posterior division",
    "Occipital Fusiform Gyrus",
    "Frontal Operculum Cortex",
    "Central Opercular Cortex",
    "Parietal Operculum Cortex",
    "Occipital Pole",
    "Superior Occipital Cortex",
    "Lateral Occipital Cortex superior division",
    "Lateral Occipital Cortex inferior division",
    "Intracalcarine Cortex",
    "Supracalcarine Cortex",
    "Cuneal Cortex",
    "Putamen",
    "Caudate",
    "Pallidum",
    "Thalamus",
    "Amygdala",
    "Hippocampus",
    "Nucleus Accumbens",
    "Brainstem",
]


def _bilateral(names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for name in names:
        out.append(name + " Left")
        out.append(name + " Right")
    return out


HOA_DEFAULT_NAMES = _bilateral(HOA_BASE_NAMES)


@dataclass
class FunctionalGroupResult:
    atlas_groups: Dict[str, np.ndarray]
    roi_names: Dict[str, List[str]]
    system_names: List[str]
    notes: List[str]


def normalize_name(name: str) -> str:
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\b(left|right|l|r)\b", " ", name, flags=re.IGNORECASE)
    name = re.sub(r"[^a-zA-Z0-9 ]+", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def _has_any(text: str, keys: Sequence[str]) -> bool:
    return any(key in text for key in keys)


def classify_region_name(name: str, systems: Sequence[str]) -> int:
    text = normalize_name(name)
    system_to_id = {name: idx for idx, name in enumerate(systems)}

    rules: List[Tuple[str, List[str]]] = [
        ("subcortical_limbic", [
            "thalam", "caudate", "putamen", "pallid", "accumbens", "amygdala",
            "hippocamp", "parahippocamp", "brainstem", "cerebel", "vermis",
            "subcallosal", "limbic",
        ]),
        ("visual", [
            "occipital", "calcar", "cune", "lingual", "fusiform", "visual",
            "supracalcarine", "intracalcarine",
        ]),
        ("auditory", [
            "heschl", "auditory", "superior temporal", "planum temporale",
            "temporal pole sup", "temporal sup",
        ]),
        ("sensorimotor", [
            "precentral", "postcentral", "paracentral", "supp motor",
            "supplementary motor", "juxtapositional", "rolandic", "central oper",
            "motor", "somato",
        ]),
        ("ventral_attention_salience", [
            "insula", "cingulate anterior", "anterior cingulate", "operculum",
            "opercular", "frontal oper", "parietal oper", "salience",
            "cingulum ant", "cingulum mid",
        ]),
        ("default_mode", [
            "precune", "angular", "cingulate posterior", "posterior cingulate",
            "frontal medial", "medial frontal", "temporal mid", "middle temporal",
            "temporal inf", "inferior temporal", "rectus", "olfactory",
            "cingulum post", "frontal sup medial", "frontal med", "temporal pole mid",
        ]),
        ("dorsal_attention", [
            "superior parietal", "inferior parietal", "supramarginal",
            "parietal", "frontal eye", "dorsal attention",
        ]),
        ("frontoparietal_control", [
            "frontal", "middle frontal", "inferior frontal", "superior frontal",
            "orbital", "orbitofrontal", "frontoparietal", "control",
        ]),
    ]
    for system, keywords in rules:
        if system in system_to_id and _has_any(text, keywords):
            return system_to_id[system]
    return system_to_id.get("default_mode", 0)


def default_roi_names(atlas: AtlasSpec) -> Tuple[List[str], str]:
    if atlas.name == "aal":
        names = AAL116_NAMES[:atlas.size]
        if len(names) < atlas.size:
            names.extend([f"AAL_ROI_{idx + 1}" for idx in range(len(names), atlas.size)])
        return names, "standard AAL116 names"
    if atlas.name == "ho":
        names = HOA_DEFAULT_NAMES[:atlas.size]
        if len(names) < atlas.size:
            names.extend([f"HOA_ROI_{idx + 1}" for idx in range(len(names), atlas.size)])
        return names, "Harvard-Oxford anatomical names truncated to data ROI count"
    return [f"CC200_ROI_{idx + 1:03d}" for idx in range(atlas.size)], "CC200 indices; no anatomical names bundled"


def _load_label_file(path: str, expected_size: int) -> List[str]:
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit():
                    names.append(parts[1])
                else:
                    names.append(parts[0])
            else:
                names.append(line)
    if len(names) != expected_size:
        raise ValueError(f"Label file {path} has {len(names)} names, expected {expected_size}.")
    return names


def named_groups_from_roi_names(names: Sequence[str], systems: Sequence[str]) -> np.ndarray:
    return np.asarray([classify_region_name(name, systems) for name in names], dtype=np.int64)


def infer_target_groups_from_named_atlases(
    corr_train: np.ndarray,
    specs: Sequence[AtlasSpec],
    source_groups: Dict[str, np.ndarray],
    target: AtlasSpec,
    systems: Sequence[str],
) -> np.ndarray:
    n_systems = len(systems)
    source_indices_by_system: Dict[int, List[int]] = {idx: [] for idx in range(n_systems)}
    for spec in specs:
        if spec.name == target.name:
            continue
        groups = source_groups.get(spec.name)
        if groups is None:
            continue
        for local_idx, group_id in enumerate(groups):
            source_indices_by_system[int(group_id)].append(spec.start + local_idx)

    labels = np.zeros(target.size, dtype=np.int64)
    abs_corr = np.abs(corr_train)
    for local_idx in range(target.size):
        global_idx = target.start + local_idx
        scores = np.full(n_systems, -np.inf, dtype=np.float64)
        for system_id, src_indices in source_indices_by_system.items():
            if src_indices:
                scores[system_id] = float(np.nanmean(abs_corr[:, global_idx, src_indices]))
        if np.isfinite(scores).any():
            labels[local_idx] = int(np.nanargmax(scores))
    return labels


def build_functional_group_result(
    specs: Sequence[AtlasSpec],
    systems: Sequence[str],
    cfg: Optional[Dict] = None,
    corr: Optional[np.ndarray] = None,
    train_idx: Optional[np.ndarray] = None,
) -> FunctionalGroupResult:
    cfg = cfg or {}
    label_files = cfg.get("label_files", {}) or {}
    cc200_strategy = cfg.get("cc200_strategy", "train_fc_to_named_atlases")
    roi_names: Dict[str, List[str]] = {}
    atlas_groups: Dict[str, np.ndarray] = {}
    notes: List[str] = []

    for spec in specs:
        if spec.name in label_files and label_files[spec.name]:
            names = _load_label_file(label_files[spec.name], spec.size)
            source = f"label file: {label_files[spec.name]}"
        else:
            names, source = default_roi_names(spec)
        roi_names[spec.name] = names
        if spec.name != "cc200":
            atlas_groups[spec.name] = named_groups_from_roi_names(names, systems)
        notes.append(f"{spec.name}: {source}")

    cc200 = next((spec for spec in specs if spec.name == "cc200"), None)
    if cc200 is not None:
        if "cc200" in label_files and label_files["cc200"]:
            atlas_groups["cc200"] = named_groups_from_roi_names(roi_names["cc200"], systems)
            notes.append("cc200: classified by provided ROI names")
        elif cc200_strategy == "train_fc_to_named_atlases" and corr is not None and train_idx is not None:
            atlas_groups["cc200"] = infer_target_groups_from_named_atlases(
                corr[train_idx],
                specs,
                atlas_groups,
                cc200,
                systems,
            )
            notes.append("cc200: inferred by train-fold FC similarity to named AAL/HO systems")
        else:
            atlas_groups["cc200"] = np.asarray(
                [idx % len(systems) for idx in range(cc200.size)], dtype=np.int64
            )
            notes.append("cc200: uniform cyclic assignment without ROI names or training correlations")

    return FunctionalGroupResult(atlas_groups, roi_names, list(systems), notes)


def summarize_group_counts(result: FunctionalGroupResult) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for atlas_name, labels in result.atlas_groups.items():
        summary[atlas_name] = {}
        for idx, system in enumerate(result.system_names):
            summary[atlas_name][system] = int(np.sum(labels == idx))
    return summary


def save_group_report(result: FunctionalGroupResult, output_dir: str, fold_id: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "systems": result.system_names,
        "counts": summarize_group_counts(result),
        "notes": result.notes,
        "roi_assignments": {},
    }
    for atlas_name, labels in result.atlas_groups.items():
        payload["roi_assignments"][atlas_name] = [
            {
                "roi": idx,
                "name": result.roi_names[atlas_name][idx],
                "system": result.system_names[int(label)],
            }
            for idx, label in enumerate(labels)
        ]
    path = os.path.join(output_dir, f"functional_groups_fold{fold_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
