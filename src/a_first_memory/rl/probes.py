from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from a_first_memory.data.synthetic import SyntheticDataset


@dataclass
class FrozenProbeHeads:
    recog_model: Ridge
    sem_model: Ridge
    perc_model: Ridge


def _selected_feature_row(dataset: SyntheticDataset, image_id: int, selected: np.ndarray) -> np.ndarray:
    units = dataset.unit_embeddings[image_id] * selected[:, None]
    return units.reshape(-1)


def pretrain_probe_heads(dataset: SyntheticDataset, alpha: float = 1.0) -> FrozenProbeHeads:
    """Pretrain probe heads on static targets, then freeze.

    These heads are intentionally simple linear decoders to keep interpretation clean.
    """
    rows = []
    y_recog = []
    y_sem = []
    y_perc = []
    full_selected = np.ones(dataset.unit_embeddings.shape[1], dtype=float)
    for image_id in dataset.image_ids:
        for exposure_idx in range(3):
            rows.append(_selected_feature_row(dataset, int(image_id), full_selected))
            y_recog.append(float(dataset.hit_rates[int(image_id), exposure_idx]))
            y_sem.append(float(dataset.semantic_targets[int(image_id), exposure_idx]))
            y_perc.append(float(dataset.perceptual_targets[int(image_id), exposure_idx]))
    x = np.vstack(rows)
    recog = Ridge(alpha=alpha).fit(x, np.array(y_recog))
    sem = Ridge(alpha=alpha).fit(x, np.array(y_sem))
    perc = Ridge(alpha=alpha).fit(x, np.array(y_perc))
    return FrozenProbeHeads(recog_model=recog, sem_model=sem, perc_model=perc)


def score_with_frozen_probes(
    probes: FrozenProbeHeads,
    dataset: SyntheticDataset,
    image_id: int,
    selected: np.ndarray,
) -> tuple[float, float, float]:
    row = _selected_feature_row(dataset, image_id, selected)[None, :]
    r_recog = float(probes.recog_model.predict(row)[0])
    r_sem = float(probes.sem_model.predict(row)[0])
    r_perc = float(probes.perc_model.predict(row)[0])
    return r_recog, r_sem, r_perc
