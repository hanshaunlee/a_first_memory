from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FeatureFamily(str, Enum):
    SEMANTIC = "semantic"
    VLM_SEMANTIC = "vlm_semantic"
    OBJECT = "object"
    OBJECT_PART = "object_part"
    SCENE_GRAPH = "scene_graph"
    OCR_TEXT = "ocr_text"
    FACE_BODY = "face_body"
    GEOMETRY = "geometry"
    DEPTH = "depth"
    SURFACE_NORMAL = "surface_normal"
    LOW_LEVEL = "low_level"
    TEXTURE = "texture"
    COLOR = "color"
    SPATIAL_FREQUENCY = "spatial_frequency"
    EDGE_SHAPE = "edge_shape"
    PATCH = "patch"
    SALIENCY = "saliency"


@dataclass(frozen=True)
class CandidateUnit:
    image_id: int
    unit_id: int
    family: FeatureFamily
    embedding: np.ndarray
    cost: float


@dataclass(frozen=True)
class ExposureRecord:
    image_id: int
    exposure_idx: int
    lag_bucket: int
    repeat_count: int
    voxel_response: np.ndarray


@dataclass(frozen=True)
class DelayedProbe:
    image_id: int
    semantic_target: int
    perceptual_target: int
