from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SyntheticDataConfig:
    n_images: int = 240
    n_units_per_family: int = 10
    embedding_dim: int = 32
    n_rois: int = 5
    roi_voxels: int = 24
    random_seed: int = 7


@dataclass
class NSDConfig:
    enabled: bool = False
    npz_path: str = ""
    raw_dir: str = ""
    layout_root: str = ""
    feature_npz_path: str = ""
    strict: bool = True


@dataclass
class RLConfig:
    # "mlp" = full neural policy (PyTorch MLP on unit context). "linear" = legacy linear θ·x.
    policy_architecture: str = "mlp"
    policy_hidden_dim: int = 128
    policy_num_hidden_layers: int = 2
    algorithm: str = "grpo"
    budget: float = 32.0
    epochs: int = 50
    learning_rate: float = 0.035
    entropy_coef: float = 0.005
    lagrangian_lr: float = 0.015
    grpo_group_size: int = 6
    grpo_adv_clip: float = 2.5
    grpo_ratio_clip_epsilon: float = 0.2
    grpo_kl_coef: float = 0.02
    grpo_ref_update_interval: int = 2
    grad_clip_norm: float = 5.0
    forward_temperature: float = 0.2
    backward_temperature: float = 0.8
    alpha_recog: float = 1.0
    beta_sem: float = 0.9
    gamma_perc: float = 0.7
    lambda_cost: float = 0.08
    delta_novelty: float = 0.45
    eta_schema: float = 0.35
    non_monotonic_low_level_peak: float = 0.45
    reward_weight_grid: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.8, 0.6),
        (1.0, 1.0, 0.5),
        (1.0, 0.6, 1.0),
    )


@dataclass
class EncodingConfig:
    ridge_alpha: float = 1.0
    banded_alphas: tuple[float, ...] = (0.25, 1.0, 4.0, 16.0)
    banded_max_iter: int = 2
    train_frac: float = 0.8


@dataclass
class RSAConfig:
    train_frac: float = 0.8
    fr_rsa_alpha: float = 1.0


@dataclass
class SubjectEvalConfig:
    enabled: bool = True
    min_images_per_subject: int = 4


@dataclass
class PipelineConfig:
    data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    nsd: NSDConfig = field(default_factory=NSDConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    rsa: RSAConfig = field(default_factory=RSAConfig)
    subject_eval: SubjectEvalConfig = field(default_factory=SubjectEvalConfig)
