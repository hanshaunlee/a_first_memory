from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from a_first_memory.config import RLConfig
from a_first_memory.data.synthetic import SyntheticDataset
from a_first_memory.rl.probes import FrozenProbeHeads, score_with_frozen_probes


@dataclass
class PolicyTrainingResult:
    retention_by_image_exposure: np.ndarray
    reward_history: list[float]
    lagrangian_history: list[float]
    training_diagnostics: dict = field(default_factory=dict)


POLICY_CHECKPOINT_FILENAME = "policy_checkpoint.npz"
POLICY_TORCH_FILENAME = "policy_checkpoint.pt"


def load_policy_checkpoint(path: str | Path) -> dict[str, np.ndarray | float | int]:
    """Load `policy_checkpoint.npz` written by `run_pipeline`."""
    path = Path(path)
    with np.load(path) as data:
        theta = np.asarray(data["theta"], dtype=np.float64)
        lagrangian = (
            float(np.asarray(data["lagrangian_budget"]).reshape(-1)[0])
            if "lagrangian_budget" in data
            else 0.2
        )
        fdim = (
            int(np.asarray(data["feature_dim"]).reshape(-1)[0])
            if "feature_dim" in data
            else int(theta.shape[0])
        )
    return {"theta": theta, "lagrangian_budget": lagrangian, "feature_dim": fdim}


class MemorySelectionPolicy:
    """Sequential budget-aware selector with decoupled Gumbel sampling."""

    def __init__(self, dataset: SyntheticDataset, cfg: RLConfig, probes: FrozenProbeHeads | None = None):
        self.dataset = dataset
        self.cfg = cfg
        self.probes = probes
        self.n_units = dataset.unit_embeddings.shape[1]
        self.n_families = len(dataset.family_names)
        self.feature_dim = self.n_families + 8
        self.rng = np.random.default_rng(13)
        arch = getattr(cfg, "policy_architecture", "mlp") or "mlp"
        self.policy_architecture = str(arch).strip().lower()
        self._policy_net = None
        self.theta: np.ndarray | None = None
        if self.policy_architecture == "mlp":
            try:
                import torch  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "policy_architecture='mlp' requires PyTorch. Install with: pip install 'torch>=2.0'"
                ) from exc
            from a_first_memory.rl.neural_policy import SelectionMLP

            self._policy_net = SelectionMLP(
                self.feature_dim,
                hidden_dim=int(getattr(cfg, "policy_hidden_dim", 128)),
                num_hidden_layers=int(getattr(cfg, "policy_num_hidden_layers", 2)),
            )
        elif self.policy_architecture == "linear":
            self.theta = self.rng.normal(0.0, 0.05, size=self.feature_dim)
        else:
            raise ValueError(f"Unknown policy_architecture: {self.policy_architecture!r} (use 'mlp' or 'linear')")
        self.lagrangian_budget = 0.2
        self._family_name_lower = [name.lower() for name in self.dataset.family_names]
        self.semantic_idx = self._idx_by_keyword("semantic", fallback=0)
        self.object_idx = self._idx_by_keyword("object", fallback=min(1, self.n_families - 1))
        self.geometry_idx = self._idx_by_keyword("geometry", fallback=min(2, self.n_families - 1))
        self.low_level_idx = self._idx_by_keyword("low", fallback=min(3, self.n_families - 1))
        self.patch_idx = self._idx_by_keyword("patch", fallback=min(4, self.n_families - 1))
        self.semantic_like_mask = self._mask_by_keywords(("semantic", "scene_graph", "ocr", "face"))
        self.object_like_mask = self._mask_by_keywords(("object", "part"))
        self.geometry_like_mask = self._mask_by_keywords(("geometry", "depth", "normal"))
        self.low_like_mask = self._mask_by_keywords(("low", "texture", "color", "edge", "spatial", "saliency"))
        self.patch_like_mask = self._mask_by_keywords(("patch",))
        self._last_train_diagnostics: dict = {}

    def _softmax(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        temp = max(float(temperature), 1e-6)
        shifted = logits / temp - np.max(logits / temp)
        exp = np.exp(shifted)
        return exp / (np.sum(exp) + 1e-8)

    def _idx_by_keyword(self, keyword: str, fallback: int) -> int:
        for i, name in enumerate(self._family_name_lower):
            if keyword in name:
                return i
        return int(np.clip(fallback, 0, self.n_families - 1))

    def _mask_by_keywords(self, keywords: tuple[str, ...]) -> np.ndarray:
        mask = np.array([any(k in name for k in keywords) for name in self._family_name_lower], dtype=float)
        if float(np.sum(mask)) == 0.0:
            return np.ones(self.n_families, dtype=float) / max(self.n_families, 1)
        return mask / np.sum(mask)

    def _family_utility(self, lag_bucket: int, repeat_count: int) -> tuple[np.ndarray, np.ndarray, float]:
        semantic_weights = np.ones(self.n_families, dtype=float) * 0.75
        perceptual_weights = np.ones(self.n_families, dtype=float) * 0.75
        semantic_weights += 0.55 * self.semantic_like_mask + 0.35 * self.object_like_mask
        semantic_weights += 0.15 * self.patch_like_mask
        perceptual_weights += 0.45 * self.geometry_like_mask + 0.40 * self.low_like_mask
        perceptual_weights += 0.20 * self.patch_like_mask
        semantic_weights[self.semantic_idx] += 0.2
        perceptual_weights[self.low_level_idx] += 0.2
        lag_factor = 1.0 / (1.0 + 0.25 * lag_bucket)
        repeat_boost = 1.0 + 0.10 * repeat_count
        semantic = semantic_weights * repeat_boost * lag_factor
        perceptual = perceptual_weights * (1.05 - 0.08 * repeat_count) * lag_factor
        # Non-monotonic low-level utility peak from semantic overlap and repetition.
        low_level_peak = self.cfg.non_monotonic_low_level_peak
        low_level_gain = np.exp(-((repeat_count - 2.0) ** 2) / max(low_level_peak, 1e-4))
        return semantic, perceptual, float(low_level_gain)

    def _policy_base_logit(self, ctx: np.ndarray) -> float:
        """Scalar score for one candidate context (before family_bonus)."""
        if self._policy_net is not None:
            import torch

            with torch.no_grad():
                x = torch.from_numpy(np.asarray(ctx, dtype=np.float32)).unsqueeze(0)
                return float(self._policy_net(x).squeeze().cpu().numpy())
        assert self.theta is not None
        return float(np.asarray(ctx, dtype=np.float64) @ self.theta)

    def _redundancy_score(self, image_id: int, unit_id: int, selected: np.ndarray) -> float:
        selected_ids = np.where(selected > 0.5)[0]
        if len(selected_ids) == 0:
            return 0.0
        emb = self.dataset.unit_embeddings[image_id, unit_id]
        pool = self.dataset.unit_embeddings[image_id, selected_ids]
        emb_norm = np.linalg.norm(emb) + 1e-8
        pool_norm = np.linalg.norm(pool, axis=1) + 1e-8
        emb_unit = emb / emb_norm
        pool_unit = pool / pool_norm[:, None]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
            cosine = np.abs(pool_unit @ emb_unit)
        cosine = np.clip(np.nan_to_num(cosine, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return float(np.mean(cosine))

    def _unit_context(self, image_id: int, unit_id: int, lag_bucket: int, repeat_count: int, selected: np.ndarray, budget_left: float) -> np.ndarray:
        fam_idx = int(self.dataset.family_index_by_unit[unit_id])
        one_hot = np.zeros(self.n_families, dtype=float)
        one_hot[fam_idx] = 1.0
        emb = self.dataset.unit_embeddings[image_id, unit_id]
        emb_norm = float(np.linalg.norm(emb))
        redundancy = self._redundancy_score(image_id, unit_id, selected)
        novelty = float(self.dataset.novelty_index[image_id])
        schema = float(self.dataset.schema_congruence[image_id])
        selected_family_fraction = float(np.mean(selected[self.dataset.family_index_by_unit == fam_idx])) if np.any(self.dataset.family_index_by_unit == fam_idx) else 0.0
        return np.concatenate(
            [
                one_hot,
                np.array(
                    [
                        self.dataset.costs_by_unit[unit_id],
                        emb_norm,
                        float(lag_bucket),
                        float(repeat_count),
                        budget_left / max(self.cfg.budget, 1e-8),
                        redundancy,
                        novelty,
                        schema - selected_family_fraction,
                    ],
                    dtype=float,
                ),
            ]
        )

    def _sequential_select(self, image_id: int, lag_bucket: int, repeat_count: int, stochastic: bool) -> tuple[np.ndarray, list[np.ndarray], list[float], list[int]]:
        selected = np.zeros(self.n_units, dtype=float)
        budget_left = float(self.cfg.budget)
        chosen_order: list[int] = []
        contexts: list[np.ndarray] = []
        soft_probs: list[float] = []
        sem_util, perc_util, low_level_gain = self._family_utility(lag_bucket, repeat_count)

        while True:
            candidates = [u for u in range(self.n_units) if selected[u] < 0.5 and self.dataset.costs_by_unit[u] <= budget_left]
            if not candidates:
                break
            logits = []
            local_contexts = []
            for u in candidates:
                ctx = self._unit_context(image_id, u, lag_bucket, repeat_count, selected, budget_left)
                local_contexts.append(ctx)
                fam_idx = int(self.dataset.family_index_by_unit[u])
                family_bonus = 0.35 * sem_util[fam_idx] + 0.20 * perc_util[fam_idx]
                if fam_idx == self.low_level_idx:
                    family_bonus += 0.25 * low_level_gain
                logits.append(float(self._policy_base_logit(ctx) + family_bonus))
            logits_arr = np.array(logits, dtype=float)
            if stochastic:
                # Decoupled temperatures: sharp forward argmax, smoother backward proxy.
                gumbel = -np.log(-np.log(np.clip(self.rng.uniform(size=logits_arr.shape[0]), 1e-8, 1 - 1e-8)))
                noisy = logits_arr / self.cfg.forward_temperature + gumbel
                choice_local = int(np.argmax(noisy))
                probs = self._softmax(logits_arr, self.cfg.backward_temperature)
                chosen_soft_prob = float(probs[choice_local])
            else:
                choice_local = int(np.argmax(logits_arr))
                chosen_soft_prob = 1.0

            unit_id = int(candidates[choice_local])
            selected[unit_id] = 1.0
            budget_left -= float(self.dataset.costs_by_unit[unit_id])
            chosen_order.append(unit_id)
            contexts.append(local_contexts[choice_local])
            soft_probs.append(chosen_soft_prob)

            if budget_left <= 1e-8:
                break

        return selected, contexts, soft_probs, chosen_order

    def _simulate_reward(
        self,
        image_id: int,
        selected: np.ndarray,
        lag_bucket: int,
        repeat_count: int,
    ) -> float:
        sem_util, perc_util, low_level_gain = self._family_utility(lag_bucket, repeat_count)
        fam_idx = self.dataset.family_index_by_unit
        selected_units = self.dataset.unit_embeddings[image_id] * selected[:, None]
        fam_energy = np.zeros(self.n_families, dtype=float)
        for i in range(self.n_families):
            fam_energy[i] = float(np.linalg.norm(selected_units[fam_idx == i]))
        fam_energy = fam_energy / (np.sum(fam_energy) + 1e-8)

        if self.probes is not None:
            recog_score, sem_score, perc_score = score_with_frozen_probes(
                self.probes,
                self.dataset,
                image_id,
                selected,
            )
        else:
            recog_score = float(np.dot(fam_energy, 0.5 * sem_util + 0.5 * perc_util))
            sem_score = float(np.dot(fam_energy, sem_util))
            perc_score = float(np.dot(fam_energy, perc_util))
        perc_score = perc_score + low_level_gain * float(fam_energy[self.low_level_idx])
        novelty_family_idxs = sorted(set([self.semantic_idx, self.object_idx, self.patch_idx]))
        novelty_score = float(self.dataset.novelty_index[image_id]) * float(np.sum(fam_energy[novelty_family_idxs]))
        novelty_score += 0.35 * float(self.dataset.novelty_index[image_id]) * float(np.sum(fam_energy * self.semantic_like_mask))
        schema_score = float(self.dataset.schema_congruence[image_id]) * float(
            np.sqrt(max(fam_energy[self.semantic_idx] * fam_energy[self.low_level_idx], 0.0))
        )
        schema_score += 0.30 * float(self.dataset.schema_congruence[image_id]) * float(
            np.sum(fam_energy * (0.6 * self.semantic_like_mask + 0.4 * self.object_like_mask))
        )
        cost_penalty = float(np.sum(self.dataset.costs_by_unit * selected))

        reward = (
            self.cfg.alpha_recog * recog_score
            + self.cfg.beta_sem * sem_score
            + self.cfg.gamma_perc * perc_score
            + self.cfg.delta_novelty * novelty_score
            + self.cfg.eta_schema * schema_score
            - self.cfg.lambda_cost * cost_penalty
            - self.lagrangian_budget * max(0.0, cost_penalty - self.cfg.budget)
        )
        return reward

    def apply_checkpoint(self, checkpoint: str | Path | dict[str, Any]) -> None:
        """Restore trained weights from ``policy_checkpoint.npz`` (linear) or ``policy_checkpoint.pt`` (MLP)."""
        if isinstance(checkpoint, (str, Path)) and str(checkpoint).endswith(".pt"):
            import torch

            if self._policy_net is None:
                raise ValueError("Checkpoint is neural (.pt) but this policy is linear.")
            blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
            fdim = int(blob.get("feature_dim", 0))
            if fdim != self.feature_dim:
                raise ValueError(f"checkpoint feature_dim {fdim} != policy {self.feature_dim}")
            self._policy_net.load_state_dict(blob["state_dict"])
            self.lagrangian_budget = float(blob.get("lagrangian_budget", self.lagrangian_budget))
            return

        ckpt = load_policy_checkpoint(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
        if self._policy_net is not None:
            raise ValueError("Policy is neural; load a .pt checkpoint produced by run_pipeline.")
        theta = np.asarray(ckpt["theta"], dtype=np.float64)
        if theta.shape != (self.feature_dim,):
            raise ValueError(
                f"theta length {theta.shape[0]} does not match policy feature_dim {self.feature_dim}"
            )
        self.theta = np.array(theta, dtype=np.float64, copy=True)
        if "lagrangian_budget" in ckpt:
            self.lagrangian_budget = float(ckpt["lagrangian_budget"])

    def train(self) -> PolicyTrainingResult:
        algo = self.cfg.algorithm.strip().lower()
        if algo == "reinforce":
            return self._train_reinforce()
        if algo == "grpo":
            return self._train_grpo()
        raise ValueError(f"Unknown RL algorithm: {self.cfg.algorithm}")

    def _train_reinforce(self) -> PolicyTrainingResult:
        if self.policy_architecture == "mlp":
            raise NotImplementedError(
                "Neural policy uses GRPO with PyTorch autograd. "
                "Use rl.algorithm='grpo', or policy_architecture='linear' for REINFORCE."
            )
        baseline = 0.0
        reward_history: list[float] = []
        lagrangian_history: list[float] = []

        for _ in range(self.cfg.epochs):
            epoch_rewards = []
            grad = np.zeros_like(self.theta)
            budget_violations = []
            for image_id in self.dataset.image_ids:
                exposure_idx = int(self.rng.integers(0, 3))
                lag_bucket = int(self.dataset.lags[int(image_id), exposure_idx])
                repeat_count = exposure_idx + 1

                selected, contexts, soft_probs, _ = self._sequential_select(
                    image_id=int(image_id),
                    lag_bucket=lag_bucket,
                    repeat_count=repeat_count,
                    stochastic=True,
                )
                reward = self._simulate_reward(int(image_id), selected, lag_bucket, repeat_count)
                epoch_rewards.append(reward)
                advantage = reward - baseline
                total_cost = float(np.sum(self.dataset.costs_by_unit * selected))
                budget_violations.append(max(0.0, total_cost - self.cfg.budget))

                # Straight-through style surrogate gradient using backward probabilities.
                for ctx, p in zip(contexts, soft_probs):
                    grad += (1.0 - p) * ctx * advantage

            mean_reward = float(np.mean(epoch_rewards))
            baseline = 0.9 * baseline + 0.1 * mean_reward
            grad /= max(len(self.dataset.image_ids), 1)
            entropy_term = -self.cfg.entropy_coef * np.sign(self.theta)
            self.theta += self.cfg.learning_rate * (grad + entropy_term)
            mean_violation = float(np.mean(budget_violations))
            self.lagrangian_budget = max(0.0, self.lagrangian_budget + self.cfg.lagrangian_lr * mean_violation)
            reward_history.append(mean_reward)
            lagrangian_history.append(self.lagrangian_budget)

        retention = self.infer_retention_all_images_exposures()
        return PolicyTrainingResult(
            retention_by_image_exposure=retention,
            reward_history=reward_history,
            lagrangian_history=lagrangian_history,
            training_diagnostics={"algorithm": "reinforce"},
        )

    def _train_grpo(self) -> PolicyTrainingResult:
        if self.policy_architecture == "mlp":
            return self._train_grpo_torch()
        assert self.theta is not None
        reward_history: list[float] = []
        lagrangian_history: list[float] = []
        group_size = max(2, int(self.cfg.grpo_group_size))
        adv_clip = max(0.1, float(self.cfg.grpo_adv_clip))
        ratio_eps = max(1e-4, float(self.cfg.grpo_ratio_clip_epsilon))
        kl_coef = max(0.0, float(self.cfg.grpo_kl_coef))
        ref_update_interval = max(1, int(self.cfg.grpo_ref_update_interval))
        grad_clip_norm = max(1e-6, float(self.cfg.grad_clip_norm))
        theta_ref = self.theta.copy()
        clip_fraction_history: list[float] = []
        approx_kl_history: list[float] = []
        grad_norm_history: list[float] = []
        epoch_reward_std_history: list[float] = []
        epoch_adv_std_history: list[float] = []

        for epoch_idx in range(self.cfg.epochs):
            grad = np.zeros_like(self.theta)
            epoch_rewards: list[float] = []
            budget_violations: list[float] = []
            sampled_trajectories = 0
            ratio_values: list[float] = []
            clipped_flags: list[float] = []
            approx_kl_values: list[float] = []
            adv_values: list[float] = []

            for image_id in self.dataset.image_ids:
                exposure_idx = int(self.rng.integers(0, 3))
                lag_bucket = int(self.dataset.lags[int(image_id), exposure_idx])
                repeat_count = exposure_idx + 1

                group_contexts: list[list[np.ndarray]] = []
                group_soft_probs: list[list[float]] = []
                group_rewards: list[float] = []

                for _k in range(group_size):
                    selected, contexts, soft_probs, _ = self._sequential_select(
                        image_id=int(image_id),
                        lag_bucket=lag_bucket,
                        repeat_count=repeat_count,
                        stochastic=True,
                    )
                    reward = self._simulate_reward(int(image_id), selected, lag_bucket, repeat_count)
                    group_contexts.append(contexts)
                    group_soft_probs.append(soft_probs)
                    group_rewards.append(reward)
                    epoch_rewards.append(reward)
                    total_cost = float(np.sum(self.dataset.costs_by_unit * selected))
                    budget_violations.append(max(0.0, total_cost - self.cfg.budget))
                    sampled_trajectories += 1

                rewards = np.array(group_rewards, dtype=float)
                rewards_std = float(np.std(rewards))
                if rewards_std < 1e-8:
                    advantages = np.zeros_like(rewards)
                else:
                    advantages = (rewards - float(np.mean(rewards))) / (rewards_std + 1e-8)
                advantages = np.clip(advantages, -adv_clip, adv_clip)

                for traj_adv, traj_contexts, traj_probs in zip(advantages, group_contexts, group_soft_probs):
                    adv_values.append(float(traj_adv))
                    for ctx, p in zip(traj_contexts, traj_probs):
                        old_logit = float(ctx @ theta_ref)
                        new_logit = float(ctx @ self.theta)
                        log_ratio = np.clip(
                            (new_logit - old_logit) / max(self.cfg.backward_temperature, 1e-6),
                            -20.0,
                            20.0,
                        )
                        ratio = float(np.exp(log_ratio))
                        clipped_ratio = float(np.clip(ratio, 1.0 - ratio_eps, 1.0 + ratio_eps))
                        use_clipped = abs(ratio - clipped_ratio) > 1e-8
                        weight = clipped_ratio if (traj_adv >= 0.0 and use_clipped) else ratio
                        grad += (1.0 - p) * ctx * float(traj_adv) * weight
                        ratio_values.append(ratio)
                        clipped_flags.append(1.0 if use_clipped else 0.0)
                        approx_kl_values.append(max(0.0, ratio - 1.0 - log_ratio))

            if sampled_trajectories == 0:
                continue
            grad /= float(sampled_trajectories)
            entropy_term = -self.cfg.entropy_coef * np.sign(self.theta)
            grad += entropy_term
            # KL trust region anchor to prevent drift from the reference policy.
            grad -= kl_coef * (self.theta - theta_ref)
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > grad_clip_norm:
                grad *= grad_clip_norm / (grad_norm + 1e-8)
            self.theta += self.cfg.learning_rate * grad
            mean_violation = float(np.mean(budget_violations)) if budget_violations else 0.0
            self.lagrangian_budget = max(0.0, self.lagrangian_budget + self.cfg.lagrangian_lr * mean_violation)
            reward_history.append(float(np.mean(epoch_rewards)))
            lagrangian_history.append(self.lagrangian_budget)
            clip_fraction_history.append(float(np.mean(clipped_flags)) if clipped_flags else 0.0)
            approx_kl_history.append(float(np.mean(approx_kl_values)) if approx_kl_values else 0.0)
            grad_norm_history.append(grad_norm)
            epoch_reward_std_history.append(float(np.std(np.array(epoch_rewards, dtype=float))))
            epoch_adv_std_history.append(float(np.std(np.array(adv_values, dtype=float))) if adv_values else 0.0)
            if (epoch_idx + 1) % ref_update_interval == 0:
                theta_ref = self.theta.copy()

        retention = self.infer_retention_all_images_exposures()
        self._last_train_diagnostics = {
            "algorithm": "grpo",
            "clip_fraction_history": clip_fraction_history,
            "approx_kl_history": approx_kl_history,
            "grad_norm_history": grad_norm_history,
            "epoch_reward_std_history": epoch_reward_std_history,
            "epoch_adv_std_history": epoch_adv_std_history,
            "ratio_clip_epsilon": ratio_eps,
            "kl_coef": kl_coef,
            "ref_update_interval": ref_update_interval,
            "grad_clip_norm": grad_clip_norm,
        }
        return PolicyTrainingResult(
            retention_by_image_exposure=retention,
            reward_history=reward_history,
            lagrangian_history=lagrangian_history,
            training_diagnostics=self._last_train_diagnostics,
        )

    def _train_grpo_torch(self) -> PolicyTrainingResult:
        """GRPO with PyTorch autograd over the full MLP policy."""
        import torch
        import torch.nn.utils as nn_utils
        from torch.optim import Adam

        assert self._policy_net is not None
        net = self._policy_net
        net.train()

        from a_first_memory.rl.neural_policy import SelectionMLP

        ref_net = SelectionMLP(
            self.feature_dim,
            hidden_dim=int(getattr(self.cfg, "policy_hidden_dim", 128)),
            num_hidden_layers=int(getattr(self.cfg, "policy_num_hidden_layers", 2)),
        )
        ref_net.load_state_dict(net.state_dict())
        ref_net.eval()
        for p in ref_net.parameters():
            p.requires_grad = False

        opt = Adam(net.parameters(), lr=self.cfg.learning_rate)

        reward_history: list[float] = []
        lagrangian_history: list[float] = []
        group_size = max(2, int(self.cfg.grpo_group_size))
        adv_clip = max(0.1, float(self.cfg.grpo_adv_clip))
        ratio_eps = max(1e-4, float(self.cfg.grpo_ratio_clip_epsilon))
        kl_coef = max(0.0, float(self.cfg.grpo_kl_coef))
        ref_update_interval = max(1, int(self.cfg.grpo_ref_update_interval))
        grad_clip_norm = max(1e-6, float(self.cfg.grad_clip_norm))
        temp = max(float(self.cfg.backward_temperature), 1e-6)

        clip_fraction_history: list[float] = []
        approx_kl_history: list[float] = []
        grad_norm_history: list[float] = []
        epoch_reward_std_history: list[float] = []
        epoch_adv_std_history: list[float] = []
        param_norm_history: list[float] = []

        for epoch_idx in range(self.cfg.epochs):
            epoch_rewards: list[float] = []
            budget_violations: list[float] = []
            sampled_trajectories = 0
            clipped_flags: list[float] = []
            approx_kl_values: list[float] = []
            adv_values: list[float] = []

            opt.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []

            for image_id in self.dataset.image_ids:
                exposure_idx = int(self.rng.integers(0, 3))
                lag_bucket = int(self.dataset.lags[int(image_id), exposure_idx])
                repeat_count = exposure_idx + 1

                group_contexts: list[list[np.ndarray]] = []
                group_soft_probs: list[list[float]] = []
                group_rewards: list[float] = []

                for _k in range(group_size):
                    selected, contexts, soft_probs, _ = self._sequential_select(
                        image_id=int(image_id),
                        lag_bucket=lag_bucket,
                        repeat_count=repeat_count,
                        stochastic=True,
                    )
                    reward = self._simulate_reward(int(image_id), selected, lag_bucket, repeat_count)
                    group_contexts.append(contexts)
                    group_soft_probs.append(soft_probs)
                    group_rewards.append(reward)
                    epoch_rewards.append(reward)
                    total_cost = float(np.sum(self.dataset.costs_by_unit * selected))
                    budget_violations.append(max(0.0, total_cost - self.cfg.budget))
                    sampled_trajectories += 1

                rewards = np.array(group_rewards, dtype=float)
                rewards_std = float(np.std(rewards))
                if rewards_std < 1e-8:
                    advantages = np.zeros_like(rewards)
                else:
                    advantages = (rewards - float(np.mean(rewards))) / (rewards_std + 1e-8)
                advantages = np.clip(advantages, -adv_clip, adv_clip)

                for traj_adv, traj_contexts, traj_probs in zip(advantages, group_contexts, group_soft_probs):
                    adv_values.append(float(traj_adv))
                    traj_adv_t = torch.tensor(float(traj_adv), dtype=torch.float32)
                    for ctx, p in zip(traj_contexts, traj_probs):
                        ctx_t = torch.from_numpy(np.asarray(ctx, dtype=np.float32))
                        new_logit = net(ctx_t).squeeze()
                        with torch.no_grad():
                            old_logit = ref_net(ctx_t).squeeze()
                        log_ratio = torch.clamp((new_logit - old_logit) / temp, -20.0, 20.0)
                        ratio = torch.exp(log_ratio)
                        clipped_ratio = torch.clamp(ratio, 1.0 - ratio_eps, 1.0 + ratio_eps)
                        use_clipped = torch.abs(ratio - clipped_ratio) > 1e-8
                        weight = torch.where(
                            torch.logical_and(traj_adv_t >= 0.0, use_clipped),
                            clipped_ratio,
                            ratio,
                        )
                        p_t = torch.tensor(float(p), dtype=torch.float32)
                        step_loss = -traj_adv_t * weight * (1.0 - p_t) * new_logit
                        losses.append(step_loss)
                        clipped_flags.append(1.0 if bool(use_clipped.item()) else 0.0)
                        approx_kl_values.append(
                            max(
                                0.0,
                                float(ratio.detach().cpu()) - 1.0 - float(log_ratio.detach().cpu()),
                            )
                        )

            if sampled_trajectories == 0 or not losses:
                continue

            loss_total = torch.stack(losses).sum() / float(sampled_trajectories)
            if kl_coef > 0.0:
                kl_loss = torch.tensor(0.0, dtype=torch.float32)
                for p_a, p_b in zip(net.parameters(), ref_net.parameters()):
                    kl_loss = kl_loss + torch.sum((p_a - p_b) ** 2)
                loss_total = loss_total + kl_coef * kl_loss

            loss_total.backward()
            grad_norm = float(nn_utils.clip_grad_norm_(net.parameters(), grad_clip_norm))
            opt.step()

            mean_violation = float(np.mean(budget_violations)) if budget_violations else 0.0
            self.lagrangian_budget = max(0.0, self.lagrangian_budget + self.cfg.lagrangian_lr * mean_violation)
            reward_history.append(float(np.mean(epoch_rewards)))
            lagrangian_history.append(self.lagrangian_budget)
            clip_fraction_history.append(float(np.mean(clipped_flags)) if clipped_flags else 0.0)
            approx_kl_history.append(float(np.mean(approx_kl_values)) if approx_kl_values else 0.0)
            grad_norm_history.append(grad_norm)
            epoch_reward_std_history.append(float(np.std(np.array(epoch_rewards, dtype=float))))
            epoch_adv_std_history.append(
                float(np.std(np.array(adv_values, dtype=float))) if adv_values else 0.0
            )
            with torch.no_grad():
                pn = float(
                    torch.sqrt(sum(torch.sum(w.detach() ** 2) for w in net.parameters())).cpu().item()
                )
            param_norm_history.append(pn)

            if (epoch_idx + 1) % ref_update_interval == 0:
                ref_net.load_state_dict(net.state_dict())

        retention = self.infer_retention_all_images_exposures()
        net.eval()
        self._last_train_diagnostics = {
            "algorithm": "grpo_torch",
            "policy_architecture": "mlp",
            "clip_fraction_history": clip_fraction_history,
            "approx_kl_history": approx_kl_history,
            "grad_norm_history": grad_norm_history,
            "epoch_reward_std_history": epoch_reward_std_history,
            "epoch_adv_std_history": epoch_adv_std_history,
            "param_norm_history": param_norm_history,
            "ratio_clip_epsilon": ratio_eps,
            "kl_coef": kl_coef,
            "ref_update_interval": ref_update_interval,
            "grad_clip_norm": grad_clip_norm,
        }
        return PolicyTrainingResult(
            retention_by_image_exposure=retention,
            reward_history=reward_history,
            lagrangian_history=lagrangian_history,
            training_diagnostics=self._last_train_diagnostics,
        )

    def infer_retention_all_images_exposures(self) -> np.ndarray:
        retention = np.zeros((len(self.dataset.image_ids), 3, self.n_units), dtype=float)
        for image_id in self.dataset.image_ids:
            for exposure_idx in range(3):
                selected, _, _, _ = self._sequential_select(
                    image_id=int(image_id),
                    lag_bucket=int(self.dataset.lags[int(image_id), exposure_idx]),
                    repeat_count=exposure_idx + 1,
                    stochastic=False,
                )
                retention[int(image_id), exposure_idx] = selected
        return retention

    def predict_hit_rates(self, retention_by_image_exposure: np.ndarray) -> np.ndarray:
        preds = np.zeros((len(self.dataset.image_ids), 3), dtype=float)
        for image_id in self.dataset.image_ids:
            for exposure_idx in range(3):
                selected = retention_by_image_exposure[int(image_id), exposure_idx]
                fam_energy = np.zeros(self.n_families, dtype=float)
                units = self.dataset.unit_embeddings[int(image_id)] * selected[:, None]
                for fam_idx in range(self.n_families):
                    fam_energy[fam_idx] = float(np.linalg.norm(units[self.dataset.family_index_by_unit == fam_idx]))
                fam_energy = fam_energy / (np.sum(fam_energy) + 1e-8)
                novelty = float(self.dataset.novelty_index[int(image_id)])
                schema = float(self.dataset.schema_congruence[int(image_id)])
                logits = (
                    1.10 * fam_energy[self.semantic_idx]
                    + 0.85 * fam_energy[self.object_idx]
                    + 0.65 * fam_energy[self.low_level_idx]
                    + 0.55 * fam_energy[self.geometry_idx]
                    + 0.75 * float(np.sum(fam_energy * self.semantic_like_mask))
                    + 0.55 * float(np.sum(fam_energy * self.object_like_mask))
                    + 0.40 * float(np.sum(fam_energy * self.geometry_like_mask))
                    + 0.75 * novelty
                    + 0.70 * schema
                    - 0.10 * float(self.dataset.lags[int(image_id), exposure_idx])
                )
                preds[int(image_id), exposure_idx] = float(1.0 / (1.0 + np.exp(-2.2 * (logits - 0.8))))
        return preds
