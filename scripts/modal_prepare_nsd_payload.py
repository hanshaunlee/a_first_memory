from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("a-first-memory-prepare-nsd-payload-v2")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "pandas>=2.2", "h5py>=3.11")
    .add_local_dir(".", remote_path="/root/project")
)
data_volume = modal.Volume.from_name("a-first-memory-data", create_if_missing=True)


def _normalize_01(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo + 1e-8)


def _trial_to_session_and_index(trial_id: int) -> tuple[int, int]:
    # NSD trial ids are 1-based over 40 sessions x 750 trials.
    trial_zero = int(trial_id) - 1
    if trial_zero < 0:
        raise ValueError(f"Invalid trial id: {trial_id}")
    session = (trial_zero // 750) + 1
    trial_idx = trial_zero % 750
    return session, trial_idx


def _select_voxel_indices(example_volume_flat, voxel_count: int):
    import numpy as np

    candidate = np.flatnonzero(np.asarray(example_volume_flat) != 0)
    if candidate.size == 0:
        candidate = np.arange(np.asarray(example_volume_flat).size)
    count = min(int(voxel_count), int(candidate.size))
    picked = np.linspace(0, candidate.size - 1, num=count, dtype=int)
    return candidate[picked]


def _extract_image_feature_blocks(img_brick, nsd_ids):
    import numpy as np

    n_images = len(nsd_ids)
    grid = 4
    units_per_family = grid * grid
    emb_dim = 8
    color_stats = np.zeros((n_images, units_per_family, emb_dim), dtype=np.float32)
    opponent_stats = np.zeros((n_images, units_per_family, emb_dim), dtype=np.float32)
    freq_stats = np.zeros((n_images, units_per_family, emb_dim), dtype=np.float32)

    for i, nsd_id in enumerate(nsd_ids):
        img = np.asarray(img_brick[int(nsd_id)], dtype=np.float32) / 255.0
        h, w, _ = img.shape
        ys = np.linspace(0, h, grid + 1, dtype=int)
        xs = np.linspace(0, w, grid + 1, dtype=int)
        unit_idx = 0
        for gy in range(grid):
            for gx in range(grid):
                patch = img[ys[gy] : ys[gy + 1], xs[gx] : xs[gx + 1], :]
                if patch.size == 0:
                    continue
                r = patch[..., 0]
                g = patch[..., 1]
                b = patch[..., 2]
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                sat = np.max(patch, axis=-1) - np.min(patch, axis=-1)
                gx_grad = np.diff(lum, axis=1, prepend=lum[:, :1])
                gy_grad = np.diff(lum, axis=0, prepend=lum[:1, :])
                grad_mag = np.sqrt(gx_grad * gx_grad + gy_grad * gy_grad)

                patch_vec = np.array(
                    [
                        float(np.mean(r)),
                        float(np.mean(g)),
                        float(np.mean(b)),
                        float(np.std(r)),
                        float(np.std(g)),
                        float(np.std(b)),
                        float(np.mean(lum)),
                        float(np.mean(grad_mag)),
                    ],
                    dtype=np.float32,
                )
                color_stats[i, unit_idx] = patch_vec

                contrast = float(np.percentile(lum, 95.0) - np.percentile(lum, 5.0))
                opponent_vec = np.array(
                    [
                        float(np.mean(r - g)),
                        float(np.mean(g - b)),
                        float(np.mean(b - r)),
                        float(np.std(lum)),
                        float(np.mean(sat)),
                        float(np.std(sat)),
                        contrast,
                        float(np.std(grad_mag)),
                    ],
                    dtype=np.float32,
                )
                opponent_stats[i, unit_idx] = opponent_vec

                fft = np.abs(np.fft.rfft2(lum.astype(np.float32)))
                fh, fw = fft.shape
                l1 = fft[: max(1, fh // 8), : max(1, fw // 8)]
                l2 = fft[: max(1, fh // 4), : max(1, fw // 4)]
                low_e = float(np.mean(l1))
                mid_e = float(np.mean(l2) - low_e)
                high_e = float(np.mean(fft) - float(np.mean(l2)))
                p = fft / (float(np.sum(fft)) + 1e-8)
                entropy = float(-np.sum(p * np.log(p + 1e-12)))
                freq_vec = np.array(
                    [
                        low_e,
                        mid_e,
                        high_e,
                        entropy,
                        float(np.max(fft)),
                        float(np.mean(fft)),
                        float(np.std(fft)),
                        float(np.mean(lum * lum)),
                    ],
                    dtype=np.float32,
                )
                freq_stats[i, unit_idx] = freq_vec
                unit_idx += 1

    return {
        "color_patch": color_stats,
        "opponent_patch": opponent_stats,
        "frequency_patch": freq_stats,
    }


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=60 * 60 * 6,
)
def build_nsd_payload_remote(
    raw_nsd_path: str = "nsd",
    output_npz_path: str = "nsd/nsd_payload_subj01.npz",
    subject: str = "subj01",
    max_images: int = 240,
    voxel_count: int = 512,
) -> dict:
    import sys

    import h5py
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "/root/project/src")
    from a_first_memory.data.payload import save_payload
    from a_first_memory.features.adapters import FamilyFeatureBlock, build_feature_bank

    subject = subject.strip().lower()
    if not subject.startswith("subj"):
        raise ValueError(f"Subject must be in form 'subj01'; got {subject}")
    subject_num = int(subject.replace("subj", ""))

    root = Path("/data") / raw_nsd_path
    if not root.exists():
        raise FileNotFoundError(f"Raw NSD root not found in Modal volume: {root}")

    stim_info_csv = root / "nsddata/experiments/nsd/nsd_stim_info_merged.csv"
    stim_h5 = root / "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    betas_root = root / "nsddata_betas/ppdata" / subject / "func1pt8mm/betas_fithrf"
    if not stim_info_csv.exists():
        raise FileNotFoundError(f"Missing stim info CSV: {stim_info_csv}")
    if not stim_h5.exists():
        raise FileNotFoundError(f"Missing stimuli HDF5: {stim_h5}")
    if not betas_root.exists():
        raise FileNotFoundError(f"Missing betas directory: {betas_root}")

    rep_cols = [f"subject{subject_num}_rep0", f"subject{subject_num}_rep1", f"subject{subject_num}_rep2"]
    df = pd.read_csv(stim_info_csv)
    missing_rep_cols = [c for c in rep_cols if c not in df.columns]
    if missing_rep_cols:
        raise KeyError(f"Missing repetition columns for {subject}: {missing_rep_cols}")

    rows = df[(df[rep_cols[0]] > 0) & (df[rep_cols[1]] > 0) & (df[rep_cols[2]] > 0)].copy()
    rows = rows.sort_values(rep_cols[0])
    if max_images > 0:
        rows = rows.head(int(max_images))
    if rows.empty:
        raise ValueError(f"No 3-repeat rows found for {subject} in stim info CSV.")

    nsd_ids = rows["nsdId"].to_numpy(dtype=int)
    trial_matrix = rows[rep_cols].to_numpy(dtype=int)
    n_images = int(nsd_ids.shape[0])

    session_ids = sorted({_trial_to_session_and_index(int(t))[0] for t in trial_matrix.reshape(-1)})
    beta_files = {sid: betas_root / f"betas_session{sid:02d}.hdf5" for sid in session_ids}
    missing_beta_files = [str(p) for p in beta_files.values() if not p.exists()]
    if missing_beta_files:
        raise FileNotFoundError(f"Missing beta session files: {missing_beta_files}")

    handles = {sid: h5py.File(path, "r") for sid, path in beta_files.items()}
    try:
        first_trial = int(trial_matrix[0, 0])
        first_session, first_idx = _trial_to_session_and_index(first_trial)
        first_volume = np.asarray(handles[first_session]["betas"][first_idx], dtype=np.float32).reshape(-1)
        voxel_idx = _select_voxel_indices(first_volume, voxel_count=int(voxel_count))
        n_voxels = int(voxel_idx.shape[0])

        voxel_responses = np.zeros((n_images, 3, n_voxels), dtype=np.float32)
        for i in range(n_images):
            for e in range(3):
                session, trial_idx = _trial_to_session_and_index(int(trial_matrix[i, e]))
                vol = np.asarray(handles[session]["betas"][trial_idx], dtype=np.float32).reshape(-1)
                voxel_responses[i, e] = vol[voxel_idx]
    finally:
        for handle in handles.values():
            handle.close()

    flat = voxel_responses.reshape(-1, voxel_responses.shape[-1])
    flat_mean = np.mean(flat, axis=0, keepdims=True)
    flat_std = np.std(flat, axis=0, keepdims=True)
    flat_std[flat_std < 1e-6] = 1.0
    flat = (flat - flat_mean) / flat_std
    flat = np.clip(flat, -8.0, 8.0)
    voxel_responses = flat.reshape(voxel_responses.shape).astype(np.float32)

    with h5py.File(stim_h5, "r") as stim_file:
        img_brick = stim_file["imgBrick"]
        blocks = _extract_image_feature_blocks(img_brick, nsd_ids)

    feature_bank = build_feature_bank(
        [
            FamilyFeatureBlock(name="color_patch", embeddings=blocks["color_patch"], unit_cost=1.0),
            FamilyFeatureBlock(name="opponent_patch", embeddings=blocks["opponent_patch"], unit_cost=1.2),
            FamilyFeatureBlock(name="frequency_patch", embeddings=blocks["frequency_patch"], unit_cost=1.5),
        ]
    )

    gaps = np.zeros_like(trial_matrix, dtype=int)
    gaps[:, 1] = np.maximum(0, trial_matrix[:, 1] - trial_matrix[:, 0])
    gaps[:, 2] = np.maximum(0, trial_matrix[:, 2] - trial_matrix[:, 1])
    lag_bins = np.array([1, 32, 256, 1024, 4096, 16384], dtype=int)
    lags = np.digitize(gaps, lag_bins, right=False).astype(int)
    lags[:, 0] = 0

    loss = rows["loss"].to_numpy(dtype=float)
    loss_norm = _normalize_01(loss)
    signal = np.mean(np.abs(voxel_responses), axis=2)
    signal_norm = _normalize_01(signal)
    hit_rates = np.clip(0.7 * signal_norm + 0.3 * loss_norm[:, None], 0.0, 1.0).astype(np.float32)

    shared = rows["shared1000"].astype(float).to_numpy()
    bold5000 = rows["BOLD5000"].astype(float).to_numpy()
    coco_split_train = rows["cocoSplit"].astype(str).str.contains("train").astype(float).to_numpy()
    novelty_index = _normalize_01(loss_norm + 0.25 * _normalize_01(np.mean(gaps[:, 1:], axis=1))).astype(np.float32)
    schema_raw = 0.5 * shared + 0.2 * bold5000 + 0.3 * coco_split_train + 0.1 * (1.0 - loss_norm)
    schema_congruence = _normalize_01(schema_raw).astype(np.float32)
    semantic_targets = np.clip(0.55 * hit_rates + 0.45 * novelty_index[:, None], 0.0, 1.0).astype(np.float32)
    perceptual_targets = np.clip(0.65 * hit_rates + 0.35 * schema_congruence[:, None], 0.0, 1.0).astype(np.float32)

    subject_ids = np.zeros(n_images, dtype=int)
    subject_names = np.array([subject], dtype=object)
    n_rois = 4
    roi_voxels = max(1, int(n_voxels // n_rois))

    payload = {
        "unit_embeddings": feature_bank["unit_embeddings"].astype(np.float32),
        "family_index_by_unit": feature_bank["family_index_by_unit"].astype(int),
        "costs_by_unit": feature_bank["costs_by_unit"].astype(np.float32),
        "family_names": feature_bank["family_names"],
        "voxel_responses": voxel_responses.astype(np.float32),
        "lags": lags.astype(int),
        "hit_rates": hit_rates.astype(np.float32),
        "novelty_index": novelty_index,
        "schema_congruence": schema_congruence,
        "semantic_targets": semantic_targets,
        "perceptual_targets": perceptual_targets,
        "subject_ids": subject_ids,
        "subject_names": subject_names,
        "n_rois": np.array(n_rois),
        "roi_voxels": np.array(roi_voxels),
    }

    out_path = Path("/data") / output_npz_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload(str(out_path), payload)
    data_volume.commit()
    return {
        "output_npz_path": output_npz_path,
        "n_images": n_images,
        "n_units": int(payload["unit_embeddings"].shape[1]),
        "embedding_dim": int(payload["unit_embeddings"].shape[2]),
        "n_voxels": int(payload["voxel_responses"].shape[2]),
        "subject": subject,
        "sessions_used": session_ids,
    }


@app.local_entrypoint()
def main(
    raw_nsd_path: str = "nsd",
    output_npz_path: str = "nsd/nsd_payload_subj01.npz",
    subject: str = "subj01",
    max_images: int = 240,
    voxel_count: int = 512,
) -> None:
    result = build_nsd_payload_remote.remote(
        raw_nsd_path=raw_nsd_path,
        output_npz_path=output_npz_path,
        subject=subject,
        max_images=max_images,
        voxel_count=voxel_count,
    )
    print(json.dumps(result, indent=2))
