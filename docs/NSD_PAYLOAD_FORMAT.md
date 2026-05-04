# NSD Payload Format

To run the pipeline on real NSD-derived data, provide a preprocessed `.npz` file and pass:

- `--data-source nsd`
- `--nsd-npz-path /absolute/path/to/nsd_payload.npz`

You can also pass a directory containing one array file per key:

- `--data-source nsd`
- `--nsd-dir /absolute/path/to/nsd_payload_dir`

Or point to a canonical-ish layout root (auto-discovery for common subpaths):

- `--data-source nsd`
- `--nsd-layout-root /absolute/path/to/nsd_layout_root`

Required keys:

- `unit_embeddings`: shape `[n_images, n_units, embedding_dim]`
- `voxel_responses`: shape `[n_images, 3, n_voxels]`
- `lags`: shape `[n_images, 3]`
- `hit_rates`: shape `[n_images, 3]`
- `family_index_by_unit`: shape `[n_units]`
- `costs_by_unit`: shape `[n_units]`
- `family_names`: shape `[n_families]`, string array

Optional keys:

- `novelty_index`: `[n_images]` (defaults to 0.5)
- `schema_congruence`: `[n_images]` (defaults to 0.5)
- `semantic_targets`: `[n_images, 3]` (defaults from hit rates)
- `perceptual_targets`: `[n_images, 3]` (defaults from hit rates)
- `n_rois`: scalar (defaults to 5)
- `roi_voxels`: scalar (defaults to inferred)
- `subject_ids`: `[n_images]` subject index per image (defaults to all-zero single subject)
- `subject_names`: `[n_subjects]` labels (defaults to `["subj01"]`)

Notes:

- The loader maps this payload into the same dataset contract used by synthetic mode.
- Probe heads are pretrained from these targets and frozen before policy learning.
- If you already have extracted multi-family features, this format is enough to run the full experiment stack.

Builder utility:

- Use `scripts/build_nsd_payload.py` to merge component `.npy/.npz` arrays into a validated payload.
- If your files are already split by key, skip bundle creation and run directly with `--nsd-dir`.
- If your dataset follows common folder names (`features/`, `fmri/`, `behavior/`), use
  `scripts/ingest_nsd_layout.py --layout-root ... --output nsd_payload.npz`.
