# NSD Getting Started (Acquisition to Modal)

This checklist makes sure you actually have the NSD assets required to run this project.

## 1) Access requirements

Before downloading, complete NSD access requirements:

- agree to Terms and Conditions and submit NSD Data Access form via CVNlab guidance: [How to get the data](https://cvnlab.slite.page/p/dC~rBTjqjb/How-to-get-the-data)
- official NSD site: [naturalscenesdataset.org](https://naturalscenesdataset.org/)

## 2) Download tooling

This repo now supports AWS CLI access via:

- `python -m awscli ...`

If needed, install:

- `python -m pip install awscli`

## 3) Download NSD data from AWS

Public bucket:

- `s3://natural-scenes-dataset/`

List it:

- `python -m awscli s3 ls --no-sign-request s3://natural-scenes-dataset/`

Use the repository helper (dry-run first by default):

- `bash scripts/download_nsd_from_aws.sh --dest "/absolute/path/nsd" --mode minimal`

Run actual download:

- `bash scripts/download_nsd_from_aws.sh --dest "/absolute/path/nsd" --mode minimal --run`

By default, minimal mode pulls only `subj01` 1.8mm HDF5 session betas.
You can set subjects explicitly:

- `bash scripts/download_nsd_from_aws.sh --dest "/absolute/path/nsd" --mode minimal --subjects "subj01,subj02" --run`

Or request all subjects (large):

- `bash scripts/download_nsd_from_aws.sh --dest "/absolute/path/nsd" --mode minimal --all-subjects --run`

For full betas (very large):

- `bash scripts/download_nsd_from_aws.sh --dest "/absolute/path/nsd" --mode full --run`

## 4) Build project payload

This project runs from an NSD payload (`.npz` or split-key directory), not raw bucket folders directly.

If your local tree matches the expected layout, ingest it:

- `python scripts/ingest_nsd_layout.py --layout-root "/absolute/path/nsd" --output "/absolute/path/nsd_payload.npz"`

Or merge arrays manually:

- `python scripts/build_nsd_payload.py ... --output "/absolute/path/nsd_payload.npz"`

If you only have raw NSD folders and want conversion to run remotely (recommended for heavier preprocessing), use Modal:

- `modal run scripts/modal_prepare_nsd_payload.py --raw-nsd-path nsd --output-npz-path nsd/nsd_payload_subj01.npz --subject subj01 --max-images 240 --voxel-count 512`

This writes the payload directly into the Modal data volume at `/data/nsd/nsd_payload_subj01.npz`.

## 5) Validate readiness + get Modal command

- `python scripts/check_nsd_modal_readiness.py --nsd-source npz --nsd-path "/absolute/path/nsd_payload.npz" --output-subdir nsd_grpo_run_01 --epochs 50 --budget 32`

If payload was created in the Modal data volume via `modal_prepare_nsd_payload.py`, use:

- `modal run scripts/modal_app.py --data-source nsd --nsd-source npz --nsd-path nsd/nsd_payload_subj01.npz --epochs 50 --budget 32 --output-subdir nsd_grpo_run_01`

This prints:

- upload command(s) for Modal volume
- exact `modal run scripts/modal_app.py ...` command

## 6) Run remotely on Modal (recommended path)

Example:

- `modal run scripts/modal_app.py --data-source nsd --nsd-source npz --nsd-path nsd/nsd_payload.npz --epochs 50 --budget 32 --output-subdir nsd_grpo_run_01`

## 7) Optional programmatic helper

NSD Python accessor:

- [tknapen/nsd_access](https://github.com/tknapen/nsd_access)

It can help inspect NSD contents, but this repo still requires payload conversion for training/evaluation.
