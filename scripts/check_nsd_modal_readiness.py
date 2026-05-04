from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a_first_memory.data.nsd_layout import load_payload_from_layout
from a_first_memory.data.payload import load_payload_dir, load_payload_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate NSD payload and print Modal run commands.")
    parser.add_argument("--nsd-source", choices=("npz", "dir", "layout"), required=True)
    parser.add_argument("--nsd-path", required=True, help="Local path to NSD npz / payload dir / layout root.")
    parser.add_argument("--feature-npz-path", default="", help="Optional local feature override npz.")
    parser.add_argument("--modal-data-prefix", default="nsd", help="Remote subdir prefix under Modal data volume.")
    parser.add_argument("--output-subdir", default="nsd_full_run_01")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--budget", type=float, default=32.0)
    return parser.parse_args()


def _load_preview(nsd_source: str, nsd_path: str) -> dict:
    if nsd_source == "npz":
        if not nsd_path.lower().endswith(".npz"):
            raise ValueError(f"For --nsd-source npz, expected a .npz file path, got: {nsd_path}")
        return load_payload_npz(nsd_path)
    if nsd_source == "dir":
        return load_payload_dir(nsd_path)
    return load_payload_from_layout(nsd_path)


def main() -> None:
    args = parse_args()
    local_nsd = Path(args.nsd_path)
    if not local_nsd.exists():
        raise FileNotFoundError(f"NSD path not found: {local_nsd}")
    if args.feature_npz_path:
        local_feat = Path(args.feature_npz_path)
        if not local_feat.exists():
            raise FileNotFoundError(f"Feature override path not found: {local_feat}")
    else:
        local_feat = None

    try:
        payload = _load_preview(args.nsd_source, str(local_nsd))
    except Exception as exc:
        raise SystemExit(f"NSD payload validation failed: {exc}") from exc
    n_images = int(payload["unit_embeddings"].shape[0])
    n_units = int(payload["unit_embeddings"].shape[1])
    n_families = int(len(payload["family_names"]))
    n_voxels = int(payload["voxel_responses"].shape[2])
    print("NSD payload validation: OK")
    print(f"  source={args.nsd_source}")
    print(f"  n_images={n_images}, n_units={n_units}, n_families={n_families}, n_voxels={n_voxels}")

    remote_base = args.modal_data_prefix.strip("/") or "nsd"
    remote_nsd = f"{remote_base}/{local_nsd.name}"
    print("\nRecommended upload commands:")
    print(f"  modal volume put a-first-memory-data {local_nsd} {remote_nsd}")
    remote_feat = ""
    if local_feat is not None:
        remote_feat = f"{remote_base}/{local_feat.name}"
        print(f"  modal volume put a-first-memory-data {local_feat} {remote_feat}")

    modal_cmd = (
        "modal run scripts/modal_app.py "
        f"--data-source nsd "
        f"--nsd-source {args.nsd_source} "
        f"--nsd-path {remote_nsd} "
        f"--nsd-strict true "
        f"--epochs {args.epochs} "
        f"--budget {args.budget} "
        f"--output-subdir {args.output_subdir}"
    )
    if remote_feat:
        modal_cmd += f" --nsd-feature-npz-path {remote_feat}"
    print("\nRecommended remote run command:")
    print(f"  {modal_cmd}")


if __name__ == "__main__":
    main()
