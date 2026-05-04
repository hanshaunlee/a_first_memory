from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_root_module():
    root_script = Path(__file__).resolve().parents[2] / "scripts" / "verify_grpo.py"
    spec = importlib.util.spec_from_file_location("_verify_grpo_root", root_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load verifier script from {root_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_verifier() -> dict:
    return _load_root_module().run_verifier()


def main() -> None:
    _load_root_module().main()


if __name__ == "__main__":
    main()

