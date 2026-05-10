"""
Create PEFT-compatible copies of LoRA adapter directories.

Example:
    python -m src.training.make_peft_adapter_compatible --adapters checkpoints/lora_fr/final checkpoints/lora_ca/final checkpoints/occitan_3b2/final
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Iterable, Set

from peft import LoraConfig


def _supported_lora_keys() -> Set[str]:
    """Return constructor keys accepted by local peft.LoraConfig."""
    keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
    keys.discard("self")
    return keys


def _metadata_keys() -> Set[str]:
    """Return metadata keys commonly expected around LoraConfig."""
    return {
        "peft_type",
        "auto_mapping",
        "base_model_name_or_path",
        "revision",
        "task_type",
        "inference_mode",
    }


def sanitize_adapter(src: Path, suffix: str = "_compat", overwrite: bool = True) -> Path:
    """Copy adapter directory and sanitize adapter_config.json in the copy."""
    if not src.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {src}")
    if not src.is_dir():
        raise ValueError(f"Adapter path is not a directory: {src}")

    cfg_src = src / "adapter_config.json"
    if not cfg_src.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in: {src}")

    dst = src.parent / f"{src.name}{suffix}"
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    cfg_dst = dst / "adapter_config.json"
    with open(cfg_dst, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    keep = _metadata_keys() | _supported_lora_keys()
    cleaned = {k: v for k, v in cfg.items() if k in keep}
    dropped = sorted(set(cfg.keys()) - set(cleaned.keys()))

    with open(cfg_dst, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=True)
        f.write("\n")

    print(f"\n[{src}] -> [{dst}]")
    print(f"  kept   ({len(cleaned)}): {sorted(cleaned.keys())}")
    print(f"  dropped({len(dropped)}): {dropped}")
    return dst


def sanitize_many(adapters: Iterable[str], suffix: str, overwrite: bool) -> None:
    for path_str in adapters:
        sanitize_adapter(Path(path_str), suffix=suffix, overwrite=overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PEFT-compatible adapter copies by sanitizing adapter_config.json.",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        required=True,
        help="Paths to adapter directories (e.g. checkpoints/lora_fr/final).",
    )
    parser.add_argument(
        "--suffix",
        default="_compat",
        help="Suffix appended to adapter dir name for the compatibility copy.",
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Do not overwrite an existing destination copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sanitize_many(
        adapters=args.adapters,
        suffix=args.suffix,
        overwrite=not args.no_overwrite,
    )
    print("\nDone. Use the generated *_compat paths in --adapters.")


if __name__ == "__main__":
    main()
