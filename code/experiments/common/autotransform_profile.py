"""Load and validate the frozen targeted AutoTransform profile."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_GENERATION_MODEL = "gpt-3.5-turbo"
EXPECTED_PROFILE_VARIANT = "minimal"
DEFAULT_PROFILE_DIR = (
    ROOT / "data/results/autotransform_targeted/profiles/gpt_3_5_turbo_minimal"
)


def load_autotransform_profile(profile_dir: Path) -> tuple[str, dict[str, Any]]:
    transformed_path = profile_dir / "transformed_profile.txt"
    metadata_path = profile_dir / "metadata.json"
    if not transformed_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Incomplete frozen AutoTransform profile: {profile_dir}")
    transformed = transformed_path.read_text(encoding="utf-8").strip()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(transformed.encode("utf-8")).hexdigest()
    if not transformed or not metadata.get("valid"):
        raise ValueError(f"Invalid frozen AutoTransform profile: {profile_dir}")
    if metadata.get("transformed_profile_sha256") != digest:
        raise ValueError(f"Frozen AutoTransform profile hash mismatch: {profile_dir}")
    if metadata.get("model") != EXPECTED_GENERATION_MODEL:
        raise ValueError(
            "AutoTransform must reuse the frozen gpt-3.5-turbo profile; "
            f"found {metadata.get('model')!r}"
        )
    if metadata.get("profile_variant") != EXPECTED_PROFILE_VARIANT:
        raise ValueError(
            f"Expected frozen profile variant {EXPECTED_PROFILE_VARIANT!r}"
        )
    return transformed, metadata
