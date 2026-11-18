"""Parse and compose cumulative Signature/Correction/Scope payload variants."""

from __future__ import annotations

import re
from collections.abc import Iterable


COMPONENTS = ("signature", "correction", "scope")
CUMULATIVE_VARIANTS: dict[str, tuple[str, ...]] = {
    "correction": ("correction",),
    "signature_correction": ("signature", "correction"),
    "signature_correction_scope": COMPONENTS,
}


def parse_payload_components(payload: str) -> dict[str, str]:
    """Return the three labeled spans without interpreting their content."""
    text = re.sub(r"\s+", " ", payload).strip().strip("`").strip()
    matches = list(re.finditer(r"\b(Signature|Correction|Scope):", text, re.IGNORECASE))
    names = [match.group(1).lower() for match in matches]
    if names != list(COMPONENTS):
        raise ValueError(
            "Payload must contain Signature, Correction, and Scope exactly once and in that order"
        )
    parts: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parts[names[index]] = text[match.start():end].strip()
    if any(not parts[name][len(name) + 1 :].strip(" .") for name in COMPONENTS):
        raise ValueError("Payload contains an empty Signature, Correction, or Scope")
    return parts


def compose_payload(parts: dict[str, str], include: Iterable[str]) -> str:
    """Compose a diagnostic from selected labeled spans in canonical order."""
    selected = set(include)
    unknown = selected.difference(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown payload components: {sorted(unknown)}")
    body = " ".join(parts[name] for name in COMPONENTS if name in selected).strip()
    if not body:
        raise ValueError("At least one payload component must be retained")
    return f"Separate retained diagnostic. {body}"


def cumulative_payloads(payload: str) -> dict[str, str]:
    """Build Correction, Signature+Correction, and full cumulative variants."""
    parts = parse_payload_components(payload)
    variants = {
        variant: compose_payload(parts, include)
        for variant, include in CUMULATIVE_VARIANTS.items()
    }
    variants["signature_correction_scope"] = payload.strip().strip("`").strip()
    return variants
