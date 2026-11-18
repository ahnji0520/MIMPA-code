"""LLM-based planting selection without exposing retriever similarity scores."""
from __future__ import annotations

import json
import re
from typing import Any

from prompts.shared.planting_selection import (
    PLANTING_SELECTION_SYSTEM,
    PLANTING_SELECTION_USER,
)


def format_task(task: dict[str, Any]) -> str:
    options = " | ".join(f"{label}) {task['choices'][label]}" for label in task["labels"])
    return f"{task['question']} [{options}]"


def parse_selection(text: str, *, attack_opportunities: int, required_positions: tuple[int, ...]) -> dict[str, Any]:
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        raise ValueError(f"Planting selector did not return JSON: {text[:200]}")
    value = json.loads(match.group(0))
    positions = [int(position) for position in value["planting_positions"]]
    if len(positions) != attack_opportunities:
        raise ValueError(
            f"Expected {attack_opportunities} planting positions, received {positions}"
        )
    if len(set(positions)) != len(positions):
        raise ValueError(f"Planting positions must be distinct: {positions}")
    if any(position not in range(1, 11) for position in positions):
        raise ValueError(f"Planting position out of range: {positions}")
    original_positions = list(positions)
    missing = set(required_positions).difference(positions)
    repaired = False
    if missing:
        # Preserve the nested ablation deterministically: retain every required
        # position, then keep the selector's proposed ranking until K is full.
        positions = list(required_positions)
        positions.extend(position for position in original_positions if position not in positions)
        positions = positions[:attack_opportunities]
        repaired = True
    return {
        "planting_positions": positions,
        "reason": str(value.get("reason", "")),
        "raw": text,
        "attack_opportunities": attack_opportunities,
        "required_positions": list(required_positions),
        "selection_repaired": repaired,
        "original_planting_positions": original_positions,
    }


def select_plantings(
    client: Any,
    *,
    episode: dict[str, Any],
    attack_opportunities: int,
    model: str,
    effort: str,
    purpose: str,
    required_positions: tuple[int, ...] = (),
    max_output_tokens: int = 700,
) -> dict[str, Any]:
    if attack_opportunities not in range(1, 11):
        raise ValueError("attack_opportunities must be between 1 and 10")
    if len(set(required_positions)) != len(required_positions):
        raise ValueError("required_positions must be distinct")
    if len(required_positions) > attack_opportunities:
        raise ValueError("required_positions exceed attack-opportunity budget")
    candidates = "\n".join(
        f"{position}. {format_task(task)}"
        for position, task in enumerate(episode["pre_target_tasks"], 1)
    )
    system = PLANTING_SELECTION_SYSTEM.format(
        attack_opportunities=attack_opportunities
    )
    user = PLANTING_SELECTION_USER.format(
        target=format_task(episode["target"]),
        candidates=candidates,
        attack_opportunities=attack_opportunities,
        required_positions=list(required_positions) if required_positions else "none",
    )
    text, _ = client.call(
        model=model,
        effort=effort,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        purpose=purpose,
        max_output_tokens=max_output_tokens,
        schema_name="planting_selection",
        schema={
            "type": "object",
            "properties": {
                "planting_positions": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 10},
                    "minItems": attack_opportunities,
                    "maxItems": attack_opportunities,
                },
                "reason": {"type": "string"},
            },
            "required": ["planting_positions", "reason"],
            "additionalProperties": False,
        },
    )
    return parse_selection(
        text,
        attack_opportunities=attack_opportunities,
        required_positions=required_positions,
    )


def select_planting(client: Any, *, episode: dict[str, Any], model: str, effort: str, purpose: str) -> dict[str, Any]:
    """Backward-compatible K=1 wrapper used by the original strict runner."""
    selection = select_plantings(
        client,
        episode=episode,
        attack_opportunities=1,
        model=model,
        effort=effort,
        purpose=purpose,
    )
    return {
        "planting_position": selection["planting_positions"][0],
        "reason": selection["reason"],
        "raw": selection["raw"],
    }
