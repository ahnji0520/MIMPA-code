"""Prompt templates for TruthfulQA shared-memory false-correction experiments."""

from .collaboration_attack import COLLABORATION_ATTACK_SYSTEM, COLLABORATION_ATTACK_USER
from .debate_update import DEBATE_UPDATE_PEER_SOLUTION, DEBATE_UPDATE_SYSTEM, DEBATE_UPDATE_USER
from .direct_attack import DIRECT_ATTACK_SYSTEM, DIRECT_ATTACK_USER
from .honest_agent import HONEST_AGENT_SYSTEM, HONEST_AGENT_USER
from .memory_context import MEMORY_CONTEXT_SYSTEM, MEMORY_CONTEXT_USER
from .memory_serialization import MEMORY_SERIALIZATION_SYSTEM, MEMORY_SERIALIZATION_USER
from .payload_generation import (
    PAYLOAD_GENERATION_FEW_SHOTS,
    PAYLOAD_GENERATION_SYSTEM,
    PAYLOAD_GENERATION_USER,
)
from .planting_attacker import PLANTING_ATTACKER_SYSTEM, PLANTING_ATTACKER_USER
from .target_attacker import TARGET_ATTACKER_SYSTEM, TARGET_ATTACKER_USER

__all__ = [name for name in globals() if name.isupper()]
