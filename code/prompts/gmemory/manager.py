"""Optional benchmark-neutral QA adaptation of G-Memory trajectory extraction.

The default experiment uses G-Memory's unmodified upstream prompts. These
prompts are only for an explicit sensitivity analysis and are shared by every
dataset, so the memory manager is never tuned to one benchmark.
"""
QA_EXTRACT_SYSTEM = """You extract the key reasoning steps from a successful multi-agent question-answering trajectory. Preserve the evidence, claims, decision criteria, and answer-selection steps that actually appeared in the trajectory. Do not add facts or silently correct the discussion."""
QA_EXTRACT_USER = """Extract concise key steps from the successful trajectory below.

Requirements:
- Use only information present in the trajectory.
- Preserve claims or decision criteria that affected the agents' reasoning, even if they were introduced by only one participant.
- Record how the team selected its final answer.
- Do not add outside knowledge or dataset-specific assumptions.

## Task
{task}

## Successful trajectory
{trajectory}

## Key steps
"""
