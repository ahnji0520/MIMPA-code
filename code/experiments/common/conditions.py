"""Canonical main-table conditions shared by both memory backends."""
CONDITIONS = (
    "clean",
    "mac_attack",
    "autotransform",
    "mimpa_wo_poisoned_record",
    "mimpa_wo_triggering",
    "mimpa",
)
ATTACK_CONDITIONS = CONDITIONS[1:]
MEMORY_POISONING_CONDITIONS = ("mimpa_wo_triggering", "mimpa")
DISPLAY_NAMES = {
    "clean": "Clean (No Attack)",
    "mac_attack": "Multi-Agent Collaboration Attack",
    "autotransform": "AutoTransform",
    "mimpa_wo_poisoned_record": "MIMPA w/o Poisoned Record",
    "mimpa_wo_triggering": "MIMPA w/o Triggering",
    "mimpa": "MIMPA",
}
