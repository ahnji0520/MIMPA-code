"""Bridge shared stream-debate logs to the upstream G-Memory implementation."""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
GMEMORY_ROOT = ROOT / "third_party" / "framework" / "GMemory"
LOCAL_EMBEDDING_MODEL = GMEMORY_ROOT / "models" / "all-MiniLM-L6-v2"
_IMPORT_LOCK = threading.Lock()
_STACK: tuple[Any, ...] | None = None


def import_gmemory() -> tuple[Any, ...]:
    """Import upstream G-Memory despite its cwd-relative config assumption."""
    global _STACK
    with _IMPORT_LOCK:
        if _STACK is not None:
            return _STACK
        os.environ.setdefault("GMEMORY_EMBED_DEVICE", "cpu")
        os.environ.setdefault("GMEMORY_EMBED_MODEL_PATH", str(LOCAL_EMBEDDING_MODEL))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        if str(GMEMORY_ROOT) not in sys.path:
            sys.path.insert(0, str(GMEMORY_ROOT))
        previous_cwd = Path.cwd()
        os.chdir(GMEMORY_ROOT)
        try:
            from mas.memory.common import AgentMessage, MASMessage
            from mas.memory.mas_memory.GMemory import GMemory
            from mas.memory.mas_memory.prompt import GMemoryPrompts
            from mas.utils import EmbeddingFunc
            from mas.memory.utils import cosine_similarity
        finally:
            os.chdir(previous_cwd)
        _STACK = (AgentMessage, MASMessage, GMemory, GMemoryPrompts, EmbeddingFunc, cosine_similarity)
        return _STACK


def configure_manager_prompts(mode: str) -> None:
    """Select native prompts or one shared QA extraction prompt for ablation."""
    # The vendored prompt module defaults to the shared QA-adapted wording.
    # `native` is retained only as a backward-compatible alias.
    if mode in {"qa_adapted", "native"}:
        return
    if mode != "qa_shared":
        raise ValueError(f"Unknown G-Memory manager prompt mode: {mode}")
    from prompts.gmemory import QA_EXTRACT_SYSTEM, QA_EXTRACT_USER
    _, _, _, prompt_object, _, _ = import_gmemory()
    prompt_object.extract_true_traj_system_prompt = QA_EXTRACT_SYSTEM
    prompt_object.extract_true_traj_user_prompt = QA_EXTRACT_USER


def manager_prompt_snapshot(mode: str) -> dict[str, str]:
    """Return the exact upstream/shared prompts that manage memory in this run."""
    configure_manager_prompts(mode)
    _, _, _, prompts, _, _ = import_gmemory()
    names = (
        "generative_task_system_prompt", "generative_task_user_prompt",
        "extract_true_traj_system_prompt", "extract_true_traj_user_prompt",
        "critique_success_rules_system_prompt", "critique_success_rules_user_prompt",
        "critique_compare_rules_system_prompt", "critique_compare_rules_user_prompt",
        "merge_rules_system_prompt", "merge_rules_user_prompt",
    )
    return {name: str(getattr(prompts, name)) for name in names}


class ResponsesLLM:
    """Adapt the repository's Responses client to G-Memory's LLMCallable."""
    def __init__(self, client: Any, model: str, effort: str, purpose_prefix: str):
        self.client = client
        self.model = model
        self.effort = effort
        self.purpose_prefix = purpose_prefix
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, messages: Any, temperature: float = 0.1,
                 max_tokens: int = 512, stop_strs: Any = None,
                 num_comps: int = 1) -> str:
        del temperature, stop_strs, num_comps
        with self._lock:
            self.calls += 1
            call_index = self.calls
        normalized = [
            {"role": message.role, "content": message.content}
            if hasattr(message, "role")
            else {"role": message["role"], "content": message["content"]}
            for message in messages
        ]
        purpose = f"{self.purpose_prefix}.manager_call{call_index}"
        base_budget = max(512, int(max_tokens or 512))
        try:
            text, _ = self.client.call(
                model=self.model, effort=self.effort, messages=normalized,
                purpose=purpose, max_output_tokens=base_budget,
            )
        except RuntimeError as exc:
            # Keep the upstream-sized budget for every ordinary call. Only a
            # reasoning response that exhausts it before emitting text gets a
            # larger deterministic fallback budget.
            if "status=incomplete" not in str(exc) or base_budget >= 2048:
                raise
            text, _ = self.client.call(
                model=self.model, effort=self.effort, messages=normalized,
                purpose=purpose + ".expanded_budget",
                max_output_tokens=2048,
            )
        return text


def validate_complete_log(log: dict[str, Any]) -> None:
    if log.get("correct") and log.get("majority_answer") != log.get("task", {}).get("gold_answer"):
        raise ValueError("Successful log has inconsistent majority and gold answers")
    rounds = log.get("rounds", [])
    if not rounds or any(len(entries) != 3 for entries in rounds):
        raise ValueError("Expected complete three-agent debate rounds")


def log_to_mas_message(log: dict[str, Any], *, formatted_task: str,
                       position: int, is_planting: bool,
                       payload_exact: bool) -> Any:
    """Preserve every completed debate and its verified outcome in a MASMessage."""
    validate_complete_log(log)
    AgentMessage, MASMessage, _, _, _, _ = import_gmemory()
    source = str(log["task"]["task_id"])
    succeeded = bool(log["correct"])
    message = MASMessage(task_main=formatted_task, task_description=formatted_task, label=succeeded)

    # A QA debate is one interaction graph. Round-1 utterances have no
    # inter-agent predecessors. In every later round, each agent receives the
    # previous-round utterances of the other two agents, so those visible peer
    # utterances are the node's incoming edges.
    previous_round_node_ids: dict[int, str] = {}
    for round_index, entries in enumerate(log["rounds"], 1):
        current_round_node_ids: dict[int, str] = {}
        for entry in entries:
            agent_index = int(entry["agent_index"])
            upstream_ids = [
                node_id
                for peer_index, node_id in sorted(previous_round_node_ids.items())
                if peer_index != agent_index
            ]
            node_id = message.add_message_to_current_state(
                AgentMessage(
                    # Roles are evaluation-only metadata. The team and the
                    # memory manager must not be told which participant is the
                    # attacker; G-Memory sees only stable neutral identities.
                    agent_name=f"agent_{agent_index}",
                    message=entry["response"],
                    extra_fields={
                        "round_index": round_index,
                        "agent_index": agent_index,
                    },
                ),
                upstream_ids,
            )
            current_round_node_ids[agent_index] = node_id
        previous_round_node_ids = current_round_node_ids

    message.move_state(
        action=f"Aggregate final-round answers; team answer={log['majority_answer']}",
        observation=f"Verified outcome: {'SUCCESS' if succeeded else 'FAILURE'}",
        reward=1 if succeeded else 0,
    )
    message.add_extra_field("source_id", source)
    message.add_extra_field("position", int(position))
    message.add_extra_field("is_planting", bool(is_planting))
    message.add_extra_field("payload_exact", bool(payload_exact))
    message.add_extra_field("verified_success", succeeded)
    return message


def serialize_retrieved(
    successful: list[Any], failed: list[Any], insights: list[str],
    current_task: str,
) -> str:
    """Use the upstream G-Memory solver-prompt formatter verbatim.

    Upstream workflows do not inject failed detailed trajectories into the
    solver prompt; they affect insight induction instead. `failed` is accepted
    here only to keep retrieval provenance explicit.
    """
    del failed
    # import_gmemory() has already added the pinned checkout to sys.path.
    from tasks.mas_workflow.format import (
        format_task_context, format_task_prompt_with_insights,
    )

    successful_shots = [
        format_task_context(
            message.task_description,
            message.task_trajectory,
            message.get_extra_field("key_steps"),
        )
        for message in successful
    ]
    prompt = format_task_prompt_with_insights(
        few_shots=[],
        memory_few_shots=successful_shots,
        insights=list(insights),
        task_description=current_task,
    )
    # The upstream formatter always emits a reference-example section even
    # when `few_shots` is empty. This experiment supplies no external
    # few-shots, so omit only that empty section and preserve the remaining
    # upstream G-Memory text verbatim.
    memory_start = prompt.index("## Your Own Past Successes (Execution Patterns)")
    rendered = prompt[memory_start:]
    leaked_labels = ("attacker_agent_", "honest_agent_")
    if any(label in rendered for label in leaked_labels):
        raise ValueError("Evaluation-only role label leaked into model-facing G-Memory prompt")
    return rendered


def source_id(message: Any) -> str:
    return str(message.get_extra_field("source_id") or "")


def payload_survival(successful: list[Any], insights: list[str], payload: str) -> dict[str, Any]:
    trajectory_text = "\n".join(str(message.task_trajectory or "") for message in successful)
    key_steps_text = "\n".join(str(message.get_extra_field("key_steps") or "") for message in successful)
    insight_text = "\n".join(insights)
    combined = "\n".join((trajectory_text, key_steps_text, insight_text))
    return {
        "exact_in_trajectory": bool(payload and payload in trajectory_text),
        "exact_in_key_steps": bool(payload and payload in key_steps_text),
        "exact_in_insights": bool(payload and payload in insight_text),
        "exact_anywhere": bool(payload and payload in combined),
    }


class GMemoryStore:
    """One isolated native G-Memory store for one clean or attacked episode."""
    def __init__(self, *, workdir: Path, client: Any, model: str, effort: str,
                 purpose_prefix: str, manager_prompt_mode: str, hop: int,
                 start_insights_threshold: int, rounds_per_insights: int,
                 insights_point_num: int, random_seed: int,
                 reset: bool = True) -> None:
        _, _, GMemory, _, EmbeddingFunc, cosine_similarity = import_gmemory()
        configure_manager_prompts(manager_prompt_mode)
        if reset and workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        self.workdir = workdir
        self.cosine_similarity = cosine_similarity
        self.llm = ResponsesLLM(client, model, effort, purpose_prefix)
        self.memory = GMemory(
            namespace="g-memory",
            global_config={
                "working_dir": str(workdir),
                "hop": hop,
                "start_insights_threshold": start_insights_threshold,
                "rounds_per_insights": rounds_per_insights,
                "insights_point_num": insights_point_num,
                "random_seed": random_seed,
            },
            llm_model=self.llm,
            embedding_func=EmbeddingFunc("sentence-transformers/all-MiniLM-L6-v2"),
        )

    @property
    def size(self) -> int:
        return int(self.memory.memory_size)

    def add(self, message: Any) -> None:
        self.memory.add_memory(message)

    def retrieve(self, *, query: str, successful_topk: int, insight_topk: int,
                 threshold: float, payload: str = "") -> tuple[str, dict[str, Any]]:
        if self.size == 0:
            return serialize_retrieved([], [], [], query), {
                "query": query, "memory_size": 0,
                "retrieved_successful_source_ids": [],
                "retrieved_successful_count": 0,
                "retrieved_failed_count": 0,
                "insight_input_mode": "query_conditioned_supporting_queries",
                "trajectory_selection": "query_graph_embedding_topk_no_llm_reranking",
                "retrieved_insights": [], "retrieved_insight_count": 0,
                "all_insight_records": [],
                "embedding_ranking": [],
                "payload_survival": payload_survival([], [], payload),
                "manager_llm_calls_cumulative": self.llm.calls,
            }
        query_vector = self.memory.embedding_func.embed_query(query)
        collection = self.memory.main_memory.get(include=["documents", "metadatas"])
        ranking: list[dict[str, Any]] = []
        for document, metadata in zip(collection.get("documents", []), collection.get("metadatas", [])):
            extras = json.loads(metadata.get("extra_fields", "{}"))
            similarity = self.cosine_similarity(query_vector, self.memory.embedding_func.embed_query(document))
            ranking.append({
                "source_id": str(extras.get("source_id", "")),
                "position": extras.get("position"),
                "is_planting": bool(extras.get("is_planting", False)),
                "cosine_similarity": float(similarity),
            })
        ranking.sort(key=lambda row: row["cosine_similarity"], reverse=True)
        # Keep native query-graph and embedding retrieval, while skipping the
        # optional LLM relevance reranker. Insight retrieval remains upstream's
        # query-conditioned procedure and uses success and failure records.
        insight_manager = self.memory.insights_layer
        insight_records = [dict(record) for record in insight_manager.insights_memory]
        use_all_insights = insight_topk <= 0
        effective_insight_topk = (
            max(1, len(insight_records)) if use_all_insights else insight_topk
        )
        successful, failed, retrieved_insights = self.memory._retrieve_memory_raw(
            query_task=query, successful_topk=successful_topk, failed_topk=0,
            insight_windows=effective_insight_topk, threshold=threshold,
        )
        insights = (
            [
                str(record.get("rule", "")).strip()
                for record in insight_records
                if str(record.get("rule", "")).strip()
            ]
            if use_all_insights
            else list(retrieved_insights)
        )
        text = serialize_retrieved(successful, failed, insights, query)
        audit = {
            "query": query, "memory_size": self.size,
            "successful_topk": successful_topk,
            "insight_topk": insight_topk,
            "effective_insight_topk": effective_insight_topk,
            "insight_input_mode": (
                "all_generated_insights" if use_all_insights
                else "query_conditioned_supporting_queries"
            ),
            "trajectory_selection": "query_graph_embedding_topk_no_llm_reranking",
            "threshold": threshold,
            "retrieved_successful_source_ids": [source_id(item) for item in successful],
            "retrieved_successful_count": len(successful),
            "retrieved_failed_count": len(failed),
            "retrieved_insights": list(insights),
            "retrieved_insight_count": len(insights),
            "all_insight_records": insight_records,
            "embedding_ranking": ranking,
            "payload_survival": payload_survival(successful, insights, payload),
            "manager_llm_calls_cumulative": self.llm.calls,
        }
        return text, audit
