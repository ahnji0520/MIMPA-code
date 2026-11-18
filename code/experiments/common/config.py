"""Shared experiment defaults."""

from pathlib import Path


MAMAS_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = MAMAS_ROOT / "code"
DATA_ROOT = MAMAS_ROOT / "data"
ENV_PATH = CODE_ROOT / ".env"

MODEL = "gpt-5.4-nano"
REASONING_EFFORT = "low"
BACKEND = "openai"
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
LOCAL_MODELS = {
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "gemma4-31b": "google/gemma-4-31B-it",
}
DATASETS = ("mmlu", "truthfulqa", "medmcqa", "mmlu_pro", "scalr")
