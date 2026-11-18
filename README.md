# MIMPA

This repository provides the code for reproducing the main MIMPA experiments.
MIMPA studies memory poisoning in multi-agent systems: a malicious agent plants
a targeted payload during an earlier task, the shared memory retains it, and
benign agents later retrieve and use it when solving a related target task.

## Evaluation settings

The experiments cover two memory architectures:

- **Basic shared memory**, which stores successful trajectories and retrieves the
  top-k records by embedding similarity.
- **G-Memory**, which organizes trajectories and extracted insights using the
  G-Memory framework.

Both architectures are evaluated with GPT-5.4-nano and Gemma-4-31B-IT on
MMLU-Pro, TruthfulQA, and MedMCQA.

The evaluation conditions are:

- **Clean**: all agents are benign.
- **MAC Attack**: a malicious agent directly argues for the target wrong answer.
- **AutoTransform**: a transformed malicious profile attacks through the debate.
- **MIMPA w/o Poisoned Record**: the attacker participates in the target debate,
  but no poisoned record is available in memory.
- **MIMPA w/o Triggering**: the poisoned record is available, but the attacker
  does not trigger it during the target debate.
- **MIMPA**: the poisoned record and the triggering attacker are both present.

The repository also provides the post-hoc auditor used to classify suspicious
behavior in saved planting and target transcripts.

## Repository structure

- `code/experiments/common`: shared API, debate, payload, retrieval, and metrics code
- `code/experiments/main`: Basic-memory runner and experiment matrix
- `code/experiments/gmemory`: G-Memory runner and integration
- `code/experiments/transcript_audit`: post-hoc auditor
- `code/prompts`: prompts used by the agents, attacks, and auditor
- `code/dataset_construction`: dataset and episode-stream builders
- `configs`: experiment settings
- `manifests`: dataset provenance and fixed target IDs
- `third_party/gmemory`: pinned G-Memory commit and QA adaptation patch

## Data preparation

Recreate the benchmark files with the scripts in `code/dataset_construction`,
then construct the fixed seed-42 episode streams. Dataset sources, splits, and
selected target IDs are recorded in `manifests/`.

- MMLU-Pro: test split, 12,032-item pool
- TruthfulQA: 790-question generation set converted to binary choice
- MedMCQA: validation split, 4,183-item pool

Benchmark text, model weights, and generated transcripts are not distributed
with this repository.

## Installation

Install the common dependencies:

```bash
pip install -r requirements.txt
```

For Gemma inference with vLLM, also install:

```bash
pip install -r requirements-vllm.txt
```

Set the OpenAI credential in the process environment when running GPT:

```bash
export OPENAI_API_KEY='...'
```

## Running the main experiments

Preview the experiment matrix without making model calls:

```bash
python code/experiments/main/run_matrix.py \
  --model gpt-5.4-nano --effort low --backend openai --sample-size 100
```

Run the prepared GPT or Gemma matrix:

```bash
scripts/run_gpt.sh
scripts/run_gemma.sh
```

The Gemma launcher expects an OpenAI-compatible vLLM endpoint. The reported
configuration uses temperature 0, seed 42, disabled thinking, BF16, and a
2,048-token output limit.

## Running the auditor

After the GPT main results have been generated, run:

```bash
scripts/run_auditor.sh
```

The auditor reads the saved transcripts and does not rerun the main agents.

## G-Memory setup

Clone G-Memory into `third_party/framework/GMemory`, check out the commit in
`third_party/gmemory/UPSTREAM_COMMIT`, and apply the QA adaptation patch:

```bash
git apply /path/to/paper_repro/third_party/gmemory/qa_adaptation.patch
```

Install the embedding model separately. Its location can be supplied through
the `GMEMORY_EMBED_MODEL_PATH` environment variable.
