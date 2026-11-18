# TruthfulQA prompts

These modules follow the same prompt organization as the BoardgameQA and MMLU
experiments. The payload generator receives the Target question, correct
option, and designated wrong option, then emits one complete
`Signature / Correction / Scope` diagnostic in a single LLM call. The runner
validates that the generated correction supports the supplied answer texts.
