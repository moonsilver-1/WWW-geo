# WWW model extension protocol (2026-10-11; before new inference)

## Scope and frozen units
Reuse the 6 development and 12 transfer Web pages, six evidence slots, questions,
offline assets and 64 Boolean configurations. Both support and credit target the
same exact `statement` string, including the original pages. Preserve the fixed
independent-score wording and neutralization controls. Every failure is retained.

## Models and inference
Qwen3-VL-8B-Instruct and InternVL3_5-8B-HF, pinned to downloaded repository commits.
Use bf16, deterministic greedy generation, SDPA, and 200 generated tokens for JSON.
Use one otherwise idle A40 per model, with physical GPU UUID and command recorded.
The queue requires three observations with no compute processes, <=512 MiB used
and <=3% utilization before launch. It records nvitop immediately before launch.

## Measurements and scoring checks
Teacher-force the fixed statement; validate that prefix input IDs match exactly.
Store both mean log-probability over the whole assistant-response token span
(legacy interface, including terminator) and the exact reference-token span.
Save token IDs/counts to permit independent checks and score-span sensitivity.
Credit requires exactly six numeric JSON scores in [0,1], without clipping.
Free-answer generation is recorded at the all-present state, separately from
candidate-conditioned attribution. Results do not select the prompt.

## Analysis
Primary strata: reader and development/transfer pages. Compute Walsh interaction
shares, held-state prediction, consistency/calibration diagnostics, and selection
metrics with identical definitions. A cube requires all 64 successfully parsed
paired states. Repeated attempts remain archived; select the last successful
paired record per state and report attempts/exclusions. Budget-matched baselines
use 16/32/48 support queries, 20 seeds, and page-level paired inference from the
previously fixed score_lift_protocol_20261008.md. Publish query costs by channel.

## Reporting
New checkpoints remain separate from the earlier 3B/2B reader results. No results
are asserted until the observed output, input hashes, spans, failures and held-out
splits pass verification. The manuscript records the actual completed experiment.
