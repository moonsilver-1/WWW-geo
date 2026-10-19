# WWW model extension protocol (2026-10-11; before new inference)

## Scope and frozen units
Reuse the 6 development and 12 transfer Web pages, six evidence slots, questions,
offline assets and 64 Boolean configurations. Both support and credit target the
same exact `statement` string, including the original pages. Preserve the fixed
independent-score wording and neutralization controls. Every failure is retained.

## Models and inference
Qwen3-VL-8B-Instruct and InternVL3_5-8B-HF, pinned to downloaded repository commits.
Use bf16, deterministic greedy generation, SDPA, and 200 generated tokens for JSON.
GPU policy update (2026-10-12, user authorized, before new-model inference):
use one A40 per model and allow existing compute tasks when at least 30 GiB
is free in two consecutive 10-second observations. Recheck memory before launch,
record nvitop, UUID and existing process IDs, and set the PyTorch allocator cap
to current free memory minus a 4 GiB reserve. Record peak allocated/reserved
memory. A memory-race defers launch; an OOM increases the next required headroom
by 4 GiB. Resume previously successful paired states. Each project worker holds
a separate GPU lock; the coordinator maintains one worker per project GPU.

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

## Vendor extension fixed on 2026-10-12
Add HuggingFaceTB/SmolVLM2-2.2B-Instruct, Mistral Experimental Pixtral-12B,
and Microsoft Phi-4-Reasoning-Vision-15B. Exact repository revisions, loader
classes and memory thresholds are in model_registry.json. All models use the
same archived pages, evidence slots, literal facts, intervention states and
independent-score instructions. Preserve native image processors and archive
their configuration. Pixtral instructions are prepended to the first user block
to retain them in both scoring and generation templates. SmolVLM2 uses typed
content and its template-leading reference whitespace. Phi uses its documented
role/image tokens, preserves the experiment system instruction, and fixes
non-thinking generation with the <nothink> prefix. Every reference-token span
must pass CPU checks before its model becomes eligible for the GPU queue.
Initial free-memory requirements are 12/32/36 GiB respectively, with a 4 GiB
reserve. The queue reads the registry live and adopts existing project workers
when its coordinator is replaced. Google Gemma is deferred pending repository
access, as selected by the user; this wave covers the three public vendors.


## Output-validity amendment — 2026-10-12 16:05 CST
This rule is fixed after observing Smol's first format failures and before
computing its budget results. It applies uniformly to all extension readers.
Archive every original greedy generation and its strict six-numeric-score parse
status. Record complete support cubes and complete paired cubes separately.
Independent token and source-hash verification covers every support observation.

For the selection task, a format-invalid reported-credit response is an
abstention. Its selection-accuracy contribution is zero on an eligible held-out
state. All fitted predictors use the original common support-query budgets,
training states and complete support cubes. Selection accuracy retains every
page and eligible held-out state. Report format validity as a separate metric.
Reported-credit regret and retention are conditional on valid decisions and
carry their coverage denominator. Exact attribution spectra and edge-field
ratios use complete paired cubes, with explicit page-level coverage counts.
Preserve zero-variance edge fields as recorded cases with undefined normalized
ratios. Archive verification and the budget evaluator must implement these
counts before a reader with format failures enters the manuscript.

Per-page metrics average the defined seed-level metrics; cohort metrics give
each page equal weight. Conditional regret/retention use valid decisions within
each seed. Their defined-run and defined-page counts accompany the separate
eligible/valid decision totals. Pages with no eligible support-removal effects
retain every budget run and record selection metrics as undefined. A constant
support cube retains its runs and records normalized prediction error as
undefined. Complete support archives require all original pages and states;
incomplete support is an explicit integrity failure.
