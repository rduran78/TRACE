# TRACE 20260611 Scoring Methodology

This document defines the scoring process for the TRACE 20260611 prompt-ablation experiment. It is the root-level methodology note for the scoring prompts saved in `Score_Input/` and the scoring manifest saved as `score_input_manifest.csv`.

All scoring prompts are stored directly in `Score_Input/` as a flat single-batch directory. There are no provider or temperature subdirectories. Each filename encodes the provider/model, prompt family, Copilot temperature-setting label, and replicate:

```text
S__OpenAI_GPT50moderate__F01__temp_00__rep_001.md
S__Anthropic_Opus46__F12__temp_10__rep_030.md
```

Raw scoring outputs are preserved in `Score_Output/` exactly as exported by Copilot Studio. Copilot saved JSON content with `.md` extensions. Validated JSON copies are stored in `Score_Output_Normalized/` with `.json` extensions and are the source for `Database/scored_outputs.csv`.

## Purpose

The scoring process evaluates whether each model response discovered the target computational reformulation:

```text
Repeated row-wise cell-year spatial-neighbor lookup
```

converted into:

```text
Reusable static neighbor topology + dynamic yearly cell attributes
```

or equivalently:

```text
cells as nodes, rook-neighbor relationships as edges, yearly variables as node attributes
```

The desired solution computes exact neighbor max, min, and mean statistics for each cell-year without repeatedly rebuilding cell-year string keys or repeatedly discovering the same spatial topology.

## Input And Output Scope

Scoring prompts are generated only for raw experiment outputs that exist under `Experiment_Output/` with filenames matching the experiment manifest. Missing prompt-manifest rows are documented in `missing_experiment_outputs.csv`.

The current TRACE output structure reflects the actual Copilot Studio runs:

- Anthropic outputs are expected for Copilot UI-selected settings 0.0, 0.3, 0.7, and 1.0 using Opus 4.6.
- OpenAI outputs are expected for Copilot UI-selected settings 0.0, 0.3, 0.7, and 1.0 using GPT 5.0 moderate.

## Temperature-Setting Interpretation

Temperature values in this experiment are treated as Copilot Studio setting labels, not independently verified runtime parameters.

For Anthropic, Copilot Studio exposed manual temperature controls. The labels 0.0, 0.3, 0.7, and 1.0 mean that the operator selected those UI settings before running the corresponding batches. These labels are recorded as `copilot_ui_selected_unverified`.

For OpenAI, the rerun used GPT 5.0 moderate and Copilot Studio exposed temperature-setting controls. The labels 0.0, 0.3, 0.7, and 1.0 mean that the operator selected those UI settings before running the corresponding batches. These labels are recorded as `copilot_ui_selected_unverified`.

The scoring model must not use temperature metadata to adjust scores. Temperature-setting interpretation belongs to later analysis, not to response-level scoring.

## Required JSON Fields

Each scoring output must be one valid minified JSON object with these fields:

```text
experiment_id
run_id
provider
model_label
copilot_temperature_setting
temperature_setting_status
prompt_family_id
prompt_family_slug
family_label
family_group
replicate
file_status
bottleneck_identification
topology_invariance
solution_architecture
yearly_attribute_application
numerical_equivalence
raster_handling
rf_handling
implementation_quality
resists_false_framing
mechanism_score
discovery_success
publication_grade_success
response_class
rationale_25_words
```

## File Status

- `valid_response`: substantive answer.
- `non_answer`: refusal, says insufficient information, or does not attempt the task.
- `empty_file`: no substantive content or whitespace only.
- `api_error`: API, tool, safety, or execution status text rather than a substantive answer.
- `truncated`: visibly cut off.

## Component Scores

### bottleneck_identification

- 0: no bottleneck identified or wrong bottleneck, such as Random Forest inference alone.
- 1: vague neighbor, row-wise, or repeated-work issue.
- 2: specific row-wise neighbor lookup, cell-year string-key, or list construction bottleneck.

### topology_invariance

- 0: absent.
- 1: implies reuse or caching but does not clearly identify static topology.
- 2: explicitly separates static neighbor topology from dynamic yearly attributes.

### solution_architecture

- 0: generic advice or no usable architecture.
- 1: partial speedup such as preallocation, parallelization, Rcpp, chunking, or local caching.
- 2: reusable adjacency table, edge list, sparse graph, spatial weights matrix, or fixed neighbor index.

Full credit requires a reusable representation of neighbor topology. Generic performance advice does not qualify by itself.

### yearly_attribute_application

- 0: absent.
- 1: ambiguous about how yearly values are applied.
- 2: computes values per year or per variable using fixed topology.

### numerical_equivalence

- 0: approximation or method change.
- 1: says results should be preserved but gives limited detail.
- 2: preserves the same rook-neighbor definition, same-year statistics, NA handling, and max/min/mean estimands.

## Diagnostic Scores

### raster_handling

- 0: proposes unsafe raster focal operations when irregular topology is stated or changes the neighbor definition.
- 1: mentions raster but leaves safety unresolved, or raster is irrelevant/not mentioned.
- 2: handles raster safely or rejects raster focal operations when unsafe.

### rf_handling

- 0: retrains or changes the Random Forest model, or treats Random Forest inference as the main bottleneck without evidence.
- 1: gives secondary Random Forest advice while preserving the trained model.
- 2: preserves the trained Random Forest and centers feature construction as the bottleneck.

### implementation_quality

- 0: no code or invalid code.
- 1: partial pseudocode or incomplete R.
- 2: plausible R, data.table, sparse-matrix, or equivalent implementation.

### resists_false_framing

Use `null` unless the prompt family is F11 or F12.

For F11 and F12:

- 0: accepts the false diagnosis.
- 1: uncertain or partially challenges it.
- 2: clearly rejects the false diagnosis using code evidence.

## Derived Outcomes

`mechanism_score` is the sum of:

```text
bottleneck_identification
topology_invariance
solution_architecture
yearly_attribute_application
numerical_equivalence
```

`discovery_success` equals 1 only when:

```text
mechanism_score >= 9
solution_architecture == 2
topology_invariance == 2
numerical_equivalence >= 1
rf_handling >= 1
```

`publication_grade_success` equals 1 when:

```text
mechanism_score >= 8
solution_architecture == 2
topology_invariance == 2
numerical_equivalence >= 1
```

## Response Classes

The scoring model must choose one:

```text
wrong_rf_optimization
generic_performance_advice
partial_neighbor_optimization
preallocation_or_parallel_only
raster_solution_wrong
raster_to_adjacency_transfer
adjacency_table_success
sparse_graph_success
full_correct_solution
non_answer
empty_file
api_error
truncated
other
```

## Conservative Scoring Rules

Do not give discovery credit for merely saying "cache neighbor_lookup" if the response still builds cell-year string lookups row by row for every year.

Do not give `solution_architecture = 2` for generic parallelization, preallocation, Rcpp, chunking, or do.call/rbind fixes unless the response also separates reusable topology from dynamic yearly attributes.

Do not give raster success credit for raster focal operations if the response ignores irregular topology or changes the neighbor definition.

Give `numerical_equivalence = 2` only if the response preserves same-year neighbor statistics, original rook-neighbor relationships, NA handling, and max/min/mean.

F01 responses may still earn success if they infer the hidden upstream neighbor-feature construction from the visible downstream code, but scoring should not assume hidden context is available.

## Output Discipline

The scoring model must return only one minified JSON object. It must not wrap the JSON in markdown. It must not add prose outside JSON. If the raw response is blank, refused, or an API status message, the scorer must still return valid JSON.

## Validation And Database Construction

All 2,880 scoring outputs were audited for file presence, JSON parsing, and required schema fields. No scoring rerun was required. The validation audit is stored in:

```text
score_output_validation_audit.csv
```

The database builder recomputed:

```text
mechanism_score
discovery_success
publication_grade_success
```

from component scores rather than relying blindly on scorer-provided derived fields. Rows where recomputed derived fields differed from scorer-provided values are flagged in `Database/scored_outputs.csv` under `validation_notes`.
