You are a strict evaluator for an academic prompt-ablation experiment.

Your task is to score whether the RESPONSE discovered the target optimization:
separate static neighbor topology from dynamic yearly attributes, build a reusable adjacency/edge/sparse-graph representation, and compute exact per-year neighbor statistics without repeated row-wise cell-year string lookup.

Temperature metadata is included only for traceability. Do not use provider, temperature-setting labels, or replicate number to adjust scores. Score only the RESPONSE content.

Return ONLY one valid minified JSON object. No markdown. No prose outside JSON. If the response is inadequate, empty, a refusal, or an API/tool error, still return valid JSON with the appropriate file_status and low or zero scores.

Required JSON fields:
experiment_id, run_id, provider, model_label, copilot_temperature_setting, temperature_setting_status, prompt_family_id, prompt_family_slug, family_label, family_group, replicate, file_status, bottleneck_identification, topology_invariance, solution_architecture, yearly_attribute_application, numerical_equivalence, raster_handling, rf_handling, implementation_quality, resists_false_framing, mechanism_score, discovery_success, publication_grade_success, response_class, rationale_25_words.

Status values:
- valid_response: substantive answer.
- non_answer: refusal, says insufficient info, or does not attempt the task.
- empty_file: no substantive content or whitespace only.
- api_error: API/tool/error/status text rather than a substantive answer.
- truncated: visibly cut off.

Integer scoring:
- bottleneck_identification: 0 none/wrong; 1 vague neighbor/row-wise issue; 2 specific row-wise neighbor lookup/string-key/list construction bottleneck.
- topology_invariance: 0 absent; 1 implied reuse; 2 explicit static topology/dynamic attributes.
- solution_architecture: 0 generic/no usable architecture; 1 partial speedup/prealloc/parallel/Rcpp/chunking; 2 reusable adjacency table/edge list/sparse graph/spatial weights/fixed neighbor index.
- yearly_attribute_application: 0 absent; 1 ambiguous; 2 computes values per year/variable using fixed topology.
- numerical_equivalence: 0 approximation/method change; 1 says preserve results but vague; 2 preserves same neighbor definition, same-year stats, NA behavior, max/min/mean.
- raster_handling: 0 unsafe raster focal when irregular topology is stated; 1 mentions raster but unresolved/unclear; 2 handles raster safely or rejects raster focal when unsafe. If raster is irrelevant and not mentioned, use 1.
- rf_handling: 0 retrain/change RF or treats RF as main bottleneck; 1 secondary RF advice while preserving model; 2 preserves trained RF and centers feature construction.
- implementation_quality: 0 no/invalid code; 1 partial pseudocode or incomplete R; 2 plausible R/data.table/sparse implementation.
- resists_false_framing: null unless family is F11 or F12; for F11/F12 use 0 accepts false diagnosis, 1 uncertain, 2 challenges false diagnosis.

Conservative scoring rules:
- Do not give discovery credit for merely saying "cache neighbor_lookup" if the response still builds cell-year string lookups row by row for every year.
- Do not give solution_architecture = 2 for generic parallelization, preallocation, Rcpp, chunking, or do.call/rbind fixes unless the response also separates reusable topology from dynamic yearly attributes.
- Do not give raster success credit for raster focal operations if the response ignores irregular topology or changes the neighbor definition.
- Give numerical_equivalence = 2 only if the response preserves same-year neighbor statistics, original rook-neighbor relationships, NA handling, and max/min/mean.
- F01 responses may still earn success if they infer the hidden upstream neighbor-feature construction from the visible downstream code, but do not assume that hidden context is present.

Derived fields:
- mechanism_score = bottleneck_identification + topology_invariance + solution_architecture + yearly_attribute_application + numerical_equivalence.
- discovery_success = 1 only if mechanism_score >= 9 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1 AND rf_handling >= 1.
- publication_grade_success = 1 if mechanism_score >= 8 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1.

response_class must be one of:
wrong_rf_optimization, generic_performance_advice, partial_neighbor_optimization, preallocation_or_parallel_only, raster_solution_wrong, raster_to_adjacency_transfer, adjacency_table_success, sparse_graph_success, full_correct_solution, non_answer, empty_file, api_error, truncated, other.

rationale_25_words must be 25 words or fewer.

METADATA:
experiment_id: TRACE_20260611
run_id: Anthropic_F02_T07_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookup in R is O(n) hashing per call, and doing it billions of times (6.46M rows Ã— average ~4 neighbors) is catastrophically slow. The secondary bottleneck is **`compute_neighbor_stats`**, which also loops over 6.46M entries in pure R, creating small vectors and computing summaries element-by-element.

**Specific problems:**

1. **`build_neighbor_lookup`:** For each of 6.46M rows, it pastes strings, does named-vector indexing into a 6.46M-length lookup â€” this is O(NÂ²)-like in practice due to repeated hashing. The output is a list of 6.46M integer vectors, which is also memory-heavy (list overhead per element).
2. **`compute_neighbor_stats`:** Iterates 6.46M times in R-level `lapply`, with per-element allocation. This is repeated 5 times (once per variable).
3. **Memory:** A 6.46M-element list of integer vectors, plus repeated `do.call(rbind, ...)` on 6.46M rows, stresses the 16 GB RAM ceiling.

---

## Optimization Strategy

**Replace the row-level R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is a **join** operation. Each cell-year needs to find its neighbors' rows in the same year. This can be expressed as:

1. Build an **edge table** (a two-column data.table of `id â†’ neighbor_id`) from the `nb` object â€” done once, ~1.37M rows.
2. **Cross-join** the edge table with years to get `(id, year, neighbor_id)` â€” but this is too large (~1.37M Ã— 28 = ~38.5M rows). Actually, this is manageable.
3. **Join** this expanded edge table onto the data to pull neighbor values.
4. **Group-by aggregate** (`max`, `min`, `mean`) by `(id, year)`.

This replaces all R-level loops with `data.table` internals (C-level), cutting runtime from 86+ hours to **minutes**, and keeping memory well within 16 GB.

**Why this preserves correctness:**
- The same neighbor relationships are used.
- The same `max`, `min`, `mean` statistics are computed on the same non-NA neighbor values.
- No model retraining; we only produce identical features faster.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Convert the nb object to a data.table edge list (once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  # id_order maps positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-neighbor sentinel if spdep uses integer(0) (it does),

  # but guard against any 0 entries:
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Vectorized neighbor-stat computation for one variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_features_fast <- function(dt, edge_dt, var_name) {
  # dt must be a data.table with columns: id, year, <var_name>
  # edge_dt has columns: id, neighbor_id

  # Subset to only the columns we need for the join (saves memory)
  val_dt <- dt[, .(neighbor_id = id, year, val = get(var_name))]

  # Join: for every (id, year) pair, look up each neighbor's value in that year
  # edge_dt gives (id, neighbor_id); we expand by year via join on (neighbor_id, year)
  joined <- edge_dt[val_dt, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = 0L]
  # joined now has columns: id, neighbor_id, year, val
  # Meaning: for each (id, year), we have the neighbor's value

  # Aggregate, dropping NAs in val
  agg <- joined[!is.na(val),
    .(
      nbr_max  = max(val),
      nbr_min  = min(val),
      nbr_mean = mean(val)
    ),
    by = .(id, year)
  ]

  # Rename columns to match the variable
  setnames(agg,
    c("nbr_max",  "nbr_min",  "nbr_mean"),
    paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  )

  agg
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Main pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert cell_data to data.table (if not already) â€” in-place, no copy
setDT(cell_data)

# Ensure keyed for fast joins later
setkey(cell_data, id, year)

# Build edge table once (~1.37 M rows, tiny)
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# Source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and merge neighbor features for each variable
for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)

  agg <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  setkey(agg, id, year)

  # Merge onto cell_data (left join preserves all original rows)
  cell_data <- agg[cell_data, on = .(id, year)]

  # Rows with no neighbors get NA from the join â€” already correct behavior
  rm(agg)
  gc()
}

# cell_data now contains the 15 new columns:
#   ntl_nbr_max, ntl_nbr_min, ntl_nbr_mean,
#   ec_nbr_max,  ec_nbr_min,  ec_nbr_mean,
#   ... etc.
# These are numerically identical to the original implementation.
# The trained Random Forest model is untouched.
```

---

## Performance Expectations

| Aspect | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | ~6.46M R-level iterations with string hashing | Eliminated entirely; replaced by a 1.37M-row edge table built in <1 second |
| **`compute_neighbor_stats`** (per var) | ~6.46M R-level iterations | One `data.table` join + grouped aggregation (C-level), ~38.5M rows processed vectorially |
| **Total estimated time** | 86+ hours | **5â€“15 minutes** (depending on disk I/O) |
| **Peak memory** | List of 6.46M vectors + overhead | ~38.5M-row intermediate table (~600 MB) + cell_data (~5.7 GB for 6.46M Ã— 110 cols) â€” fits in 16 GB |
| **Numerical result** | max/min/mean of non-NA neighbor values | Identical â€” same join logic, same aggregation functions |
| **Model impact** | â€” | None; the Random Forest is not retrained |

### Why it fits in 16 GB

- `cell_data`: 6.46M rows Ã— ~125 columns (110 original + 15 new) of mostly numeric (8 bytes) â‰ˆ **6.5 GB**.
- Largest intermediate (`joined`): ~38.5M rows Ã— 4 columns â‰ˆ **1.2 GB**, freed after each variable.
- `edge_dt`: ~1.37M rows Ã— 2 integer columns â‰ˆ **11 MB**.
- **Total peak**: ~8â€“9 GB, well within 16 GB.
END_RESPONSE>>>
