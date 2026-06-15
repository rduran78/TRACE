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
run_id: Anthropic_F12_T07_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` â€” Colleague's Claim

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed wasteful (it creates a matrix row-by-row), and could be replaced with a single `matrix()` call. However, this operation runs only **5 times** (once per neighbor source variable). Binding 6.46M small vectors into a matrix is on the order of seconds, not hours. The `lapply` inside `compute_neighbor_stats` is also simple: it indexes into a pre-extracted numeric vector and computes `max/min/mean` on a small handful of neighbors (~4 for rook contiguity). This is lightweight. **This is not the 86-hour bottleneck.**

### `build_neighbor_lookup()` â€” The True Bottleneck

This function runs `lapply` over **6.46 million rows**, and for each row it:

1. **Character coercion + named-vector lookup**: `id_to_ref[as.character(data$id[i])]` â€” named vector lookup via string matching, O(n) in the worst case per call.
2. **Subsetting `id_order` by neighbor indices**: `id_order[neighbors[[ref_idx]]]`.
3. **`paste()` to build keys**: `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” creates temporary character vectors 6.46M times.
4. **Named-vector lookup on `idx_lookup`**: `idx_lookup[neighbor_keys]` â€” this named vector has **6.46 million entries**. Named vector lookup in R is **O(n)** per query (linear scan or partial hashing), performed ~4 times per row (one per neighbor). Over 6.46M rows Ã— ~4 neighbors = ~25.8 million string-match lookups into a 6.46M-length named vector.

**This is the catastrophic bottleneck.** The total cost is roughly O(nÂ²) in character matching. At 6.46M rows, this explains the 86+ hour estimate.

### Summary

| Component | Calls | Per-call cost | Total cost | Bottleneck? |
|---|---|---|---|---|
| `build_neighbor_lookup` | 1 | 6.46M Ã— ~4 named-vector lookups into 6.46M-entry vector | **O(n Ã— k Ã— n) â‰ˆ O(nÂ²)** | **YES â€” dominant** |
| `compute_neighbor_stats` `lapply` | 5 | 6.46M Ã— trivial arithmetic on ~4 values | O(n Ã— k) | No |
| `do.call(rbind, ...)` | 5 | Bind 6.46M vectors | O(n) | Minor |

**Verdict: Reject the colleague's diagnosis.** The true bottleneck is `build_neighbor_lookup()`, specifically the repeated `paste()`-based key construction and named-vector lookups over a 6.46M-entry character vector. Replace it with integer-indexed hash lookups (via `data.table` or environments), and vectorize the entire operation to eliminate the per-row `lapply`.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup`** with a fully vectorized, hash-based approach using `data.table` keyed joins. Instead of pasting keys and doing named-vector lookups, build an integer-keyed lookup table `(id, year) â†’ row_index` and join directly.

2. **Vectorize `compute_neighbor_stats`** using `data.table` grouped operations â€” expand all neighbor pairs into a long table, join values, and compute `max/min/mean` by group in one pass. This eliminates both the `lapply` and the `do.call(rbind, ...)`.

3. **Preserve** the trained Random Forest model (no retraining) and the original numerical estimand (same `max`, `min`, `mean` neighbor statistics, same column names).

Expected speedup: from ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup, compute_neighbor_stats,
# and the outer loop. Produces identical numerical output.
# ==============================================================================

# --- Step 0: Convert to data.table and assign row indices --------------------
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# --- Step 1: Build (id, year) -> row_idx lookup via keyed data.table ---------
# This replaces the paste()-based named vector with an O(1) hash join.
lookup_dt <- cell_dt[, .(id, year, row_idx)]
setkey(lookup_dt, id, year)

# --- Step 2: Build id -> ref_idx mapping (position in id_order) --------------
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# --- Step 3: Expand all (row_idx, neighbor_row_idx) pairs in one shot --------
# For each row in cell_dt, find its neighbors' row indices via a keyed join.
#
# 3a. Map each row's id to its ref_idx in the nb object
cell_dt[, ref_idx := id_to_ref[as.character(id)]]

# 3b. Build a long table of (source_row_idx, neighbor_cell_id, year)
#     by expanding the nb list for each unique cell, then joining on year.

# First, build a data.table of (ref_idx, neighbor_cell_id) from the nb object.
# This is done once and is small: ~1.37M directed relationships.
nb_edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(r) {
  nb <- rook_neighbors_unique[[r]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(data.table(ref_idx = integer(0), neighbor_id = integer(0)))
  }
  data.table(ref_idx = r, neighbor_id = id_order[nb])
}))

# 3c. Join nb_edges to cell_dt to get (source_row_idx, neighbor_id, year)
#     For every row in cell_dt, we know its ref_idx; join to nb_edges.
setkey(nb_edges, ref_idx)
cell_ref <- cell_dt[, .(row_idx, ref_idx, year)]
setkey(cell_ref, ref_idx)

# This join expands each source row by its number of neighbors (~4 for rook).
# Result: ~25.8M rows of (source_row_idx, neighbor_id, year)
edge_expanded <- nb_edges[cell_ref, on = "ref_idx", allow.cartesian = TRUE,
                          nomatch = NA,
                          .(source_row_idx = i.row_idx,
                            neighbor_id    = x.neighbor_id,
                            year           = i.year)]

# Drop rows where neighbor_id is NA (cells with no neighbors)
edge_expanded <- edge_expanded[!is.na(neighbor_id)]

# 3d. Join to lookup_dt to resolve (neighbor_id, year) -> neighbor_row_idx
setnames(edge_expanded, "neighbor_id", "id")
setkey(edge_expanded, id, year)
edge_expanded <- lookup_dt[edge_expanded, on = c("id", "year"),
                           nomatch = NA,
                           .(source_row_idx = i.source_row_idx,
                             neighbor_row_idx = x.row_idx)]

# Drop unresolved neighbors
edge_expanded <- edge_expanded[!is.na(neighbor_row_idx)]

# --- Step 4: Compute neighbor stats for all 5 variables in vectorized fashion -
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Extract the variable values and attach to edges by neighbor_row_idx
  vals <- cell_dt[[var_name]]
  edge_expanded[, nval := vals[neighbor_row_idx]]

  # Compute grouped stats (excluding NAs), keyed by source_row_idx
  stats <- edge_expanded[!is.na(nval),
                         .(nb_max  = max(nval),
                           nb_min  = min(nval),
                           nb_mean = mean(nval)),
                         by = source_row_idx]

  # Initialize columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Assign computed stats back by row index
  cell_dt[stats$source_row_idx, (max_col)  := stats$nb_max]
  cell_dt[stats$source_row_idx, (min_col)  := stats$nb_min]
  cell_dt[stats$source_row_idx, (mean_col) := stats$nb_mean]
}

# Clean up helper columns
cell_dt[, c("row_idx", "ref_idx") := NULL]
edge_expanded[, nval := NULL]

# --- Step 5: Convert back to data.frame for downstream compatibility ---------
cell_data <- as.data.frame(cell_dt)

# ==============================================================================
# The trained Random Forest model is untouched. Use cell_data with the new
# neighbor feature columns for prediction as before:
#   predictions <- predict(rf_model, newdata = cell_data)
# ==============================================================================
```

---

## Why This Works and What Changed

| Original | Optimized | Why faster |
|---|---|---|
| `paste()` key construction per row (6.46M calls) | `data.table` keyed join on integer `(id, year)` | O(1) hash join vs O(n) string scan |
| Named-vector lookup into 6.46M-entry vector | Eliminated entirely | â€” |
| Row-level `lapply` in `build_neighbor_lookup` (6.46M iterations) | Single vectorized `data.table` cartesian join | No R-level loop |
| Row-level `lapply` in `compute_neighbor_stats` (6.46M Ã— 5) | `data.table` grouped `by = source_row_idx` aggregation | C-level grouped ops |
| `do.call(rbind, ...)` on 6.46M vectors (Ã—5) | Direct column assignment | No intermediate list |

**Expected runtime**: ~2â€“10 minutes on a 16 GB laptop (dominated by the ~25.8M-row cartesian join and grouped aggregation, both highly optimized in `data.table`).

**Numerical equivalence**: The same `max`, `min`, and `mean` are computed over the same neighbor sets, producing identical column values. The Random Forest model is never modified.
END_RESPONSE>>>
