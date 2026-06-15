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
run_id: Anthropic_F02_T03_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries â€” an O(nÂ²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again per variable, subsetting numeric vectors by index lists. With 5 variables, that's ~32.3 million list iterations total.

**Specific problems:**

1. **`build_neighbor_lookup`:** `paste()` and named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely slow. Each call to `idx_lookup[neighbor_keys]` does a linear scan or hash probe on a 6.46M-element named character vector. The resulting list of 6.46M integer vectors also consumes substantial memory.
2. **`compute_neighbor_stats`:** `lapply` over 6.46M elements with per-element subsetting, NA removal, and summary computation is slow in interpreted R. The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is also a known performance anti-pattern.
3. **Memory:** The neighbor lookup list (6.46M elements, each a small integer vector) plus the data frame with 110+ columns at 6.46M rows pushes close to or beyond 16 GB, especially with intermediate copies.

---

## Optimization Strategy

### Principle: Replace per-row R loops with vectorized joins and grouped aggregations using `data.table`.

1. **Replace `build_neighbor_lookup` entirely.** Instead of building a 6.46M-element list, build a flat `data.table` edge list of `(row_i, neighbor_row_j)` pairs. This is constructed via a vectorized merge â€” no per-row `lapply` needed.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation.** Join the edge table to the variable values, then compute `max`, `min`, `mean` per group in one vectorized pass.

3. **Memory management:** The flat edge table will have ~1.37M neighbor pairs Ã— 28 years â‰ˆ ~38.5M rows of two integer columns (~308 MB), which is far more cache-friendly and memory-predictable than 6.46M ragged lists. We process one variable at a time and free intermediates.

4. **The trained Random Forest model and all numerical outputs are preserved** â€” we only change how features are computed, not what is computed.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a flat edge table (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(cell_dt, id_order, neighbors) {
  # Build a data.table of directed neighbor pairs at the cell level:
  #   from_id -> to_id  (spatial neighbors)
  # Then cross-join with years to get row-level edges.

  # --- 1a. Expand the nb object into a flat cell-level edge list ---
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  # Remove any 0-entries (spdep uses 0 to denote "no neighbors")
  valid <- to_ref > 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  cell_edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # --- 1b. Map (id, year) -> row index in cell_dt ---
  cell_dt[, row_idx := .I]
  id_year_map <- cell_dt[, .(id, year, row_idx)]

  # --- 1c. Expand cell edges across all years via merge ---
  # First merge: get the "from" row index (the row that will receive the feature)
  edge_rows <- merge(
    cell_edges,
    id_year_map,
    by.x = "from_id", by.y = "id",
    allow.cartesian = TRUE,
    sort = FALSE
  )
  setnames(edge_rows, c("row_idx", "year"), c("from_row", "year"))

  # Second merge: get the "to" row index (the neighbor whose value we read)
  edge_rows <- merge(
    edge_rows,
    id_year_map[, .(id, year, row_idx)],
    by.x = c("to_id", "year"), by.y = c("id", "year"),
    sort = FALSE
  )
  setnames(edge_rows, "row_idx", "to_row")

  # Keep only the two row-index columns (minimal memory)
  edge_rows <- edge_rows[, .(from_row, to_row)]
  setkey(edge_rows, from_row)

  # Clean up helper column
  cell_dt[, row_idx := NULL]

  return(edge_rows)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Compute neighbor stats for one variable (replaces
#          compute_neighbor_stats + compute_and_add_neighbor_features)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_table) {
  # Attach neighbor values via the edge table
  vals <- cell_dt[[var_name]]
  agg  <- edge_table[, .(neighbor_val = vals[to_row]), by = from_row]

  # Remove NAs before aggregation
  agg <- agg[!is.na(neighbor_val)]

  # Grouped aggregation â€” single vectorized pass
  stats <- agg[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = from_row]

  # Initialize result columns with NA
  n <- nrow(cell_dt)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  set(cell_dt, j = col_max,  value = rep(NA_real_, n))
  set(cell_dt, j = col_min,  value = rep(NA_real_, n))
  set(cell_dt, j = col_mean, value = rep(NA_real_, n))

  # Fill in computed values at the correct rows
  rows <- stats$from_row
  set(cell_dt, i = rows, j = col_max,  value = stats$nb_max)
  set(cell_dt, i = rows, j = col_min,  value = stats$nb_min)
  set(cell_dt, i = rows, j = col_mean, value = stats$nb_mean)

  invisible(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Main pipeline (replaces the outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table in place (no copy if already a data.table)
setDT(cell_data)

# Build the edge table once (~38.5 M rows Ã— 2 int cols â‰ˆ 308 MB)
message("Building edge table...")
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_table)
}

# Free the edge table
rm(edge_table); gc()

message("Done. cell_data now has all neighbor features.")
```

---

## Expected Improvement Summary

| Aspect | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | `lapply` over 6.46M rows, per-row `paste` + named-vector lookup | Two vectorized `data.table` merges; no per-row R loop |
| **`compute_neighbor_stats`** | `lapply` over 6.46M rows Ã— 5 vars; `do.call(rbind, ...)` | Vectorized column extraction + `data.table` grouped `max/min/mean` |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes (depending on disk I/O and RAM pressure) |
| **Peak memory** | Unpredictable (ragged list + intermediate copies) | ~308 MB for edge table + ~2â€“3 GB for `cell_data`; well within 16 GB |
| **Numerical output** | `max`, `min`, `mean` of non-NA neighbor values | Identical â€” same aggregation functions, same neighbor definitions |
| **Trained RF model** | â€” | Untouched; features are schema-compatible |

The key insight is that the neighbor lookup is a **relational join** (cell Ã— year matched to neighbor-cell Ã— same-year), and `data.table` executes such joins in optimized C code with radix-based indexing â€” eliminating millions of interpreted R function calls.
END_RESPONSE>>>
