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
run_id: Anthropic_F06_T10_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows

For every single cell-year row, the function:
1. Looks up the spatial cell's reference index (`id_to_ref`).
2. Retrieves that cell's rook neighbors from the `nb` object.
3. Constructs string keys by pasting neighbor IDs with the current year.
4. Matches those keys against `idx_lookup` (a named character vector).

**String concatenation (`paste`) and named-vector lookup on 6.46 million rows is extremely slow in R.** The `idx_lookup` named vector has ~6.46M entries; each lookup is effectively a hash-table probe, but doing it millions of times inside `lapply` with character key construction is the dominant cost.

The resulting `neighbor_lookup` list has **6.46 million elements**, each containing integer indices. Just constructing and storing this list is memory-heavy and slow.

### Bottleneck B: `compute_neighbor_stats` â€” Another O(n) `lapply` over 6.46 million rows

For each of 5 variables, it iterates over all 6.46M rows, subsets a numeric vector by the neighbor indices, removes NAs, and computes `max`, `min`, `mean`. That's 5 Ã— 6.46M = 32.3 million R-level function calls with subsetting.

### Why raster focal/kernel operations don't directly apply

The hint in the prompt about raster focal operations is a useful *analogy*: focal operations compute neighborhood summaries over a regular grid extremely efficiently using compiled C code. However, this panel's spatial topology is defined by an irregular `spdep::nb` object (not necessarily a regular raster grid), and the data is in long (cell-year) panel format. Forcing it into a raster stack would require confirming a regular grid and reshaping â€” and could introduce subtle mismatches with the `nb` object. **The implementation below preserves the exact `nb`-defined neighbor relationships** while borrowing the *spirit* of focal operations: vectorized, column-wise, compiled-code computation over neighbor groups.

### Summary of time sinks

| Step | Calls | Estimated share of 86 hrs |
|---|---|---|
| `build_neighbor_lookup` (string ops) | 6.46M | ~40â€“50% |
| `compute_neighbor_stats` (5 vars) | 32.3M | ~50â€“60% |

---

## 2. Optimization Strategy

### Strategy: Fully vectorized sparse-matrix approach

**Key insight:** The neighbor relationship is *time-invariant* â€” the same spatial neighbors apply to every year. We can:

1. **Expand the spatial neighbor list (344K cells) into a cell-year adjacency structure using `data.table` joins** â€” replacing all string-paste and named-vector lookups with integer-indexed operations.
2. **Compute all neighbor statistics using a single grouped `data.table` aggregation** per variable â€” replacing 6.46M `lapply` iterations with compiled C-level grouped operations.

This reduces the 86+ hour runtime to **minutes**.

### Specific techniques

| Original | Optimized |
|---|---|
| `paste()` + named vector lookup | `data.table` integer join on `(id, year)` |
| `lapply` over 6.46M rows | `data.table` grouped `[, .(max, min, mean), by=...]` |
| One list element per cell-year | One row in an edge table per directed neighbor-year pair |
| 5 separate full passes | 5 grouped aggregations on the same edge table structure |

### Memory estimate

- Edge table: ~1.37M spatial edges Ã— 28 years = ~38.4M rows Ã— ~3 integer columns â‰ˆ 900 MB. Fits in 16 GB.
- The original `cell_data` (~6.46M rows Ã— 110 cols) is perhaps 5â€“6 GB. Total stays within 16 GB.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure there's a row-order column so we can restore original order later
cell_data[, .row_id := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a spatial edge table from the nb object
#
#   rook_neighbors_unique: an nb object of length = length(id_order)
#   id_order: vector mapping nb-list position -> cell id
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  # For each spatial cell, expand its neighbor list into (from_id, to_id) rows
  from_list <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_list   <- unlist(nb_obj)

  # Remove the 0-neighbor sentinel that spdep uses (nb encodes no-neighbors as 0L)
  valid <- to_list != 0L
  from_list <- from_list[valid]
  to_list   <- to_list[valid]

  data.table(
    from_id = id_order[from_list],
    to_id   = id_order[to_list]
  )
}

spatial_edges <- build_edge_table(id_order, rook_neighbors_unique)
cat("Spatial edges:", nrow(spatial_edges), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Expand spatial edges across all years via join
#
# Instead of 6.46M string-paste lookups, we join the edge
# table against cell_data on (id, year) for the neighbor side.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Unique years in the panel
years <- sort(unique(cell_data$year))

# Cross-join edges Ã— years  (~38.4M rows)
edge_year <- spatial_edges[, CJ(from_id = from_id, to_id = to_id, year = years),
                           .SDcols = c("from_id", "to_id")]
# More memory-efficient cross join:
edge_year <- spatial_edges[rep(seq_len(.N), each = length(years))]
edge_year[, year := rep(years, times = nrow(spatial_edges))]

cat("Edge-year rows:", nrow(edge_year), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Attach neighbor values and compute stats
#
# For each neighbor source variable, join neighbor values onto
# the edge table, then aggregate by (from_id, year).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor stats for:", var_name, "\n")

  # Subset only needed columns from cell_data for the join (memory efficiency)
  lookup_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(lookup_dt, id, year)

  # Join: for each edge-year row, get the neighbor cell's value
  # Join on to_id == id AND year == year
  edge_vals <- lookup_dt[edge_year, on = .(id = to_id, year = year),
                         .(from_id, year, val = x.val),
                         nomatch = NA]

  # Compute grouped stats: max, min, mean of non-NA neighbor values
  stats <- edge_vals[!is.na(val),
                     .(nmax  = max(val),
                       nmin  = min(val),
                       nmean = mean(val)),
                     by = .(from_id, year)]

  # Name the new columns to match original convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # Join stats back onto cell_data
  # Rows with no valid neighbors will get NA (desired behavior matching original)
  cell_data <- stats[cell_data, on = .(from_id = id, year = year)]

  # The join renames from_id; fix it back
  setnames(cell_data, "from_id", "id")

  # Clean up

  rm(lookup_dt, edge_vals, stats)
  gc()
}

# Restore original row order
setorder(cell_data, .row_id)
cell_data[, .row_id := NULL]

cat("Done. cell_data now has", ncol(cell_data), "columns and", nrow(cell_data), "rows.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the pre-trained Random Forest
#
# The model is NOT retrained. We simply call predict().
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example (adjust to your actual model object and predict interface):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor definitions** | We use the exact same `rook_neighbors_unique` nb object â€” no approximation, no raster resampling. |
| **Same statistics** | `max`, `min`, `mean` of non-NA neighbor values, identical to the original `compute_neighbor_stats`. |
| **Same NA handling** | Rows with no neighbors (or all-NA neighbors) get `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |
| **Same RF model** | The trained model is loaded and used as-is via `predict()`; no retraining occurs. |
| **Column names** | Named identically (`neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc.) so the RF model's expected feature names match. |

---

## 5. Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~40 hrs (6.46M string ops) | ~30 sec (integer cross-join) |
| Compute stats (5 vars) | ~46 hrs (32.3M lapply calls) | ~5â€“10 min (5 grouped aggregations) |
| **Total** | **86+ hours** | **~10â€“15 minutes** |

The speedup factor is roughly **300â€“500Ã—**, achieved entirely by replacing R-level iteration and string operations with `data.table`'s compiled, vectorized join and group-by machinery.
END_RESPONSE>>>
