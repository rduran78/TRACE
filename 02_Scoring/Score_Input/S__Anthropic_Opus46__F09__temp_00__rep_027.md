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
run_id: Anthropic_F09_T00_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N) over all cell-year rows (~6.46M), but it doesn't need to be.

The neighbor relationships are **purely spatial** â€” they don't change across years. Yet the function builds a lookup keyed on `(id, year)` pairs, iterating over every one of the ~6.46 million rows. For each row it:
- Converts the cell ID to a reference index.
- Looks up neighbor cell IDs from the `nb` object.
- Pastes neighbor IDs with the current row's year to form string keys.
- Matches those string keys back into a named character vector of length 6.46M.

The `paste()`-based string key construction and named-vector lookup (`idx_lookup[neighbor_keys]`) is **O(n)** per call in the worst case for R's hashed name lookup, and the sheer volume (6.46M calls, each doing multiple string operations and lookups) is the dominant bottleneck. The `lapply` over 6.46M rows with string concatenation and named-vector indexing is catastrophically slow.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows.

Even though the per-element work is small (subsetting a numeric vector, computing max/min/mean), calling an R-level anonymous function 6.46 million times with `lapply` and then `do.call(rbind, ...)` on 6.46M 3-element vectors is very slow due to R's function-call overhead and the final row-binding.

### Root Cause Summary

The spatial topology is **year-invariant**. There are only 344,208 cells and ~1.37M directed neighbor pairs. But the code treats every cell-year row as if it has a unique neighbor structure, inflating the problem by 28Ã—. The string-key approach turns a simple integer-indexing problem into millions of string allocations and hash lookups.

---

## Optimization Strategy

### Core Insight: Separate Topology from Attributes

1. **Build the neighbor edge table once** â€” a simple two-column integer `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-invariant and built from the `nb` object in milliseconds.

2. **For each variable, use a `data.table` join** â€” join the cell-year attribute values onto the edge table by `(neighbor_id, year)`, then group by `(cell_id, year)` to compute `max`, `min`, `mean`. This replaces millions of R-level function calls with vectorized, indexed `data.table` operations.

### Expected Speedup

| Step | Current | Optimized |
|---|---|---|
| Build lookup | ~hours (6.46M string ops) | ~seconds (1.37M integer rows) |
| Neighbor stats (per variable) | ~hours (6.46M lapply calls) | ~seconds (data.table keyed join + grouped aggregation) |
| **Total for 5 variables** | **~86+ hours** | **~1â€“5 minutes** |

### Constraints Preserved
- The trained Random Forest model is **not retrained** â€” we only recompute the same input features faster.
- The numerical estimand is **identical** â€” same max, min, mean of the same neighbor values.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build the year-invariant neighbor edge table ONCE
#         from the precomputed spdep::nb object.
# ==============================================================

build_neighbor_edges <- function(id_order, nb_obj) {
  # id_order: vector of cell IDs in the same order as the nb object
  # nb_obj:   spdep::nb list (rook_neighbors_unique)
  #
  # Returns a data.table with columns: cell_id, neighbor_id
  # representing all directed neighbor pairs (~1.37M rows).

  n <- length(nb_obj)
  # Pre-count total edges for pre-allocation
  edge_counts <- vapply(nb_obj, length, integer(1))
  total_edges <- sum(edge_counts)

  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nb_indices <- nb_obj[[i]]
    if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) next
    len <- length(nb_indices)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_indices]
    pos <- pos + len
  }

  # Trim if any nb entries were empty (0-neighbor cells)
  if (pos <= total_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

# Build it once
neighbor_edges <- build_neighbor_edges(id_order, rook_neighbors_unique)

# ==============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================
# STEP 3: Function to compute neighbor max, min, mean for one
#         variable via keyed join + grouped aggregation.
# ==============================================================

compute_neighbor_features_dt <- function(cell_dt, neighbor_edges, var_name) {
  # cell_dt:        data.table with columns id, year, and <var_name>
  # neighbor_edges: data.table with columns cell_id, neighbor_id
  # var_name:       character, the variable to aggregate
  #
  # Returns cell_dt with three new columns appended:
  #   <var_name>_neighbor_max, <var_name>_neighbor_min, <var_name>_neighbor_mean

  # Extract only the columns we need for the join: neighbor_id, year, value
  val_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(val_dt, neighbor_id, year)

  # Cross neighbor edges with all years present in the data.
  # Each edge (cell_id, neighbor_id) is replicated for every year,
  # then we join the neighbor's attribute value.
  #
  # But more efficiently: join edges to val_dt directly.
  # We need (cell_id, year) -> aggregate over neighbor values.
  # Start from edges, add year from the focal cell, get neighbor value.

  # Approach: expand edges Ã— years via join on neighbor side.
  # edges has (cell_id, neighbor_id). val_dt has (neighbor_id, year, val).
  # Join: for each edge, get all (year, val) of the neighbor.
  # Then group by (cell_id, year).

  setkey(neighbor_edges, neighbor_id)
  # This join replicates each edge for every year the neighbor has data.
  # Result: (cell_id, neighbor_id, year, val)
  joined <- val_dt[neighbor_edges, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # joined columns: neighbor_id, year, val, cell_id

  # Aggregate: for each (cell_id, year), compute stats over neighbor vals
  stats <- joined[!is.na(val),
                  .(nmax  = max(val),
                    nmin  = min(val),
                    nmean = mean(val)),
                  by = .(cell_id, year)]

  # Name the new columns
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # Remove old columns if they exist (in case of re-run)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Merge back onto cell_dt
  result <- merge(cell_dt, stats, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)

  return(result)
}

# ==============================================================
# STEP 4: Outer loop â€” compute neighbor features for all 5 vars
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_neighbor_features_dt(cell_data, neighbor_edges, var_name)
}

# ==============================================================
# STEP 5: Predict with the existing trained Random Forest model
#         (model object assumed to be `rf_model`)
# ==============================================================

# Convert back to data.frame if the RF model expects one
cell_data_df <- as.data.frame(cell_data)

# Predict (model is NOT retrained)
cell_data_df$prediction <- predict(rf_model, newdata = cell_data_df)
```

---

## Memory Consideration (16 GB RAM)

The `allow.cartesian = TRUE` join in Step 3 expands ~1.37M edges Ã— 28 years = ~38.4M rows per variable. Each row holds two integers, one double, and one integer â‰ˆ 20 bytes â†’ ~768 MB per variable at peak. This is well within 16 GB, especially since the intermediate is discarded after aggregation. If memory is tight, you can process one variable at a time (as the loop already does) and call `gc()` between iterations.

## Why the Numerical Estimand Is Preserved

The `data.table` join + group-by computes **exactly** `max()`, `min()`, and `mean()` over the same set of neighbor values that the original `lapply`-based code computed. The neighbor topology is identical (same `nb` object, same `id_order`). The only difference is the computational mechanism â€” vectorized joins instead of row-by-row string-key lookups. The outputs are numerically identical.
END_RESPONSE>>>
