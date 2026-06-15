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
run_id: Anthropic_F09_T00_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor Ã— that row's year (expensive string allocation).
- Matches those keys against a named character vector (`idx_lookup`) of length 6.46M (expensive named-vector lookup â€” R's named vector lookup is O(n) in the worst case or hash-based but still slow at this scale).

This means **~6.46 million iterations**, each doing string construction and hash lookups against a 6.46M-entry table. The string-key approach is the primary bottleneck.

### 2. `compute_neighbor_stats` is less problematic but still suboptimal
It iterates over the 6.46M-entry `neighbor_lookup` list, subsetting a numeric vector each time. This is tolerable but could be vectorized.

### 3. The fundamental architectural flaw
The neighbor lookup is **rebuilt for every cell-year row**, even though **the spatial neighbor topology is static** â€” cell A's neighbors are the same in 1992 as in 2019. The only thing that changes yearly is the attribute values. The current code conflates spatial topology with temporal indexing.

---

## Optimization Strategy

**Core idea:** Separate the static spatial topology from the dynamic yearly attributes.

1. **Build a spatial-only neighbor edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-invariant.

2. **For each year and each variable**, join the yearly attribute values onto the neighbor edge table, then aggregate (max, min, mean) by `cell_id` using `data.table` grouped operations. This replaces all `lapply` loops with vectorized joins and group-by aggregations.

3. **Join the resulting neighbor stats back** onto the main `cell_data` table.

**Expected speedup:** The ~6.46M-row `lapply` with string hashing is replaced by `data.table` keyed joins and `group_by` aggregations over ~1.37M Ã— 28 â‰ˆ 38.5M edge-year rows per variable. `data.table` does this in seconds to low minutes per variable. Total runtime: **minutes instead of days**.

**Memory:** The edge table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. The expanded edge-year table is ~38.5M rows Ã— 3 columns â‰ˆ 900 MB per variable at peak, which fits in 16 GB RAM comfortably (and we process one variable at a time).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static spatial neighbor edge table (once, ~1.37M rows)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Inputs:
#   id_order             â€” integer/numeric vector of cell IDs (length 344,208)
#   rook_neighbors_unique â€” spdep nb object (list of length 344,208)
#
# Output:
#   neighbor_edges â€” data.table with columns (cell_id, neighbor_id)

build_neighbor_edge_table <- function(id_order, neighbors_nb) {
  n <- length(id_order)
  # Pre-allocate: count total edges
  edge_counts <- vapply(neighbors_nb, length, integer(1))
  total_edges <- sum(edge_counts)

  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    len <- length(nb_idx)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_idx]
    pos <- pos + len
  }

  # Trim if any nb entries were empty (0-neighbor cells)
  if (pos <= total_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

neighbor_edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)
setkey(neighbor_edges, neighbor_id)  # key on neighbor_id for fast join

cat("Neighbor edge table:", nrow(neighbor_edges), "directed edges\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year columns are present
stopifnot(all(c("id", "year") %in% names(cell_data)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor stats for each variable, join back
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_neighbor_features_fast <- function(cell_dt, neighbor_edges, var_name) {
  cat("Processing neighbor features for:", var_name, "...\n")

  # Extract only the columns we need: id, year, and the variable
  # This keeps memory usage low
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # Expand neighbor edges by year:
  #   For each (cell_id, neighbor_id) edge and each year,
  #   look up the neighbor's attribute value, then aggregate.
  #
  # Strategy: join attr_dt onto neighbor_edges by neighbor_id = id,
  #   which gives us (cell_id, neighbor_id, year, value_of_neighbor).
  #   Then group by (cell_id, year) to get max, min, mean.

  # Create the join: for every edge, get all year-value pairs of the neighbor
  setkey(attr_dt, id)
  setkey(neighbor_edges, neighbor_id)

  # This join expands to ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
  # Each row: (cell_id, neighbor_id, year, value)
  edge_year <- neighbor_edges[attr_dt,
    .(cell_id, year = i.year, neighbor_value = i.value),
    on = .(neighbor_id = id),
    nomatch = NULL,
    allow.cartesian = TRUE
  ]

  # Aggregate: for each (cell_id, year), compute max/min/mean of neighbor values
  stats <- edge_year[
    !is.na(neighbor_value),
    .(
      nb_max  = max(neighbor_value),
      nb_min  = min(neighbor_value),
      nb_mean = mean(neighbor_value)
    ),
    by = .(cell_id, year)
  ]

  # Rename columns to match expected naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Join stats back onto cell_data by (id, year)
  setkey(stats, cell_id, year)
  setkey(cell_dt, id, year)

  cell_dt <- stats[cell_dt, on = .(cell_id = id, year = year)]

  # The join creates a cell_id column; rename back to id
  if ("cell_id" %in% names(cell_dt)) {
    cell_dt[, cell_id := NULL]  
    # id is preserved from the right table in a right join (X[i])
  }

  # Clean up large intermediate objects

  rm(edge_year, stats, attr_dt)
  gc()

  cat("  Done:", var_name, "\n")
  return(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3 (alternative, cleaner join logic to avoid column confusion):
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_dt, neighbor_edges, var_name) {
  cat("Processing neighbor features for:", var_name, "\n")
  t0 <- proc.time()

  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Remove old columns if re-running
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Subset: only id, year, variable
  attr_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]

  # Join neighbor attribute values onto edge table
  # Result: for each directed edge (cell_id -> neighbor_id) and each year,
  #         the neighbor's value of var_name
  edge_vals <- merge(
    neighbor_edges,
    attr_dt,
    by = "neighbor_id",
    allow.cartesian = TRUE
  )
  # edge_vals columns: neighbor_id, cell_id, year, val

  # Aggregate by (cell_id, year)
  stats <- edge_vals[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    keyby = .(cell_id, year)
  ]

  rm(edge_vals, attr_dt)
  gc()

  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # Merge back onto cell_dt
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)

  rm(stats)
  gc()

  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done:", var_name, "in", round(elapsed, 1), "seconds\n")
  return(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Run for all 5 variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, neighbor_edges, var_name)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained RF model object is preserved as-is.
# Ensure cell_data has the same column names/order the model expects.
# Example (adjust to your actual model object name):
#
#   cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
#
# The numerical estimand is unchanged because the neighbor max, min,
# and mean values are computed identically â€” only the implementation
# mechanism changed (vectorized joins vs. row-wise lapply).
```

---

## Summary of Changes and Guarantees

| Aspect | Before | After |
|---|---|---|
| **Neighbor topology** | Re-derived per cell-year row via string keys | Built once as a 1.37M-row edge `data.table` |
| **Neighbor stats** | `lapply` over 6.46M rows with named-vector lookup | `data.table` merge + grouped aggregation |
| **Estimated runtime** | ~86+ hours | **~5â€“15 minutes** (5 variables Ã— 1â€“3 min each) |
| **Peak RAM** | Moderate but slow | ~2â€“4 GB peak per variable (fits in 16 GB) |
| **Numerical results** | max, min, mean of rook neighbors | **Identical** â€” same directed edges, same aggregation functions |
| **Trained RF model** | â€” | **Untouched** â€” no retraining, same `predict()` call |

The key insight is that **spatial topology is time-invariant**: the ~1.37M neighbor edges are the same across all 28 years. By separating topology from attributes and using `data.table`'s optimized hash joins and grouped aggregations, the entire pipeline drops from days to minutes.
END_RESPONSE>>>
