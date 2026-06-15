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
run_id: Anthropic_F09_T00_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N) over all 6.46 million cell-year rows
For every single cell-year row, the function:
- Looks up the cell's spatial neighbors.
- Pastes together `(neighbor_id, year)` string keys.
- Performs named-vector character lookups into `idx_lookup` (a named character vector of length 6.46M).

Named vector lookups in R are **linear scans** on large vectors. With ~6.46M rows, each doing ~4 neighbor lookups into a 6.46M-length named vector, this is catastrophically slow. The string pasting (`paste(id, year)`) for every row compounds the cost.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows
Each call iterates over all 6.46M rows, subsetting and computing `max/min/mean` per row. This is repeated 5 times (once per variable). While less catastrophic than the lookup construction, it is still unnecessarily slow because it's pure R-loop work that could be vectorized.

### Core Insight
The **spatial neighbor topology is static** â€” it does not change across years. The current code rebuilds the full cell-year-to-cell-year mapping in one monolithic step, entangling spatial structure with temporal structure. This is the fundamental design flaw.

---

## Optimization Strategy

**Separate spatial topology from temporal attributes, then use vectorized joins.**

1. **Build a cell-level neighbor edge table once** â€” a simple two-column `data.table` of `(cell_id, neighbor_id)` derived from the `nb` object. This has ~1.37M rows and never changes.

2. **For each variable, join yearly attributes onto the edge table** â€” use `data.table` keyed joins to attach each neighbor's variable value for the matching year. This is a vectorized merge, not a per-row R loop.

3. **Aggregate neighbor stats with `data.table` grouping** â€” compute `max`, `min`, `mean` per `(cell_id, year)` group in one vectorized pass.

4. **Join the aggregated stats back** onto the main dataset.

This eliminates:
- All 6.46M `paste()` calls.
- All named-vector character lookups.
- All `lapply` loops over millions of rows.

**Expected speedup**: from 86+ hours to **minutes** (typically 5â€“15 minutes on a 16 GB laptop).

**Numerical equivalence**: The `max`, `min`, and `mean` are computed over exactly the same neighbor sets and values, preserving the original estimand. The trained Random Forest model is never touched.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build a static cell-level neighbor edge table (done once)
# ==============================================================================
# Input:
#   id_order            â€” vector of cell IDs in the order matching the nb object
#   rook_neighbors_unique â€” spdep nb object (list of integer index vectors)
#
# Output:
#   neighbor_edges â€” data.table with columns (cell_id, neighbor_id)
#                    ~1.37M rows, one per directed neighbor relationship

build_neighbor_edge_table <- function(id_order, neighbors_nb) {
  # For each cell index, expand its neighbor indices into (focal, neighbor) pairs
  n <- length(neighbors_nb)
  focal_idx <- rep(seq_len(n), lengths(neighbors_nb))
  neighbor_idx <- unlist(neighbors_nb)
  
  # Remove the 0-entries that spdep uses to denote "no neighbors"
  valid <- neighbor_idx > 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]
  
  data.table(
    cell_id     = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

neighbor_edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Neighbor edge table: %s rows (expected ~1,373,394)\n",
  format(nrow(neighbor_edges), big.mark = ",")
))

# ==============================================================================
# STEP 2: Convert main data to data.table (if not already)
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist and are of consistent type
stopifnot("id" %in% names(cell_data), "year" %in% names(cell_data))

# ==============================================================================
# STEP 3: For each neighbor source variable, compute neighbor max/min/mean
#          via vectorized joins and grouped aggregation
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_dt, edges, var_name) {
  # Subset to only the columns we need for the join
  # cell_dt must have: id, year, <var_name>
  lookup_cols <- c("id", "year", var_name)
  lookup <- cell_dt[, ..lookup_cols]
  setnames(lookup, old = "id", new = "neighbor_id")
  
  # Key the lookup for fast join
  setkeyv(lookup, c("neighbor_id", "year"))
  
  # Expand edges Ã— years: join neighbor attributes onto the edge table
  # Start with edges, add year from the focal cell, then join neighbor value
  # 
  # Strategy: 
  #   1. Create (cell_id, year) from cell_dt
  #   2. Join edges to get (cell_id, year, neighbor_id)
  #   3. Join lookup to get neighbor's variable value
  #   4. Aggregate by (cell_id, year)
  
  # Get unique (cell_id, year) combinations from the focal cells that appear in edges
  focal <- unique(cell_dt[, .(cell_id = id, year)])
  
  # Cross join: focal Ã— edges  â†’  (cell_id, year, neighbor_id)
  # But we only want edges for each cell_id, so this is an inner join on cell_id
  setkeyv(edges, "cell_id")
  setkeyv(focal, "cell_id")
  
  # Join: for each (cell_id, year), attach all neighbor_ids
  expanded <- edges[focal, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # Result columns: cell_id, neighbor_id, year
  
  # Now join the neighbor's variable value for the same year
  setkeyv(expanded, c("neighbor_id", "year"))
  expanded[lookup, on = c("neighbor_id", "year"), paste0("n_", var_name) := get(var_name)]
  
  val_col <- paste0("n_", var_name)
  
  # Aggregate: max, min, mean per (cell_id, year), dropping NAs
  agg <- expanded[!is.na(get(val_col)),
                  .(
                    nb_max  = max(get(val_col)),
                    nb_min  = min(get(val_col)),
                    nb_mean = mean(get(val_col))
                  ),
                  by = .(cell_id, year)]
  
  # Name the output columns to match the original pipeline's naming convention
  setnames(agg, 
           old = c("nb_max", "nb_min", "nb_mean"),
           new = paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  
  return(agg)
}

# ==============================================================================
# STEP 4: Loop over variables, compute features, and join back to cell_data
# ==============================================================================
for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s ...\n", var_name))
  
  agg <- compute_neighbor_features_fast(cell_data, neighbor_edges, var_name)
  
  # Merge back onto cell_data
  # Use cell_id â†’ id mapping
  cell_data <- merge(
    cell_data, agg,
    by.x = c("id", "year"), by.y = c("cell_id", "year"),
    all.x = TRUE, sort = FALSE
  )
  
  cat(sprintf("  Done. Added: %s\n",
              paste(paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")), collapse = ", ")))
}

# ==============================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) is used as-is.
# cell_data now contains the same 15 neighbor feature columns (5 vars Ã— 3 stats)
# that the model expects, with numerically identical values.
#
# Example (adjust to your actual model object and prediction call):
#
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
#

cat("Neighbor feature computation complete. Ready for prediction.\n")
```

---

## Why This Is Numerically Equivalent

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor set per cell-year | Rook neighbors present in that year | Identical â€” same edge table, same `nomatch=0` / NA filtering |
| `max` | `max(neighbor_vals)` after removing NAs | `max(get(val_col))` after `!is.na` filter â€” identical |
| `min` | `min(neighbor_vals)` after removing NAs | `min(get(val_col))` after `!is.na` filter â€” identical |
| `mean` | `mean(neighbor_vals)` after removing NAs | `mean(get(val_col))` after `!is.na` filter â€” identical |
| Cells with no valid neighbors | Returns `NA, NA, NA` | `all.x = TRUE` merge leaves `NA` â€” identical |

The trained Random Forest model is never modified, retrained, or re-tuned. Only the feature-engineering step is accelerated.

---

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M string lookups into named vector) | ~1 second (vectorized `unlist` + indexing) |
| Per-variable neighbor stats | ~hours (6.46M `lapply` iterations Ã— 5) | ~1â€“3 min each (keyed `data.table` join + grouped agg) |
| **Total estimated time** | **86+ hours** | **5â€“15 minutes** |
| RAM peak | Moderate (but slow) | ~4â€“6 GB peak during the `allow.cartesian` expansion (fits in 16 GB) |
END_RESPONSE>>>
