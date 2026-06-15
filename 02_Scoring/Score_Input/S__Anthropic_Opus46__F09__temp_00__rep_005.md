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
run_id: Anthropic_F09_T00_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(n) over all 6.46 million cell-year rows
For every single cell-year row, the function:
- Looks up the cell's spatial neighbors.
- Pastes together `(neighbor_id, year)` string keys.
- Performs named-vector character lookups into `idx_lookup` (a named character vector of length 6.46M).

Named vector lookups in R are **linear scans** on large vectors. With ~6.46M rows, each doing ~4 neighbor lookups into a 6.46M-length named vector, this is catastrophically slow. The string pasting (`paste(id, year)`) for every row compounds the cost.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows
Each call iterates over all 6.46M rows, subsetting and computing `max/min/mean` per row. This is repeated for each of the 5 neighbor source variables (5 Ã— 6.46M iterations). While each individual operation is small, the R-level loop overhead on 32.3 million iterations is substantial.

### Root Cause Summary
The fundamental architectural mistake is that **the spatial neighbor topology is time-invariant, but the lookup is rebuilt entangled with time**. The neighbor relationships between cells never change across years â€” only the attribute values do. By conflating spatial structure with temporal data, the code forces a 6.46M-row loop where a 344,208-cell loop (or better, a vectorized join) would suffice.

---

## Optimization Strategy

### Core Idea: Separate Spatial Topology from Temporal Attributes

1. **Build a static cell-neighbor edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is time-invariant.

2. **For each variable, join yearly attributes onto the edge table** â€” use `data.table` keyed joins to attach each neighbor's attribute value for each year in a single vectorized operation.

3. **Aggregate neighbor stats with `data.table` grouped operations** â€” compute `max`, `min`, `mean` per `(cell_id, year)` group in one vectorized pass.

### Complexity Reduction

| Step | Current | Proposed |
|---|---|---|
| Neighbor lookup construction | 6.46M string pastes + named vector lookups | 1.37M-row static edge table (built once) |
| Per-variable stats | `lapply` over 6.46M rows Ã— 5 vars | Vectorized `data.table` join + group-by Ã— 5 vars |
| Estimated time | ~86+ hours | **~2â€“5 minutes** |

### Constraints Preserved
- The trained Random Forest model is **not retouched**.
- The output columns (neighbor max, min, mean for each variable) are **numerically identical**.
- Memory footprint stays well within 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a static cell-neighbor edge table (time-invariant)
#
# rook_neighbors_unique : spdep nb object (list of integer index vectors)
# id_order              : vector of cell IDs in the same order as the nb object
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for cell i's neighbors
  n <- length(id_order)
  
  # Pre-allocate: count total edges
  edge_counts <- vapply(neighbors, length, integer(1))
  total_edges <- sum(edge_counts)
  
  # Build vectors directly
  from_id <- rep(id_order, times = edge_counts)
  to_id   <- id_order[unlist(neighbors, use.names = FALSE)]
  
  edge_dt <- data.table(cell_id = from_id, neighbor_id = to_id)
  return(edge_dt)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (cell_id, neighbor_id)
# This is built ONCE and reused for every variable and every year.

cat(sprintf("Edge table: %d directed neighbor relationships\n", nrow(edge_table)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: For each neighbor source variable, compute neighbor stats
#         via vectorized join + grouped aggregation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Create a slim lookup: (cell_id, year, value)
  lookup <- cell_dt[, .(cell_id = id, year, value = get(var_name))]
  setkey(lookup, cell_id, year)
  
  # Cross join edge table with all years present in the data
  # Instead of a full cross join (expensive), we join edges onto the data:
  #
  # For each (cell_id, year) row, we need the neighbor values.
  # Strategy: 
  #   1. Join cell_dt's (id, year) with edge_table to get (cell_id, year, neighbor_id)
  #   2. Join that result with lookup on (neighbor_id, year) to get neighbor values
  #   3. Aggregate by (cell_id, year)
  
  # Step 3a: Expand edges by year
  # Get unique (cell_id, year) pairs from the data
  cell_years <- cell_dt[, .(cell_id = id, year)]
  setkey(cell_years, cell_id)
  
  # Set key on edge_table for join
  edge_copy <- copy(edge_dt)
  setkey(edge_copy, cell_id)
  
  # Join: for each (cell_id, year), attach all neighbor_ids
  # This produces ~1.37M * 28 â‰ˆ 38.5M rows (but many cells don't have all years;
  # the actual count depends on the panel balance)
  expanded <- edge_copy[cell_years, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: cell_id, neighbor_id, year
  
  # Step 3b: Look up neighbor values
  setkey(expanded, neighbor_id, year)
  expanded[lookup, on = .(neighbor_id = cell_id, year = year), neighbor_val := i.value]
  
  # Step 3c: Aggregate by (cell_id, year), dropping NAs
  stats <- expanded[!is.na(neighbor_val),
                    .(nb_max  = max(neighbor_val),
                      nb_min  = min(neighbor_val),
                      nb_mean = mean(neighbor_val)),
                    by = .(cell_id, year)]
  
  # Rename columns to match original naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  return(stats)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Loop over the 5 variables, compute stats, and join back
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data has a key for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s ...\n", var_name))
  
  stats_dt <- compute_neighbor_features_fast(cell_data, edge_table, var_name)
  setkey(stats_dt, cell_id, year)
  
  # Determine the new column names
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # Join stats back onto cell_data
  cell_data[stats_dt, on = .(id = cell_id, year = year),
            c(max_col, min_col, mean_col) := mget(paste0("i.", c(max_col, min_col, mean_col)))]
  
  cat(sprintf("  Done: %s\n", var_name))
}

cat("All neighbor features computed.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the existing trained Random Forest model
#
# The trained model object (e.g., `rf_model`) is unchanged.
# cell_data now has the same neighbor feature columns as before,
# with numerically identical values.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Example (adjust to your actual model object and prediction call):
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Identical

| Original code | Optimized code | Equivalence |
|---|---|---|
| For row `i`, finds neighbor row indices via string key lookup | For `(cell_id, year)`, joins `edge_table` on `cell_id`, then joins attribute on `(neighbor_id, year)` | Same neighbor set, same values |
| `max(neighbor_vals)` | `max(neighbor_val)` grouped by `(cell_id, year)` | Identical |
| `min(neighbor_vals)` | `min(neighbor_val)` grouped by `(cell_id, year)` | Identical |
| `mean(neighbor_vals)` | `mean(neighbor_val)` grouped by `(cell_id, year)` | Identical |
| NAs excluded before aggregation | `!is.na(neighbor_val)` filter before aggregation | Identical |
| Cells with no valid neighbors get `NA` | Left join leaves `NA` for unmatched `(id, year)` | Identical |

## Memory Estimate

| Object | Approximate Size |
|---|---|
| `edge_table` (1.37M rows Ã— 2 int cols) | ~11 MB |
| `expanded` (worst case ~38.5M rows Ã— 3 cols) | ~920 MB |
| `cell_data` (6.46M rows Ã— ~125 cols) | ~6.5 GB |
| **Total peak** | **~8â€“9 GB** (fits in 16 GB) |

## Expected Runtime

- Edge table construction: **< 1 second**
- Per-variable join + aggregation: **~20â€“40 seconds each**
- Total for 5 variables: **~2â€“4 minutes**
- **Speedup: ~1,500Ã— compared to the original 86+ hours**
END_RESPONSE>>>
