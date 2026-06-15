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
run_id: Anthropic_F09_T07_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

### 1. `build_neighbor_lookup` â€” O(nÂ²)-scale overhead via row-level `lapply` over 6.46M rows

The function iterates over every **cell-year row** (6.46 million), but the neighbor topology is **time-invariant**. There are only 344,208 unique cells; the neighbor graph doesn't change across 28 years. By building the lookup at the cell-year level, the function:

- Creates 6.46M character-pasted keys (`paste(id, year)`).
- For each of the 6.46M rows, looks up neighbor cell IDs, pastes year suffixes onto them, and indexes into a named character vector â€” all in an R-level `lapply` loop with no vectorization.
- The named-vector lookup (`idx_lookup[neighbor_keys]`) is an O(n) hash probe repeated millions of times, and the key construction via `paste()` generates enormous temporary character vectors.

**This is the dominant bottleneck.** The neighbor structure is static. Recomputing it per cell-year is pure waste.

### 2. `compute_neighbor_stats` â€” Repeated R-level loops over 6.46M rows

For each of the 5 variables, another `lapply` over 6.46M rows extracts neighbor values, removes NAs, and computes max/min/mean in pure R. That's 5 Ã— 6.46M = 32.3M R function calls with per-element vector subsetting.

### Summary

| Component | Calls | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations | Rebuilds time-invariant topology per cell-year |
| `compute_neighbor_stats` | 5 Ã— 6.46M iterations | Pure-R loop, no vectorization |
| `paste()` key construction | 6.46M + neighbor expansions | String allocation overhead |

---

## Optimization Strategy

### Core Insight: Separate topology from attributes

The rook-neighbor graph is a **spatial constant**. Build it **once** as a cell-to-cell adjacency table (a two-column `data.table` of `id â†’ neighbor_id`). Then, for each year, **join** the yearly cell attributes onto this table and compute grouped `max`, `min`, `mean` using `data.table` â€” fully vectorized in C, no R-level row loops.

### Steps

1. **Build a static adjacency edge list** from `rook_neighbors_unique` (the `nb` object) and `id_order`. This produces ~1.37M rows of `(id, neighbor_id)`. Done once.

2. **Join yearly attributes** by joining `cell_data[, .(id, year, var)]` onto the edge list by `neighbor_id` and `year`. This expands to ~1.37M Ã— 28 â‰ˆ 38.5M rows but is handled in memory-efficient columnar form by `data.table`.

3. **Group-aggregate** by `(id, year)` to compute `max`, `min`, `mean` of each neighbor variable. This is a single vectorized `data.table` operation â€” no R-level loops.

4. **Join results back** onto `cell_data`.

### Expected speedup

| Step | Old | New |
|---|---|---|
| Build topology | 6.46M R iterations | 344K iterations (once), producing a data.table |
| Compute stats (per var) | 6.46M R iterations | One vectorized `data.table` grouped aggregation |
| **Total estimated time** | **86+ hours** | **~5â€“15 minutes** |

Memory: The adjacency edge list is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. The largest join intermediate (with year expansion) is ~38.5M rows Ã— 4 columns â‰ˆ 1.2 GB per variable, well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the static cell-to-cell adjacency table (run ONCE)
# ==============================================================================
# Inputs:
#   id_order             â€” integer/numeric vector of cell IDs, length 344,208
#                          (positional index matches the nb object)
#   rook_neighbors_unique â€” an nb object (list of integer index vectors)
#
# Output:
#   adj_dt â€” data.table with columns: id, neighbor_id
# ==============================================================================

build_adjacency_table <- function(id_order, neighbors_nb) {
  n <- length(id_order)
  # Pre-allocate lists for speed
  from_ids <- vector("list", n)
  to_ids   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L; skip those
    if (length(nb_idx) == 1L && nb_idx[1L] == 0L) next
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) next
    from_ids[[i]] <- rep(id_order[i], length(nb_idx))
    to_ids[[i]]   <- id_order[nb_idx]
  }
  
  data.table(
    id          = unlist(from_ids, use.names = FALSE),
    neighbor_id = unlist(to_ids,   use.names = FALSE)
  )
}

adj_dt <- build_adjacency_table(id_order, rook_neighbors_unique)
setkey(adj_dt, neighbor_id)  # key on neighbor_id for fast joins

cat("Adjacency table:", nrow(adj_dt), "directed edges\n")

# ==============================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================================

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================================
# STEP 3: Compute neighbor features for all variables â€” vectorized
# ==============================================================================
# For each source variable, we:
#   (a) Join cell_data attributes onto adj_dt by (neighbor_id = id, year)
#   (b) Aggregate by (id, year) to get max, min, mean
#   (c) Join the results back onto cell_data
#
# This replaces both build_neighbor_lookup and compute_neighbor_stats.
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_dt, adj, var_name) {
  # Column names for the output (must match original pipeline's naming)
  col_max  <- paste0("n_max_",  var_name)
  col_min  <- paste0("n_min_",  var_name)
  col_mean <- paste0("n_mean_", var_name)
  
  # Extract only the columns we need from cell_data for the join
  # neighbor_id in adj matches id in cell_dt
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)
  
  # Join: for each edge (id, neighbor_id), attach the neighbor's value and year
  # adj has (id, neighbor_id); we join attr_dt on neighbor_id == id
  # We also need the focal cell's year, so we do a two-step join:
  
  # First, get all (id, year) combinations that exist for focal cells
  focal_years <- cell_dt[, .(id, year)]
  
  # Expand adjacency by year: each edge exists in every year the focal cell exists
  # But it's more efficient to join edges â†’ focal years â†’ neighbor attributes
  
  # Merge focal cell's years onto adjacency
  # This gives us (id, neighbor_id, year) for every edge Ã— year
  setkey(focal_years, id)
  setkey(adj, id)
  edge_year <- adj[focal_years, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: id, neighbor_id, year
  
  # Now join the neighbor's attribute value for that year
  setkey(edge_year, neighbor_id, year)
  setkey(attr_dt, id, year)
  edge_year[attr_dt, on = c(neighbor_id = "id", "year"), neighbor_val := i.val]
  
  # Aggregate by (id, year)
  stats <- edge_year[
    !is.na(neighbor_val),
    .(
      nmax  = max(neighbor_val),
      nmin  = min(neighbor_val),
      nmean = mean(neighbor_val)
    ),
    by = .(id, year)
  ]
  
  setnames(stats, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  # Join back onto cell_data
  setkey(stats, id, year)
  setkey(cell_dt, id, year)
  cell_dt <- stats[cell_dt, on = c("id", "year")]
  
  cell_dt
}

# --- Run for all 5 neighbor source variables ---
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- proc.time()
  cell_data <- compute_neighbor_features_fast(cell_data, adj_dt, var_name)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done in", round(elapsed, 1), "seconds\n")
}

# ==============================================================================
# STEP 4: Predict with the existing trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) is preserved as-is.
# The feature columns in cell_data now include n_max_*, n_min_*, n_mean_*
# with identical numerical values to the original pipeline, so predictions
# are numerically equivalent.
#
# Example (adjust to your actual model object and prediction call):
#
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)
#
# ==============================================================================

cat("\nAll neighbor features computed. cell_data has", ncol(cell_data), "columns and",
    nrow(cell_data), "rows.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | Never touched. We only modify the feature-engineering input pipeline. |
| **Numerical estimand** | The `max`, `min`, `mean` aggregations operate on exactly the same neighbor sets and values. `data.table`'s grouped aggregation is numerically identical to the original R `max()`/`min()`/`mean()` calls. |
| **Column names** | Output columns follow the same `n_max_<var>`, `n_min_<var>`, `n_mean_<var>` naming convention, so the RF model's expected feature names are matched. (Adjust the prefix if your original `compute_and_add_neighbor_features` used a different convention.) |
| **NA handling** | Rows with `is.na(neighbor_val)` are excluded before aggregation, matching the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` for all three stats (via the non-match in the final join), matching the original `return(c(NA, NA, NA))`. |

---

## Memory Profile (16 GB laptop)

| Object | Estimated Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `adj_dt` (1.37M Ã— 2 int cols) | ~11 MB |
| `edge_year` intermediate (1.37M Ã— 28 Ã— 4 cols) | ~1.2 GB peak per variable |
| **Peak total** | **~8â€“9 GB** âœ“ fits in 16 GB |

If memory is tight, the loop processes one variable at a time and the intermediate `edge_year` is garbage-collected between iterations.

---

## Summary

| Metric | Before | After |
|---|---|---|
| Topology construction | 6.46M R-level iterations with string hashing | 344K iterations â†’ 1.37M-row `data.table` (once) |
| Per-variable stats | 6.46M R-level `lapply` calls | One vectorized `data.table` join + grouped agg |
| Estimated wall time | **86+ hours** | **~5â€“15 minutes** |
| Trained RF model | Preserved | Preserved |
| Numerical output | Original | Identical |
END_RESPONSE>>>
