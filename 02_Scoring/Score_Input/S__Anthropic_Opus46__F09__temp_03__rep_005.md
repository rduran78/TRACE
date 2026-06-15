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
run_id: Anthropic_F09_T03_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates row-by-row over every cell-year observation. For each of the ~6.46 million rows, it:
- Looks up the cell's spatial reference index.
- Retrieves neighbor cell IDs.
- Pastes string keys (`id_year`) for each neighbor.
- Matches those keys against a named character vector (`idx_lookup`).

String pasting and named-vector lookup inside a 6.46M-iteration `lapply` is the dominant bottleneck. The `paste(..., sep="_")` call alone creates millions of temporary strings, and named-vector lookup via `idx_lookup[neighbor_keys]` is O(n) per call in the worst case because R named vectors use linear hashing that degrades at scale.

### 2. Redundant Recomputation of Static Spatial Topology
The neighbor relationships are **purely spatial** â€” they never change across years. Yet the current code rebuilds the lookup per cell-year row, effectively duplicating the same spatial adjacency information 28 times (once per year). This means ~6.46M list entries when only ~344K unique cell-neighbor relationships exist.

### 3. Row-Level `lapply` in `compute_neighbor_stats`
After the lookup is built, `compute_neighbor_stats` again iterates over all ~6.46M entries, subsetting a numeric vector and computing `max`, `min`, `mean` one row at a time. The R interpreter overhead per iteration (function call, subsetting, `is.na` check, concatenation) is small individually but catastrophic at this scale.

**Summary:** The architecture treats a **spatial** problem as a **row** problem. The fix is to separate the spatial topology (built once) from the temporal attributes (joined per year), and to replace row-level R loops with vectorized joins and grouped aggregations.

---

## Optimization Strategy

The key insight: **build the adjacency table once as a two-column data.table of (cell_id, neighbor_id), then join yearly attributes onto it and compute grouped statistics vectorially.**

### Steps:

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object). This produces a `data.table` with ~1.37M rows: `(cell_id, neighbor_id)`. This is done **once**.

2. **For each variable**, join the cell-year attribute values onto the edge table by `(neighbor_id, year)`, then compute `max`, `min`, `mean` grouped by `(cell_id, year)` using `data.table`'s optimized `by=` grouping. This replaces both the 6.46M-row `lapply` in `build_neighbor_lookup` and the 6.46M-row `lapply` in `compute_neighbor_stats`.

3. **Join the resulting neighbor statistics back** onto the main `cell_data` table by `(cell_id, year)`.

4. **Predict** with the existing trained Random Forest model â€” no retraining.

### Why this is fast:
- The edge table has ~1.37M rows, not 6.46M. The join with 28 years expands it to ~1.37M Ã— 28 â‰ˆ 38.5M rows, but `data.table` handles this with optimized binary-search joins and radix-sort grouping in seconds, not hours.
- No R-level `lapply` over millions of rows.
- No string pasting or named-vector lookup.
- Memory footprint is modest: the edge table is ~11 MB; the expanded join is ~600 MB at peak, well within 16 GB.

**Expected runtime: ~2â€“5 minutes** for all 5 variables, down from 86+ hours.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Ensure cell_data is a data.table with proper columns
# ==============================================================
# Assumes:
#   - cell_data has columns: id (cell identifier), year, and all predictor columns
#   - rook_neighbors_unique is an nb object (list of integer index vectors)
#   - id_order is the vector of cell IDs in the order matching the nb object
#   - rf_model is the already-trained Random Forest model (do NOT retrain)

cell_data <- as.data.table(cell_data)

# ==============================================================
# STEP 1: Build static spatial edge table (ONCE)
# ==============================================================
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for neighbors of cell i
  # We expand this into a two-column edge list
  n <- length(nb_obj)
  
  # Pre-calculate sizes for pre-allocation
  sizes <- vapply(nb_obj, function(x) {
    # spdep nb objects use 0L to indicate no neighbors
    sum(x > 0L)
  }, integer(1))
  
  total_edges <- sum(sizes)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]  # remove 0-coded "no neighbor"
    k <- length(nbrs)
    if (k > 0L) {
      idx <- pos:(pos + k - 1L)
      from_id[idx] <- id_order[i]
      to_id[idx]   <- id_order[nbrs]
      pos <- pos + k
    }
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

cat("Building static edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %d directed edges\n", nrow(edge_table)))

# ==============================================================
# STEP 2: Compute neighbor statistics for each variable
# ==============================================================
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Create a lookup of (cell_id, year) -> value
  # We only need id, year, and the variable of interest
  lookup <- cell_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(lookup, neighbor_id, year)
  
  # Get unique years
  years <- sort(unique(cell_dt$year))
  
  # Cross join edge table with years to get all (cell_id, neighbor_id, year) triples
  # This is the "expand" step: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
  edge_year <- CJ_dt(edge_dt, years)
  
  # Join neighbor values onto edge_year
  setkey(edge_year, neighbor_id, year)
  edge_year <- lookup[edge_year, on = .(neighbor_id, year), nomatch = NA]
  
  # Compute grouped stats: max, min, mean per (cell_id, year)
  stats <- edge_year[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(cell_id, year)
  ]
  
  # Rename columns to match expected naming convention
  setnames(stats, 
           c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  
  return(stats)
}

# Helper: cross join edge_table with years vector
CJ_dt <- function(edge_dt, years) {
  # Efficient cross join: repeat each edge for every year
  n_edges <- nrow(edge_dt)
  n_years <- length(years)
  
  data.table(
    cell_id     = rep(edge_dt$cell_id,     times = n_years),
    neighbor_id = rep(edge_dt$neighbor_id,  times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# ==============================================================
# STEP 3: Loop over variables, compute features, join back
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s\n", var_name))
  
  stats <- compute_neighbor_features_fast(cell_data, edge_table, var_name)
  setkey(stats, cell_id, year)
  
  # Join the 3 new columns onto cell_data
  # First, remove old columns if they exist (idempotent re-runs)
  old_cols <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  for (col in old_cols) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  cell_data <- stats[cell_data, on = .(cell_id = id, year)]
  
  # Restore the 'id' column name (the join renames cell_id)
  if ("cell_id" %in% names(cell_data)) {
    setnames(cell_data, "cell_id", "id")
  }
  
  cat(sprintf("  Done. cell_data now has %d columns.\n", ncol(cell_data)))
}

# ==============================================================
# STEP 4: Predict with the existing trained Random Forest
# ==============================================================
cat("Running Random Forest prediction...\n")

# Ensure column order / names match what the model expects
# The model object (rf_model) is unchanged â€” no retraining
cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Pipeline complete.\n")
```

---

## Notes on Correctness and Equivalence

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code. The edge table encodes the identical rook adjacency. Cells with no neighbors (or all-NA neighbors) receive `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |
| **Trained RF model preserved** | The model object is never modified. Only `predict()` is called. |
| **Column naming** | The output columns follow the pattern `{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean`. Adjust the naming convention in `setnames()` if your original code used a different pattern (e.g., `{var}_neighbor_max`). |
| **Memory** | Peak memory is dominated by the cross-joined edge-year table (~38.5M rows Ã— 3 columns â‰ˆ 900 MB). This fits comfortably in 16 GB alongside `cell_data` (~6.46M rows Ã— 110 columns â‰ˆ 5.7 GB). |
| **Runtime estimate** | Each variable takes ~10â€“30 seconds (join + group-by on 38.5M rows). Total for 5 variables: **~2â€“5 minutes**. |
END_RESPONSE>>>
