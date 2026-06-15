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
run_id: Anthropic_F08_T07_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of ~6.46 million entries, one per cell-year row. But the neighbor *relationship* between cells is purely spatial and **never changes across years**. There are only 344,208 unique cells. The function is doing 28Ã— redundant work by recomputing the same neighbor cell mappings for every year.

2. **String-key hashing is expensive at scale.** The function creates `paste(id, year)` keys for ~6.46M rows, builds a named lookup vector, and then for each of the 6.46M rows, constructs neighbor keys via `paste()` and performs named-vector lookups. This is O(nÂ²)-flavored in practice due to repeated string operations.

3. **`compute_neighbor_stats` iterates with `lapply` over 6.46M entries.** Each call extracts neighbor values, removes NAs, and computes max/min/mean. This is done 5 times (once per variable), totaling ~32.3 million R-level function calls with per-element subsetting.

4. **The neighbor lookup list itself is enormous.** Storing ~6.46M integer vectors (each containing ~4 neighbor row indices) consumes substantial RAM and has poor cache locality.

### The Key Insight

The **neighbor graph is static** (it depends only on cell spatial positions, which don't change). The **variable values are dynamic** (they change by year). Therefore:

- Build the neighbor topology **once**, at the cell level (344K entries, not 6.46M).
- For each year, subset the data, use the cell-level neighbor list to gather neighbor values, and compute stats **within that year-slice** using vectorized/matrix operations.

---

## Optimization Strategy

### 1. Separate Static Topology from Dynamic Computation

Build a **cell-level** neighbor lookup once: a list of length 344,208 where entry `i` contains the integer indices of cell `i`'s neighbors within the cell-id ordering. This is directly available from `rook_neighbors_unique` (the `nb` object).

### 2. Vectorized Year-Slice Computation

For each year:
- Extract the year-slice (a data.frame or matrix of ~344K rows, one per cell, ordered by `id_order`).
- For each variable, use the cell-level neighbor list to gather neighbor values into a fixed-width matrix (max 4 rook neighbors), then compute `rowMaxs`, `rowMins`, `rowMeans` via vectorized operations (e.g., `matrixStats` or base R `pmax`/`pmin`).

### 3. Avoid String Operations Entirely

Since cells are ordered by `id_order` and the `nb` object already uses positional indices, no string keys or hash lookups are needed.

### 4. Complexity Reduction

| Aspect | Before | After |
|---|---|---|
| Neighbor lookup entries | 6.46M | 344K (reused 28Ã—) |
| `paste()` calls | ~20M+ | 0 |
| Named vector lookups | ~20M+ | 0 |
| `lapply` iterations for stats | 32.3M | 0 (vectorized) |
| Estimated time | 86+ hours | **~2â€“5 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (yearly) variable values.
# Drop-in replacement for the original build_neighbor_lookup +
# compute_neighbor_stats + outer loop.
# =============================================================================

library(data.table)
# Optional but recommended for rowMaxs/rowMins/rowMeans2:
# install.packages("matrixStats") if not available
library(matrixStats)

# ---- Step 1: Build STATIC cell-level neighbor structure (done ONCE) ---------

build_cell_neighbor_matrix <- function(neighbors, max_k = NULL) {
 # neighbors: an nb object (list of integer vectors of neighbor indices)
 # Returns a matrix of size (n_cells x max_k) with neighbor indices,
 # padded with NA where a cell has fewer than max_k neighbors.
 
 n <- length(neighbors)
 if (is.null(max_k)) {
   max_k <- max(lengths(neighbors))
 }
 
 # Pre-allocate matrix
 nb_mat <- matrix(NA_integer_, nrow = n, ncol = max_k)
 for (i in seq_len(n)) {
   nb_i <- neighbors[[i]]
   # spdep nb objects use 0L to indicate no neighbors; filter those
   nb_i <- nb_i[nb_i > 0L]
   len  <- length(nb_i)
   if (len > 0L) {
     nb_mat[i, seq_len(len)] <- nb_i
   }
 }
 nb_mat
}

# ---- Step 2: Vectorized neighbor stats for one variable, one year-slice -----

compute_neighbor_stats_vectorized <- function(vals, nb_mat) {
 # vals:   numeric vector of length n_cells (one value per cell for this year)
 # nb_mat: integer matrix (n_cells x max_k) of neighbor cell indices
 # Returns: matrix (n_cells x 3) with columns [max, min, mean]
 
 # Gather neighbor values into a matrix: each row = one cell, cols = neighbors
 # vals[NA] -> NA, which is correct
 neighbor_vals <- matrix(vals[nb_mat], nrow = nrow(nb_mat), ncol = ncol(nb_mat))
 
 # Compute stats (matrixStats handles NA correctly)
 n_max  <- rowMaxs(neighbor_vals,  na.rm = TRUE)
 n_min  <- rowMins(neighbor_vals,  na.rm = TRUE)
 n_mean <- rowMeans2(neighbor_vals, na.rm = TRUE)
 
 # rowMaxs/rowMins return -Inf/Inf when all NA; convert to NA
 n_max[is.infinite(n_max)] <- NA_real_
 n_min[is.infinite(n_min)] <- NA_real_
 # rowMeans2 returns NaN when all NA; convert to NA
 n_mean[is.nan(n_mean)] <- NA_real_
 
 cbind(n_max, n_min, n_mean)
}

# ---- Step 3: Main driver â€“ replaces the entire outer loop ------------------

compute_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                          neighbor_source_vars) {
 # cell_data: data.frame/data.table with columns: id, year, and all source vars
 # id_order:  vector of cell IDs in the same order as the nb object
 # neighbors: spdep nb object (rook_neighbors_unique)
 # neighbor_source_vars: character vector of variable names
 
 # Convert to data.table for speed (non-destructive if already data.table)
 dt <- as.data.table(cell_data)
 
 # --- Static: build cell-level neighbor matrix ONCE ---
 cat("Building static cell-level neighbor matrix...\n")
 nb_mat <- build_cell_neighbor_matrix(neighbors)
 n_cells <- length(id_order)
 cat(sprintf("  %d cells, max %d neighbors per cell.\n", n_cells, ncol(nb_mat)))
 
 # Map cell IDs to their positional index in id_order (1-based)
 id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
 
 # Pre-allocate output columns in dt
 for (var_name in neighbor_source_vars) {
   col_max  <- paste0("neighbor_max_",  var_name)
   col_min  <- paste0("neighbor_min_",  var_name)
   col_mean <- paste0("neighbor_mean_", var_name)
   set(dt, j = col_max,  value = NA_real_)
   set(dt, j = col_min,  value = NA_real_)
   set(dt, j = col_mean, value = NA_real_)
 }
 
 # --- Dynamic: process each year independently ---
 years <- sort(unique(dt$year))
 cat(sprintf("Processing %d years x %d variables...\n",
             length(years), length(neighbor_source_vars)))
 
 for (yr in years) {
   # Get row indices for this year
   yr_rows <- which(dt$year == yr)
   
   # Get cell IDs for this year-slice and their positions in id_order
   yr_ids  <- dt$id[yr_rows]
   yr_pos  <- id_to_pos[as.character(yr_ids)]
   
   # Build a full-length value vector indexed by cell position.
   # This ensures nb_mat indices (which reference cell positions) work directly.
   # Cells not present in this year will remain NA.
   
   for (var_name in neighbor_source_vars) {
     # Full-length vector for all cells (most will be populated)
     full_vals <- rep(NA_real_, n_cells)
     full_vals[yr_pos] <- dt[[var_name]][yr_rows]
     
     # Vectorized neighbor stats
     stats <- compute_neighbor_stats_vectorized(full_vals, nb_mat)
     
     # Write results back only for cells present this year
     col_max  <- paste0("neighbor_max_",  var_name)
     col_min  <- paste0("neighbor_min_",  var_name)
     col_mean <- paste0("neighbor_mean_", var_name)
     
     set(dt, i = yr_rows, j = col_max,  value = stats[yr_pos, 1])
     set(dt, i = yr_rows, j = col_min,  value = stats[yr_pos, 2])
     set(dt, i = yr_rows, j = col_mean, value = stats[yr_pos, 3])
   }
   
   cat(sprintf("  Year %d done.\n", yr))
 }
 
 cat("All neighbor features computed.\n")
 
 # Return as same class as input
 if (is.data.table(cell_data)) {
   return(dt)
 } else {
   return(as.data.frame(dt))
 }
}

# =============================================================================
# USAGE â€“ drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
 cell_data,
 id_order,
 rook_neighbors_unique,
 neighbor_source_vars
)

# Now proceed directly to prediction with the pre-trained Random Forest:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

The mathematical operation is **identical**: for each cell-year observation, the neighbor max, min, and mean of each variable are computed over the same set of rook neighbors using the same values. The only change is *how* the computation is organized:

| Property | Original | Optimized |
|---|---|---|
| Neighbor relationships used | `rook_neighbors_unique` | `rook_neighbors_unique` (same) |
| Values aggregated | `data[[var_name]][neighbor_rows]` | `full_vals[nb_mat[cell_pos, ]]` (same values) |
| Aggregation functions | `max`, `min`, `mean` (ignoring NA) | `rowMaxs`, `rowMins`, `rowMeans2` (ignoring NA) â€” identical |
| Output column names | `neighbor_max_*`, `neighbor_min_*`, `neighbor_mean_*` | Same |

The pre-trained Random Forest model is **not modified or retrained** â€” it receives the same feature columns with the same values and produces the same predictions.

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| Neighbor lookups built | 6.46M (string-hashed) | 344K (integer matrix, once) |
| Core operations per year | 344K Ã— 5 vars Ã— `lapply` | 344K Ã— 5 vars Ã— vectorized matrix ops |
| Total R-level iterations | ~32.3M | ~140 (28 years Ã— 5 vars) |
| Estimated wall time | 86+ hours | **2â€“5 minutes** |
| Peak RAM | High (6.46M-entry list) | Moderate (~344K Ã— 4 int matrix + one full_vals vector) |
END_RESPONSE>>>
