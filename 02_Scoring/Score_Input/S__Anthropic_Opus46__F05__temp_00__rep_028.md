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
run_id: Anthropic_F05_T00_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` â€” **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **allocates a new character vector per row** (~4 neighbor keys Ã— 6.46M rows â‰ˆ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` â€” **named-vector lookup via string matching on a 6.46M-element named vector**, repeated 6.46M times.

The string-key lookup on a named vector of length 6.46M is approximately O(N) per probe (R's named vector lookup is hash-based, but the hash table was built over 6.46M entries and is probed ~25.8M times total). The real cost is the **per-row R-level loop overhead** and **millions of small string allocations**.

### The Broader Pattern

The same neighbor topology is **identical across all 28 years** â€” cell `i`'s rook neighbors don't change over time. Yet the code re-discovers the spatial relationship for every cell-year row. This is the fundamental redundancy: **spatial structure is conflated with the panel (space Ã— time) structure**.

Then `compute_neighbor_stats` is called 5 times, each time iterating over the 6.46M-element `neighbor_lookup` list. This is acceptable if the lookup is precomputed, but the lookup itself is the bottleneck.

## Optimization Strategy

**Principle: Separate space from time. Vectorize everything.**

1. **Build the neighbor lookup in spatial-ID space only** (344K cells, not 6.46M cell-years). Each cell has a fixed set of neighbor cell IDs. This is a one-time operation on 344K entries.

2. **For each variable and each year, use vectorized matrix indexing** to gather neighbor values and compute stats. No per-row R loop. No string keys.

3. **Use `data.table` for fast split-apply-combine** by year, with integer indexing into a spatial-ID lookup.

This reduces the algorithmic complexity from ~6.46M Ã— (string ops + hash probes) to ~28 Ã— (vectorized operations on 344K cells).

**Expected speedup**: From 86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 1: Build spatial-only neighbor lookup (once, 344K cells)
# =============================================================================
# rook_neighbors_unique: spdep nb object, indexed by position in id_order
# id_order: vector of 344,208 cell IDs in the order matching the nb object

build_spatial_neighbor_lookup <- function(id_order, nb_obj) {
  # Returns a list of length length(id_order).
  # Element i contains the integer positions (in id_order) of cell i's neighbors.
  # This is essentially what the nb object already is, but we make it explicit.
  #
  # spdep nb objects already store neighbor indices as integer vectors
  # referencing positions in the original spatial object (= id_order here).
  # We just need to ensure no zero-length entries cause issues.
  
  n <- length(id_order)
  stopifnot(length(nb_obj) == n)
  
  # nb objects store integer indices; 0L means no neighbors in spdep convention
  lapply(nb_obj, function(x) {
    x <- as.integer(x)
    x[x != 0L]
  })
}

spatial_nb <- build_spatial_neighbor_lookup(id_order, rook_neighbors_unique)

# =============================================================================
# STEP 2: Convert to data.table and create spatial index
# =============================================================================
dt <- as.data.table(cell_data)

# Create a mapping from cell ID -> position in id_order (spatial index)
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# Add spatial index to data
dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Verify
stopifnot(!anyNA(dt$spatial_idx))

# =============================================================================
# STEP 3: Vectorized neighbor stats computation
# =============================================================================
compute_all_neighbor_features <- function(dt, id_order, spatial_nb, var_names) {
  # Strategy:
  # For each year, we have up to 344,208 cells.
  # We build a value vector indexed by spatial position, then use the
  # neighbor list to gather values and compute stats â€” all vectorized.
  
  n_spatial <- length(id_order)
  years <- sort(unique(dt$year))
  
  # Pre-allocate result columns
  for (var_name in var_names) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }
  
  # Precompute neighbor lengths and a flat (row, col) structure for matrix ops
  # For each spatial cell, we know its neighbors. We build CSR-like vectors:
  nb_lengths <- vapply(spatial_nb, length, integer(1))  # length 344,208
  max_nb     <- max(nb_lengths)
  
  # Build a padded neighbor matrix: n_spatial x max_nb
  # Pad with NA so we can do matrix indexing
  nb_matrix <- matrix(NA_integer_, nrow = n_spatial, ncol = max_nb)
  for (i in seq_len(n_spatial)) {
    nbs <- spatial_nb[[i]]
    if (length(nbs) > 0L) {
      nb_matrix[i, seq_along(nbs)] <- nbs
    }
  }
  # nb_matrix[i, j] = spatial index of the j-th neighbor of cell i, or NA
  
  # For each year, fill a spatial-indexed value vector, then compute stats
  setkey(dt, year, spatial_idx)
  
  for (yr in years) {
    # Subset rows for this year
    yr_rows <- dt[.(yr)]  # keyed lookup
    
    # Build spatial_idx -> row index in yr_rows
    # (not all 344K cells may be present every year)
    yr_spatial_idx <- yr_rows$spatial_idx
    
    # Value vector indexed by spatial position (NA for missing cells)
    for (var_name in var_names) {
      val_vec <- rep(NA_real_, n_spatial)
      val_vec[yr_spatial_idx] <- yr_rows[[var_name]]
      
      # Gather neighbor values using the padded neighbor matrix
      # nb_matrix is n_spatial x max_nb; index into val_vec
      neighbor_vals_mat <- matrix(val_vec[nb_matrix], 
                                  nrow = n_spatial, ncol = max_nb)
      # neighbor_vals_mat[i, j] = value of j-th neighbor of cell i, or NA
      
      # Compute row-wise stats (only for cells present this year)
      # Use matrixStats if available for speed, otherwise base R
      present <- yr_spatial_idx  # spatial indices of cells present this year
      
      sub_mat <- neighbor_vals_mat[present, , drop = FALSE]
      
      # rowMins, rowMaxs, rowMeans ignoring NA
      # Base R approach (no extra dependency):
      row_max  <- apply(sub_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x)
      })
      row_min  <- apply(sub_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else min(x)
      })
      row_mean <- apply(sub_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else mean(x)
      })
      
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Write back into dt for this year's rows
      # We need the actual row indices in dt
      dt_row_idx <- which(dt$year == yr)
      # These are in the same order as yr_spatial_idx because of the key
      dt[dt_row_idx, (col_max)  := row_max]
      dt[dt_row_idx, (col_min)  := row_min]
      dt[dt_row_idx, (col_mean) := row_mean]
    }
    
    message(sprintf("Year %d complete.", yr))
  }
  
  return(dt)
}

# =============================================================================
# STEP 3b: Even faster version using matrixStats (recommended)
# =============================================================================
compute_all_neighbor_features_fast <- function(dt, id_order, spatial_nb, var_names) {
  
  if (!requireNamespace("matrixStats", quietly = TRUE)) {
    stop("Install matrixStats for the fast path: install.packages('matrixStats')")
  }
  
  n_spatial <- length(id_order)
  years <- sort(unique(dt$year))
  
  # Pre-allocate result columns
  for (var_name in var_names) {
    dt[, paste0("neighbor_max_", var_name)  := NA_real_]
    dt[, paste0("neighbor_min_", var_name)  := NA_real_]
    dt[, paste0("neighbor_mean_", var_name) := NA_real_]
  }
  
  # Build padded neighbor matrix (one-time cost)
  nb_lengths <- vapply(spatial_nb, length, integer(1))
  max_nb <- max(nb_lengths)
  
  nb_matrix <- matrix(NA_integer_, nrow = n_spatial, ncol = max_nb)
  for (i in seq_len(n_spatial)) {
    nbs <- spatial_nb[[i]]
    if (length(nbs) > 0L) {
      nb_matrix[i, seq_along(nbs)] <- nbs
    }
  }
  
  # Create a row-index column for fast assignment
  dt[, .row_id := .I]
  setkey(dt, year)
  
  for (yr in years) {
    # Get row indices in dt for this year
    dt_idx <- dt[.(yr), which = TRUE]
    yr_spatial <- dt$spatial_idx[dt_idx]
    
    for (var_name in var_names) {
      # Build spatial value vector
      val_vec <- rep(NA_real_, n_spatial)
      val_vec[yr_spatial] <- dt[[var_name]][dt_idx]
      
      # Gather neighbor values: only for present cells
      sub_nb <- nb_matrix[yr_spatial, , drop = FALSE]
      sub_vals <- matrix(val_vec[sub_nb], nrow = length(yr_spatial), ncol = max_nb)
      
      # matrixStats handles NA natively and is C-optimized
      r_max  <- matrixStats::rowMaxs(sub_vals,  na.rm = TRUE)
      r_min  <- matrixStats::rowMins(sub_vals,  na.rm = TRUE)
      r_mean <- matrixStats::rowMeans2(sub_vals, na.rm = TRUE)
      
      # matrixStats returns -Inf/Inf/NaN when all NA; fix to NA
      all_na <- matrixStats::rowAlls(is.na(sub_vals))
      r_max[all_na]  <- NA_real_
      r_min[all_na]  <- NA_real_
      r_mean[all_na] <- NA_real_
      
      set(dt, i = dt_idx, j = paste0("neighbor_max_", var_name),  value = r_max)
      set(dt, i = dt_idx, j = paste0("neighbor_min_", var_name),  value = r_min)
      set(dt, i = dt_idx, j = paste0("neighbor_mean_", var_name), value = r_mean)
    }
    
    message(sprintf("Year %d complete.", yr))
  }
  
  dt[, .row_id := NULL]
  return(dt)
}

# =============================================================================
# STEP 4: Execute
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data_dt <- compute_all_neighbor_features_fast(
  dt          = as.data.table(cell_data),
  id_order    = id_order,
  spatial_nb  = spatial_nb,
  var_names   = neighbor_source_vars
)

# Convert back to data.frame if needed downstream
cell_data <- as.data.frame(cell_data_dt)

# The trained Random Forest model is untouched â€” 
# only the feature-construction pipeline has been optimized.
# The numerical estimand is preserved: same max, min, mean over 
# the same rook-neighbor sets per cell-year.
```

## Complexity Comparison

| Aspect | Original | Optimized |
|---|---|---|
| **Loop iterations** | 6.46M (R-level `lapply`) | 28 years Ã— 5 vars = 140 vectorized passes |
| **String allocations** | ~25.8M `paste()` calls + 6.46M named-vector probes | **Zero** |
| **Lookup structure** | 6.46M-element named character vector | Integer matrix indexing |
| **Neighbor discovery** | Per cell-year row (redundant across years) | Per cell (spatial only, once) |
| **Stats computation** | 5 Ã— 6.46M list iterations | 140 Ã— vectorized `matrixStats` on ~344K rows |
| **Memory** | 6.46M-element list of integer vectors | One `344K Ã— max_nb` integer matrix (~5â€“11 MB) |
| **Estimated time** | 86+ hours | **~5â€“15 minutes** |

## Summary

The string-key construction was the visible symptom, but the root cause was **treating a space Ã— time panel as flat and re-discovering spatial topology per row via string hashing**. The fix separates the spatial neighbor structure (which is time-invariant) from the temporal panel, builds a padded integer neighbor matrix once, and uses vectorized matrix indexing + `matrixStats` to compute all neighbor statistics in bulk per year. The trained Random Forest model and the numerical estimand (max, min, mean of rook-neighbor values per cell-year) are exactly preserved.
END_RESPONSE>>>
