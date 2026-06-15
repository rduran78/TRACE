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
run_id: Anthropic_F06_T10_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (by cell ID).
2. Constructs character-paste keys (`"id_year"`) for each neighbor-year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`), which are hash-table lookups on character keys repeated millions of times.

The character-paste and named-vector lookup inside a per-row `lapply` is extremely expensive. With ~4 rook neighbors per cell on average and 6.46M rows, this is tens of millions of string concatenations and hash lookups.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated R-level loops

For each of the 5 source variables, another `lapply` iterates over 6.46M rows, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, `mean`. This is called 5 times. The subsetting itself is fast, but 5 Ã— 6.46M R-level function calls (`lapply` iterations) with per-element anonymous functions accumulate substantial overhead.

### Why raster focal / kernel operations are **not** appropriate here

Focal/kernel operations assume a regular complete grid with a fixed rectangular neighborhood. Here the panel is irregular (not all cells are present in all years, neighbor structure is from an `spdep::nb` object which may reflect irregular geometries/boundaries), and the data is in long panel format. Forcing it into raster layers would risk altering the numerical results (boundary cells, missing cells, non-rectangular grids). We must **preserve the exact estimand**, so we stay with the sparse neighbor structure but make it fast.

### Root cause summary

| Component | Calls | Per-call cost | Total |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M | `paste` + hash lookup | ~hours |
| `compute_neighbor_stats` | 5 Ã— 6.46M | R-level subsetting + stats | ~hours |

---

## Optimization Strategy

### Strategy 1: Vectorized integer-arithmetic neighbor lookup (eliminate all `paste` and character hashing)

Instead of character keys, we assign each cell ID a dense integer index (1..344,208) and use the fact that years are contiguous (1992â€“2019, 28 years). A cell-year row's position can be computed arithmetically:

```
row_index = (cell_index - 1) * 28 + (year - 1991)
```

This requires sorting the data by `(id, year)` and assumes complete panel (or handling gaps). The neighbor lookup becomes pure integer arithmetic â€” no strings, no hashing.

### Strategy 2: Sparse matrix multiplication for mean; vectorized group operations for max/min

- **Mean**: Construct a sparse row-normalized neighbor weight matrix **W** (344,208 Ã— 344,208) per the rook structure, then expand it to the panel dimension (or, more efficiently, compute per-year). Then `neighbor_mean = W %*% x` is a single sparse matrix-vector multiply per year per variable. 28 years Ã— 5 variables = 140 sparse matrix-vector multiplies â€” each takes milliseconds.
- **Max and Min**: Use `data.table` grouping. Expand the neighbor pairs into an edge list with year, join values, then group-by to get max/min per (cell, year). `data.table` does this in C-level optimized code.

### Strategy 3: `data.table` throughout

Replace all `data.frame` operations with `data.table` for memory efficiency and speed.

### Expected speedup

From ~86 hours â†’ **~2â€“5 minutes**.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# ==============================================================================
# 
# Requirements: data.table, Matrix, spdep (for nb object structure)
# 
# Inputs:
#   cell_data              â€” data.frame/data.table with columns: id, year, and
#                            the 5 neighbor_source_vars
#   id_order               â€” integer vector of cell IDs in the order matching
#                            rook_neighbors_unique
#   rook_neighbors_unique  â€” spdep::nb object (list of integer index vectors)
#
# Output:
#   cell_data with 15 new columns: {var}_max, {var}_min, {var}_mean
#   for each of the 5 neighbor source variables
# ==============================================================================

library(data.table)
library(Matrix)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # --------------------------------------------------------------------------
  # 0. Convert to data.table if needed
  # --------------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  cat("Cells:", n_cells, " Years:", n_years, " Rows:", nrow(cell_data), "\n")
  
  # --------------------------------------------------------------------------
  # 1. Build sparse adjacency matrix (n_cells x n_cells) from nb object
  #    and row-normalize for mean computation
  # --------------------------------------------------------------------------
  cat("Building sparse neighbor matrix...\n")
  
  # Build edge list from the nb object
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0) {
      from_list[[i]] <- rep(i, length(nb_i))
      to_list[[i]]   <- nb_i
    }
  }
  
  edge_from <- unlist(from_list)
  edge_to   <- unlist(to_list)
  
  cat("  Total directed neighbor relationships:", length(edge_from), "\n")
  
  # Binary adjacency matrix (sparse)
  W_binary <- sparseMatrix(
    i = edge_from,
    j = edge_to,
    x = 1.0,
    dims = c(n_cells, n_cells)
  )
  
  # Row-normalized version for computing means
  row_sums <- rowSums(W_binary)
  row_sums[row_sums == 0] <- 1  # avoid division by zero; those cells have no neighbors
  W_mean <- W_binary / row_sums  # Divides each row by its sum
  
  # --------------------------------------------------------------------------
  # 2. Create cell-index mapping
  # --------------------------------------------------------------------------
  # Map cell IDs to dense indices 1..n_cells
  id_to_cidx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add cell index to data
  cell_data[, cidx := id_to_cidx[as.character(id)]]
  
  # --------------------------------------------------------------------------
  # 3. Build edge-list data.table for max/min computation
  # --------------------------------------------------------------------------
  cat("Building edge list data.table for max/min...\n")
  
  # Edge list in terms of cell indices (not year-expanded yet)
  edges_dt <- data.table(from_cidx = edge_from, to_cidx = edge_to)
  
  # We need year-expanded edges. For each year, every edge connects

  # from_cidx in that year to to_cidx in that year.
  # Instead of expanding edges Ã— years (expensive in memory),
  # we join per year.
  
  # --------------------------------------------------------------------------
  # 4. Compute features per variable
  # --------------------------------------------------------------------------
  # 
  # Strategy:
  #   - MEAN: For each year, extract the variable as a vector over cells
  #           (length n_cells, with NA for missing cells), multiply by W_mean.
  #   - MAX/MIN: For each year, join edge list with values and aggregate.
  #
  # To extract per-year vectors efficiently, we need a (cidx, year) -> value 
  # lookup. We'll pivot to wide by year for the sparse-matrix approach, or 
  # use keyed data.table joins for max/min.
  # --------------------------------------------------------------------------
  
  # Key cell_data for fast lookups
  setkeyv(cell_data, c("cidx", "year"))
  
  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "\n")
    
    col_max  <- paste0(var_name, "_max")
    col_min  <- paste0(var_name, "_min")
    col_mean <- paste0(var_name, "_mean")
    
    # Initialize result columns
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
    
    # ------ MEAN via sparse matrix (per year) ------
    # ------ MAX/MIN via data.table join (per year) ------
    
    for (yr in years) {
      
      # Extract values for this year as a dense vector (length n_cells)
      year_data <- cell_data[year == yr, .(cidx, val = get(var_name))]
      
      # Dense vector for sparse matrix multiply
      val_vec <- rep(NA_real_, n_cells)
      val_vec[year_data$cidx] <- year_data$val
      
      # --- MEAN ---
      # Replace NA with 0 for matrix multiply, but we need to account for
      # the fact that some neighbors are NA. We need:
      #   mean_i = sum(val_j for j in N(i) where !is.na(val_j)) / 
      #            count(j in N(i) where !is.na(val_j))
      #
      # So we do two multiplies:
      #   numerator   = W_binary %*% val_vec_0   (sum of non-NA neighbor values)
      #   denominator = W_binary %*% valid_vec   (count of non-NA neighbors)
      
      val_vec_0 <- val_vec
      val_vec_0[is.na(val_vec_0)] <- 0
      
      valid_vec <- as.numeric(!is.na(val_vec))
      
      sum_vals   <- as.numeric(W_binary %*% val_vec_0)
      count_vals <- as.numeric(W_binary %*% valid_vec)
      
      mean_vals <- ifelse(count_vals > 0, sum_vals / count_vals, NA_real_)
      
      # Map back to cell_data rows for this year
      # Rows in cell_data for this year:
      yr_rows <- cell_data[year == yr, which = TRUE]
      yr_cidx <- cell_data$cidx[yr_rows]
      
      set(cell_data, i = yr_rows, j = col_mean, value = mean_vals[yr_cidx])
      
      # --- MAX and MIN ---
      # Join edge list with values for the neighbor (to_cidx)
      # and aggregate by from_cidx
      
      setkey(year_data, cidx)
      
      # Join: for each edge, get the neighbor's value
      edge_vals <- year_data[edges_dt, on = .(cidx = to_cidx), nomatch = NA]
      # edge_vals has columns: cidx (= to_cidx), val, from_cidx
      
      # Remove edges where neighbor value is NA
      edge_vals <- edge_vals[!is.na(val)]
      
      if (nrow(edge_vals) > 0) {
        # Aggregate by from_cidx
        agg <- edge_vals[, .(
          nb_max = max(val),
          nb_min = min(val)
        ), by = from_cidx]
        
        # Map back
        # Create lookup
        setkey(agg, from_cidx)
        
        # For the year's rows, look up from_cidx = cidx
        max_lookup <- agg[J(yr_cidx), nb_max]
        min_lookup <- agg[J(yr_cidx), nb_min]
        
        set(cell_data, i = yr_rows, j = col_max, value = max_lookup)
        set(cell_data, i = yr_rows, j = col_min, value = min_lookup)
      }
    }
    
    cat("  Done:", var_name, "\n")
  }
  
  # Clean up helper column
  cell_data[, cidx := NULL]
  
  cat("All neighbor features computed.\n")
  return(cell_data)
}

# ==============================================================================
# USAGE
# ==============================================================================
# 
# # Load your data
# cell_data <- fread("cell_data.csv")  # or however it's stored
# load("rook_neighbors_unique.rda")     # spdep::nb object
# load("id_order.rda")                  # cell ID vector
# load("trained_rf_model.rda")          # pre-trained Random Forest
# 
# # Compute neighbor features (replaces the 86-hour loop)
# cell_data <- compute_all_neighbor_features(
#   cell_data, 
#   id_order, 
#   rook_neighbors_unique
# )
# 
# # Predict with the already-trained Random Forest (unchanged)
# predictions <- predict(trained_rf_model, newdata = cell_data)
# ==============================================================================
```

---

## Why This Works and Preserves the Estimand

| Concern | Verification |
|---|---|
| **Numerical equivalence of mean** | `W_binary %*% val / W_binary %*% valid` exactly replicates `mean(neighbor_vals[!is.na(neighbor_vals)])` â€” it's the same sum divided by the same count, just vectorized. |
| **Numerical equivalence of max/min** | The `data.table` aggregation computes `max(val)` and `min(val)` grouped by the focal cell, after removing NAs â€” identical to the original `lapply` logic. |
| **NA handling** | Cells with zero valid neighbors get `NA` for all three stats, exactly as in the original. The sparse matrix mean uses `count_vals > 0` guard; the `data.table` max/min uses `nomatch = NA` and filters. |
| **No retraining** | The Random Forest model is never touched. Only the feature-computation pipeline is optimized. The new columns have identical names and values. |
| **Rook structure preserved** | We build `W_binary` directly from `rook_neighbors_unique` â€” the same `spdep::nb` object. No kernel/focal approximation is used. |

## Performance Estimate

| Operation | Estimated time |
|---|---|
| Build sparse matrix | ~2 seconds |
| Per variable, per year: sparse mat-vec (mean) | ~0.01s Ã— 28 Ã— 5 = ~1.4s |
| Per variable, per year: data.table join + agg (max/min) | ~0.3s Ã— 28 Ã— 5 = ~42s |
| **Total** | **~1â€“3 minutes** |

This represents a speedup of roughly **~2,000â€“5,000Ã—** over the original 86-hour estimate, achieved entirely through vectorization (sparse linear algebra for means, `data.table` keyed joins for max/min) while preserving exact numerical equivalence.
END_RESPONSE>>>
