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
run_id: Anthropic_F06_T07_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row lookups via `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index via a named character vector.
2. Retrieves neighbor cell IDs.
3. Pastes strings (`paste(id, year, sep="_")`) to form keys.
4. Looks up row indices via a named character vector (`idx_lookup`).

**String concatenation and named-vector lookup are extremely slow at this scale.** The `paste()` call alone creates millions of temporary character vectors, and named-vector lookup in R is O(n) in the worst case per call because R's named vectors use linear hashing with poor scaling. The result is a list of ~6.46M integer vectors, consuming significant memory and taking hours to build.

### Bottleneck B: `compute_neighbor_stats` â€” `lapply` over 6.46M elements

For each of 5 variables, the code iterates over 6.46M rows, subsets values, removes NAs, and computes `max`, `min`, `mean`. The `lapply` + `do.call(rbind, ...)` pattern is slow: each iteration creates a small 3-element vector, and `do.call(rbind, ...)` on 6.46M tiny vectors is catastrophically slow (repeated memory reallocation).

### Why raster focal/kernel operations don't directly apply

The comment in the prompt asks whether raster focal operations are a useful analogy. They are â€” conceptually, computing neighbor statistics over a regular grid is exactly a focal operation. However:

- The data is in **long panel format** (cell Ã— year), not a raster stack.
- The neighbor structure (`spdep::nb`) may encode an **irregular grid** or a grid with missing cells (e.g., ocean cells removed).
- Raster focal operations would require reshaping to a 3D array (row Ã— col Ã— year) and handling missing cells with a mask.

If the grid is complete and regular, `terra::focal()` would be fastest. But the safest general approach that **preserves the exact numerical estimand** is to vectorize the neighbor computation using sparse matrix multiplication, which handles arbitrary neighbor structures and exactly replicates the rook-neighbor max/min/mean.

---

## 2. Optimization Strategy

### Step 1: Replace `build_neighbor_lookup` with a sparse adjacency matrix

Build a sparse **row-adjacency matrix** W of dimension (N_rows Ã— N_rows) where N_rows â‰ˆ 6.46M. Entry W[i,j] = 1 if row j is a rook neighbor of row i **in the same year**. This is constructed once using vectorized joins â€” no `lapply`, no `paste`.

### Step 2: Compute neighbor stats via sparse matrix operations

- **Mean**: `W %*% x / rowSums(W)` â€” one sparse matrix-vector multiply per variable.
- **Max and Min**: Use a grouped operation. Since the sparse matrix encodes which rows are neighbors, we can extract neighbor values in bulk and compute grouped max/min using `data.table` or vectorized C-level operations.

### Step 3: Avoid `do.call(rbind, lapply(...))` entirely

All results are computed as dense vectors and assigned directly as columns.

### Expected speedup

| Component | Current | Optimized |
|---|---|---|
| Neighbor lookup build | ~hours (string ops) | ~30â€“60 seconds (integer join) |
| Stats per variable | ~hours (lapply) | ~10â€“30 seconds (sparse mat + grouped ops) |
| **Total for 5 vars** | **86+ hours** | **~5â€“10 minutes** |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Requirements: data.table, Matrix, spdep (for the nb object)
# 
# Inputs:
#   cell_data              â€” data.frame/data.table with columns: id, year, 
#                            and the neighbor_source_vars
#   id_order               â€” vector of cell IDs in the order matching 
#                            rook_neighbors_unique
#   rook_neighbors_unique  â€” spdep::nb object (list of integer neighbor indices)
#   neighbor_source_vars   â€” character vector of variable names
#
# Output:
#   cell_data with new columns: {var}_max, {var}_min, {var}_mean for each var
# =============================================================================

library(data.table)
library(Matrix)

optimized_neighbor_features <- function(cell_data, id_order, 
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {
  
  # --- Convert to data.table for speed; preserve original row order ----------
  dt <- as.data.table(cell_data)
  dt[, ..row_id := .I]  # preserve original row order
  
  # --- Step 1: Build edge list at the cell level ----------------------------
  # rook_neighbors_unique[[i]] gives neighbor indices (into id_order) for 

  # cell id_order[i]
  n_cells <- length(id_order)
  
  # Build cell-level edge list: (from_cell_idx, to_cell_idx) in id_order space
  from_cell <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_cell   <- unlist(rook_neighbors_unique)
  
  # Map from cell index to cell ID
  from_id <- id_order[from_cell]
  to_id   <- id_order[to_cell]
  
  # Cell-level edge table
  edges_cell <- data.table(from_id = from_id, to_id = to_id)
  
  cat("Cell-level edges:", nrow(edges_cell), "\n")
  
  # --- Step 2: Build row-level edge list by joining on year -----------------
  # We need row indices in dt. Create a lookup: (id, year) -> row index
  dt[, ..row_idx := .I]
  
  # Create lookup keyed by id and year
  lookup <- dt[, .(id, year, ..row_idx)]
  setkey(lookup, id, year)
  
  # Get unique years
  years <- sort(unique(dt$year))
  
  # For each year, expand cell edges to row edges
  # This is the key vectorized step replacing the slow lapply
  cat("Building row-level adjacency for", length(years), "years...\n")
  
  row_edges_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Rows in this year
    rows_yr <- lookup[year == yr]
    setkey(rows_yr, id)
    
    # Join edges: from_id -> from_row, to_id -> to_row (within same year)
    edge_yr <- copy(edges_cell)
    
    # Join from side
    edge_yr[rows_yr, from_row := i...row_idx, on = .(from_id = id)]
    # Join to side
    edge_yr[rows_yr, to_row := i...row_idx, on = .(to_id = id)]
    
    # Keep only edges where both from and to exist in this year
    edge_yr <- edge_yr[!is.na(from_row) & !is.na(to_row)]
    
    row_edges_list[[yi]] <- edge_yr[, .(from_row, to_row)]
  }
  
  row_edges <- rbindlist(row_edges_list)
  rm(row_edges_list)
  
  cat("Row-level edges:", nrow(row_edges), "\n")
  
  n_rows <- nrow(dt)
  
  # --- Step 3: Build sparse adjacency matrix --------------------------------
  # W[i,j] = 1 means row j is a rook neighbor of row i (same year)
  W <- sparseMatrix(
    i = row_edges$from_row,
    j = row_edges$to_row,
    x = 1,
    dims = c(n_rows, n_rows)
  )
  
  # Number of neighbors per row (for mean computation)
  n_neighbors <- as.numeric(rowSums(W))  # dense vector
  
  cat("Sparse adjacency matrix built:", n_rows, "x", n_rows, "\n")
  cat("Non-zeros:", nnzero(W), "\n")
  
  # --- Step 4: Compute neighbor stats for each variable ---------------------
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor features for:", var_name, "...\n")
    
    x <- dt[[var_name]]
    
    # --- MEAN: sparse matrix-vector multiply --------------------------------
    # Replace NA with 0 for multiplication, but track valid counts
    x_nona <- ifelse(is.na(x), 0, x)
    x_valid <- as.numeric(!is.na(x))
    
    # Sum of neighbor values (treating NA as 0)
    neighbor_sum <- as.numeric(W %*% x_nona)
    # Count of non-NA neighbors
    neighbor_count <- as.numeric(W %*% x_valid)
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- MAX and MIN: grouped operations on edge list -----------------------
    # For each "from_row", gather all neighbor values and compute max/min
    # This is vectorized via data.table grouping
    
    # Get neighbor values for all edges
    edge_vals <- data.table(
      from_row = row_edges$from_row,
      val      = x[row_edges$to_row]
    )
    
    # Remove edges where neighbor value is NA
    edge_vals_valid <- edge_vals[!is.na(val)]
    
    # Grouped max and min
    if (nrow(edge_vals_valid) > 0) {
      agg <- edge_vals_valid[, .(
        vmax = max(val),
        vmin = min(val)
      ), by = from_row]
      
      # Initialize with NA
      neighbor_max <- rep(NA_real_, n_rows)
      neighbor_min <- rep(NA_real_, n_rows)
      
      neighbor_max[agg$from_row] <- agg$vmax
      neighbor_min[agg$from_row] <- agg$vmin
    } else {
      neighbor_max <- rep(NA_real_, n_rows)
      neighbor_min <- rep(NA_real_, n_rows)
    }
    
    # --- Assign to data.table -----------------------------------------------
    set(dt, j = paste0(var_name, "_max"),  value = neighbor_max)
    set(dt, j = paste0(var_name, "_min"),  value = neighbor_min)
    set(dt, j = paste0(var_name, "_mean"), value = neighbor_mean)
    
    cat("  Done:", var_name, "\n")
  }
  
  # --- Clean up helper columns and return -----------------------------------
  dt[, c("..row_id", "..row_idx") := NULL]
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# =============================================================================
# USAGE
# =============================================================================
# 
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# 
# cell_data <- optimized_neighbor_features(
#   cell_data, 
#   id_order, 
#   rook_neighbors_unique, 
#   neighbor_source_vars
# )
#
# # Then predict with the pre-trained Random Forest (unchanged):
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
# =============================================================================
```

---

## 4. Why This Preserves the Exact Numerical Estimand

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **max** | `max(vals[idx])` after removing NAs | `data.table` grouped `max(val)` after removing NA edges | Identical â€” same values, same `max()` function |
| **min** | `min(vals[idx])` after removing NAs | `data.table` grouped `min(val)` after removing NA edges | Identical |
| **mean** | `mean(vals[idx])` after removing NAs | `sum(non-NA vals) / count(non-NA vals)` via sparse matrix | Identical â€” algebraically equivalent to `mean()` on non-NA values |
| **NA handling** | Returns `c(NA,NA,NA)` when no valid neighbors | `neighbor_count == 0 â†’ NA`; no entry in grouped agg â†’ `NA` | Identical |

The trained Random Forest model is never modified â€” only the feature-engineering step is optimized. The model receives exactly the same predictor columns with the same values, so predictions are bit-for-bit identical.

---

## 5. Memory Considerations (16 GB Laptop)

| Object | Estimated Size |
|---|---|
| `cell_data` (~6.46M Ã— 110 cols, numeric) | ~5.7 GB |
| Sparse matrix W (6.46M Ã— 6.46M, ~180M non-zeros) | ~2.2 GB |
| `row_edges` data.table (~180M Ã— 2 integer cols) | ~1.4 GB |
| `edge_vals` temporary per variable | ~1.4 GB (freed each iteration) |
| **Peak** | **~12â€“14 GB** âœ“ fits in 16 GB |

If memory is tight, the `edge_vals` temporary can be computed year-by-year and aggregated, or `row_edges` can be discarded after building `W` (and max/min computed from `W` directly using `summary(W)`). But the above should fit.
END_RESPONSE>>>
