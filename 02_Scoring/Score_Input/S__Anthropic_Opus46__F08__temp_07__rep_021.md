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
run_id: Anthropic_F08_T07_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates the static spatial topology with the time-varying data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds neighbor indices per cell-year row (~6.46M rows)** even though the neighbor *relationship* between cells is purely spatial and identical across all 28 years. The function creates ~6.46M list entries, each requiring string-pasting, hash lookups, and NA filtering â€” all of which are redundant repetitions of the same spatial logic.

2. **`compute_neighbor_stats` iterates over ~6.46M list entries** using `lapply`, performing per-element subsetting and aggregation in pure R. This is inherently slow for millions of rows.

3. **Memory bloat**: The `neighbor_lookup` list stores ~6.46M integer vectors. Since each of the 344,208 cells has on average ~4 rook neighbors, and this is replicated 28 times, the list contains ~6.46M entries Ã— ~4 integers each â‰ˆ 25.8M integers plus massive list overhead.

### The Key Insight

> **Neighbor relationships are static across years; only the variable values change by year.**

This means:
- The **topology** (which cell neighbors which cell) needs to be computed **once** for 344,208 cells, not for 6.46M cell-year rows.
- The **aggregation** (max, min, mean of neighbor values) should be done **per year** using the same static topology, leveraging vectorized/matrix operations.

---

## Optimization Strategy

### Step 1: Build a Static Neighbor Structure Once (344K cells, not 6.46M rows)
Construct a **sparse adjacency matrix** from `rook_neighbors_unique`. This is a 344,208 Ã— 344,208 sparse matrix `W` where `W[i,j] = 1` if cell `j` is a neighbor of cell `i`. This is built once and reused for all years and all variables.

### Step 2: Compute Neighbor Stats via Sparse Matrixâ€“Vector Multiplication Per Year
For each year and each variable:
- Extract the variable vector for that year (344,208 values).
- Use sparse matrix operations to compute neighbor **mean** (`W %*% x / neighbor_count`), **max**, and **min**.

For **mean**, sparse matrix multiplication is directly applicable. For **max** and **min**, we iterate over rows of the sparse matrix, but only over 344K cells (not 6.46M), and we do it in a vectorized manner using the `Matrix` package internals or `data.table` group-by operations.

### Step 3: Join Results Back to the Panel
Merge the per-cell-per-year results back into the full `cell_data` data.table.

### Expected Speedup
- Topology: 6.46M â†’ 344K (18.8Ã— reduction)
- Per-variable-year aggregation: vectorized sparse ops on 344K cells Ã— 28 years = 9.6M ops, but each is vectorized, not R-level lapply.
- Estimated runtime: **minutes, not hours**.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build the sparse adjacency matrix ONCE from the static nb object
# ==============================================================================
build_sparse_adjacency <- function(neighbors, id_order) {
  # neighbors: spdep nb object (list of integer index vectors)
  # id_order:  vector of cell IDs in the order used by the nb object
  n <- length(id_order)
  stopifnot(length(neighbors) == n)
  
  # Build COO triplets
  from <- integer(0)
  to   <- integer(0)
  
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0L) {
      from <- c(from, rep.int(i, length(nb_i)))
      to   <- c(to,   nb_i)
    }
  }
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  
  # Precompute the number of neighbors per cell (for mean calculation)
  neighbor_count <- diff(W@p)  # CSC column counts if transposed; use rowSums
  neighbor_count <- rowSums(W)
  
  list(W = W, neighbor_count = neighbor_count, id_order = id_order, n = n)
}

# ==============================================================================
# STEP 2: Compute neighbor max, min, mean for one variable across all years
#          using the static adjacency
# ==============================================================================
compute_neighbor_features_fast <- function(cell_dt, var_name, adj) {
  # cell_dt:  data.table with columns: id, year, <var_name>
  # adj:      output of build_sparse_adjacency
  
  W            <- adj$W
  neighbor_cnt <- adj$neighbor_count
  id_order     <- adj$id_order
  n            <- adj$n
  
  # Map cell IDs to row indices in the adjacency matrix
  id_to_idx <- setNames(seq_len(n), as.character(id_order))
  
  # Output column names (must match original pipeline)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Ensure data.table
  if (!is.data.table(cell_dt)) cell_dt <- as.data.table(cell_dt)
  
  # Add adjacency index for each row
  cell_dt[, adj_idx__ := id_to_idx[as.character(id)]]
  
  # Pre-extract the CSR structure of W for row-wise max/min

  # Convert W to dgRMatrix (row-compressed) for efficient row access
  W_csr <- as(W, "RsparseMatrix")
  
  years <- sort(unique(cell_dt$year))
  
  # Preallocate output columns
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]
  
  for (yr in years) {
    # Row indices in cell_dt for this year
    yr_mask <- cell_dt$year == yr
    yr_rows <- which(yr_mask)
    
    if (length(yr_rows) == 0L) next
    
    # Build a full-length vector aligned to adjacency matrix rows
    # (some cells may be missing in a given year; they get NA)
    x_full <- rep(NA_real_, n)
    x_full[cell_dt$adj_idx__[yr_rows]] <- cell_dt[[var_name]][yr_rows]
    
    # --- Neighbor MEAN via sparse matrix multiplication ---
    # Replace NA with 0 for multiplication, track valid counts
    x_notna   <- !is.na(x_full)
    x_zero    <- ifelse(x_notna, x_full, 0)
    
    sum_vals   <- as.numeric(W %*% x_zero)          # sum of neighbor values
    count_vals <- as.numeric(W %*% as.numeric(x_notna))  # count of non-NA neighbors
    
    n_mean <- ifelse(count_vals > 0, sum_vals / count_vals, NA_real_)
    
    # --- Neighbor MAX and MIN via CSR row traversal ---
    n_max <- rep(NA_real_, n)
    n_min <- rep(NA_real_, n)
    
    # Extract CSR components
    row_ptr <- W_csr@p    # length n+1
    col_idx <- W_csr@j    # 0-based column indices
    
    for (i in seq_len(n)) {
      start <- row_ptr[i] + 1L   # convert 0-based to 1-based
      end   <- row_ptr[i + 1L]
      if (end < start) next       # no neighbors
      
      nb_cols <- col_idx[start:end] + 1L  # 1-based
      nb_vals <- x_full[nb_cols]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      
      if (length(nb_vals) > 0L) {
        n_max[i] <- max(nb_vals)
        n_min[i] <- min(nb_vals)
      }
    }
    
    # Write results back to cell_dt for this year's rows
    adj_indices_yr <- cell_dt$adj_idx__[yr_rows]
    set(cell_dt, i = yr_rows, j = col_max,  value = n_max[adj_indices_yr])
    set(cell_dt, i = yr_rows, j = col_min,  value = n_min[adj_indices_yr])
    set(cell_dt, i = yr_rows, j = col_mean, value = n_mean[adj_indices_yr])
  }
  
  # Clean up temp column
  cell_dt[, adj_idx__ := NULL]
  
  return(cell_dt)
}

# ==============================================================================
# STEP 2b: Fully vectorized max/min using data.table (avoids R-level for loop 
#          over 344K cells). This replaces the inner for-loop over cells.
# ==============================================================================
compute_neighbor_features_dt <- function(cell_dt, var_name, adj) {
  W            <- adj$W
  neighbor_cnt <- adj$neighbor_count
  id_order     <- adj$id_order
  n            <- adj$n
  
  id_to_idx <- setNames(seq_len(n), as.character(id_order))
  
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  if (!is.data.table(cell_dt)) cell_dt <- as.data.table(cell_dt)
  
  # Build edge list from sparse matrix (once, static)
  W_coo <- summary(W)  # returns data.frame with i, j, x
  edge_dt <- data.table(from = W_coo$i, to = W_coo$j)
  
  # Add adjacency index to cell_dt
  cell_dt[, adj_idx__ := id_to_idx[as.character(id)]]
  
  # Create a lookup: (adj_idx, year) -> variable value
  lookup <- cell_dt[, .(adj_idx__ = adj_idx__, year = year, val = get(var_name))]
  setkey(lookup, adj_idx__, year)
  
  years <- sort(unique(cell_dt$year))
  
  # Preallocate
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]
  
  for (yr in years) {
    # Get values for this year, indexed by adj_idx
    yr_lookup <- lookup[year == yr, .(adj_idx__, val)]
    setkey(yr_lookup, adj_idx__)
    
    # Join neighbor values: for each edge (from -> to), get val of 'to' in this year
    edge_vals <- merge(edge_dt, yr_lookup, by.x = "to", by.y = "adj_idx__",
                       all.x = FALSE, allow.cartesian = FALSE)
    # edge_vals has columns: to, from, val
    # 'from' is the focal cell, 'to' is the neighbor, 'val' is the neighbor's value
    
    # Remove NA values
    edge_vals <- edge_vals[!is.na(val)]
    
    # Aggregate by focal cell (from)
    agg <- edge_vals[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = from]
    
    setkey(agg, from)
    
    # Write back to cell_dt
    yr_rows <- which(cell_dt$year == yr)
    adj_indices_yr <- cell_dt$adj_idx__[yr_rows]
    
    matched <- agg[J(adj_indices_yr)]
    
    set(cell_dt, i = yr_rows, j = col_max,  value = matched$nb_max)
    set(cell_dt, i = yr_rows, j = col_min,  value = matched$nb_min)
    set(cell_dt, i = yr_rows, j = col_mean, value = matched$nb_mean)
  }
  
  cell_dt[, adj_idx__ := NULL]
  return(cell_dt)
}

# ==============================================================================
# MAIN PIPELINE (drop-in replacement)
# ==============================================================================

# --- 1. Build static adjacency ONCE ---
adj <- build_sparse_adjacency(rook_neighbors_unique, id_order)

# --- 2. Convert to data.table if not already ---
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# --- 3. Compute neighbor features for each variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data <- compute_neighbor_features_dt(cell_data, var_name, adj)
}

# --- 4. Predict with pre-trained Random Forest (unchanged) ---
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Topology computation** | Per cell-year row (6.46M entries) | Once for 344K cells (sparse matrix) |
| **Neighbor lookup structure** | R list of 6.46M integer vectors | Sparse matrix + edge `data.table` (~1.37M edges) |
| **Aggregation method** | `lapply` over 6.46M entries, pure R | `data.table` grouped aggregation per year (vectorized C) |
| **Mean computation** | Element-wise in R loop | Sparse matrix multiplication (BLAS) |
| **Memory footprint** | ~6.46M list entries + overhead | ~1.37M edge rows + 344K sparse matrix |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |
| **Numerical equivalence** | â€” | âœ… Identical max, min, mean values |
| **RF model** | â€” | âœ… Untouched; not retrained |

### Why `compute_neighbor_features_dt` is preferred over the `_fast` variant

The `_fast` version still contains an R-level `for` loop over 344K cells for max/min. The `_dt` version replaces this with a single `data.table` grouped aggregation over the ~1.37M edge list, which runs entirely in C and is dramatically faster. Both produce identical results; the `_dt` version is the recommended production path.
END_RESPONSE>>>
