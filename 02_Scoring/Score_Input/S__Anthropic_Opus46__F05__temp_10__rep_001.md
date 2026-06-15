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
run_id: Anthropic_F05_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: Per-Row String Operations at Scale

`build_neighbor_lookup` constructs the `idx_lookup` hash table once (good), but then calls `paste()` inside `lapply()` for every one of the ~6.46 million rows, each time generating string keys for that row's neighbors. With ~4 neighbors per cell on average (rook contiguity), that's roughly **26 million `paste()` calls** just to build the lookup. This is O(N Ã— k) string work where N = 6.46M and k â‰ˆ 4.

But the deeper architectural issue is that **the neighbor topology is year-invariant**. Every cell has the same neighbors in every year. The spatial neighbor structure (`rook_neighbors_unique`) is defined over the 344,208 grid cells and doesn't change across years. Yet the current code re-discovers the row indices of each cell's neighbors on a per-cell-year basis using string hashing. This means the same spatial lookup is repeated 28 times (once per year per cell).

### The Broader Pattern

1. **String-key construction** (`paste(id, year, "_")`) â€” O(N) to build the table, O(N Ã— k) inside the loop.
2. **Named-vector hash lookup** (`idx_lookup[neighbor_keys]`) â€” R's named vector lookup is O(n) per query in the worst case and has high constant overhead due to string comparison.
3. **`compute_neighbor_stats`** is efficient once `neighbor_lookup` is built, but is called 5 times (once per variable), each time iterating over 6.46M entries. This is unavoidable in a naive loop, but can be vectorized.

**Estimated cost of current approach:**
- `build_neighbor_lookup`: ~6.46M iterations Ã— (string paste + hash lookup) â‰ˆ very slow (hours).
- `compute_neighbor_stats`: 5 vars Ã— 6.46M iterations Ã— subsetting â‰ˆ slow but more tolerable.
- Total: the 86+ hour estimate is credible.

## Optimization Strategy

### Key Insight: Separate Spatial Topology from Temporal Indexing

Since the neighbor graph is constant across years, we can:

1. **Build a cell-level neighbor list once** (344K entries, not 6.46M).
2. **Build a year-to-row-offset mapping** using integer arithmetic, not string keys.
3. **Use matrix/vectorized operations** for the statistics, eliminating the per-row `lapply`.

### Specific Optimizations

| Problem | Solution | Speedup Factor |
|---|---|---|
| String `paste()` in loop | Eliminate entirely; use integer indexing | ~100Ã— |
| Per-row `lapply` over 6.46M rows | Vectorized sparse-matrix multiply | ~50-200Ã— |
| 5 separate passes over `neighbor_lookup` | Single sparse matrix, apply to all vars at once | ~5Ã— |
| Named-vector hash lookup | Direct integer row indexing | ~10Ã— |

### Algorithm

1. Sort (or index) data by `(id, year)` so that each cell's 28 years occupy a contiguous block.
2. Represent the rook neighbor graph as a **sparse adjacency matrix** over cell-year rows.
3. Compute neighbor means via sparse matrixâ€“dense matrix multiplication.
4. Compute neighbor max/min via grouped operations on the sparse structure.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites: data.table, Matrix, (optional: collapse for fast grouped ops)
# 
# Inputs:
#   cell_data            â€” data.frame/data.table with columns: id, year, and
#                          the neighbor_source_vars
#   id_order             â€” integer vector of cell IDs in the order matching
#                          rook_neighbors_unique
#   rook_neighbors_unique â€” spdep nb object (list of integer index vectors)
#   neighbor_source_vars  â€” character vector of variable names
#
# Output:
#   cell_data with new columns: {var}_neighbor_max, {var}_neighbor_min,
#                                {var}_neighbor_mean
# =============================================================================

library(data.table)
library(Matrix)

optimized_neighbor_features <- function(cell_data, id_order, 
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {
  
  # --- Step 0: Convert to data.table for efficiency, preserve original order ---
  dt <- as.data.table(cell_data)
  dt[, .original_row_order := .I]
  
  # --- Step 1: Create integer cell index and sort by (cell_idx, year) ---
  # Map cell IDs to their position in id_order (1..N_cells)
  n_cells <- length(id_order)
  id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))
  dt[, cell_idx := id_to_cellidx[as.character(id)]]
  
  # Get sorted unique years and create year-to-position mapping
  years_sorted <- sort(unique(dt$year))
  n_years <- length(years_sorted)
  year_to_yearidx <- setNames(seq_len(n_years), as.character(years_sorted))
  dt[, year_idx := year_to_yearidx[as.character(year)]]
  
  # Sort by (cell_idx, year_idx) so we can use arithmetic indexing
  setorder(dt, cell_idx, year_idx)
  
  # After sorting, row for (cell c, year t) is at position:
  #   (c - 1) * n_years + t
  # But only if the panel is perfectly balanced. Let's verify and handle both.
  
  expected_rows <- n_cells * n_years
  is_balanced <- (nrow(dt) == expected_rows)
  
  if (is_balanced) {
    message("Panel is balanced (", n_cells, " cells Ã— ", n_years, " years = ", 
            expected_rows, " rows). Using fast arithmetic indexing.")
    
    # ----- BALANCED PANEL: ARITHMETIC INDEXING (fastest) -----
    
    # Row index for cell c, year t (1-based):
    #   row(c, t) = (c - 1) * n_years + t
    
    # --- Step 2: Build sparse neighbor matrix (N_rows Ã— N_rows) ---
    # For each cell-year row i = (c, t), its neighbors are (c', t) for each
    # neighbor c' of c. We build COO triplets.
    
    message("Building sparse adjacency matrix over cell-year rows...")
    
    # Pre-calculate total number of directed neighbor pairs
    n_neighbors_per_cell <- lengths(rook_neighbors_unique)
    total_directed_pairs <- sum(n_neighbors_per_cell)  # ~1.37M
    total_entries <- total_directed_pairs * n_years     # ~1.37M Ã— 28 â‰ˆ 38.5M
    
    # Pre-allocate COO vectors
    row_i <- integer(total_entries)
    col_j <- integer(total_entries)
    
    ptr <- 0L
    for (c_idx in seq_len(n_cells)) {
      nb_indices <- rook_neighbors_unique[[c_idx]]
      if (length(nb_indices) == 0) next
      
      n_nb <- length(nb_indices)
      # For each year, cell c_idx's row is (c_idx - 1)*n_years + t
      # Its neighbor c_nb's row is (c_nb - 1)*n_years + t
      for (t in seq_len(n_years)) {
        source_row <- (c_idx - 1L) * n_years + t
        dest_rows  <- (nb_indices - 1L) * n_years + t
        
        idx_range <- ptr + seq_len(n_nb)
        row_i[idx_range] <- source_row
        col_j[idx_range] <- dest_rows
        ptr <- ptr + n_nb
      }
    }
    
    message("  Total entries in sparse matrix: ", ptr)
    
    N <- nrow(dt)
    # All values are 1 (unweighted adjacency)
    W <- sparseMatrix(
      i = row_i[1:ptr], j = col_j[1:ptr], x = rep(1, ptr),
      dims = c(N, N)
    )
    rm(row_i, col_j)
    gc()
    
    # --- Step 3: Compute neighbor stats using sparse matrix operations ---
    
    # neighbor_count per row (for computing mean)
    neighbor_count <- as.numeric(W %*% rep(1, N))
    neighbor_count[neighbor_count == 0] <- NA_real_
    
    for (var_name in neighbor_source_vars) {
      message("Computing neighbor stats for: ", var_name)
      
      vals <- dt[[var_name]]
      
      # ---- MEAN: sparse matrix multiply ----
      neighbor_sum <- as.numeric(W %*% vals)
      nb_mean <- neighbor_sum / neighbor_count
      
      # ---- MAX and MIN: iterate over sparse structure ----
      # We need to handle NAs properly
      # Use the sparse matrix structure directly
      nb_max <- rep(NA_real_, N)
      nb_min <- rep(NA_real_, N)
      
      # Extract CSC (compressed sparse column) -> convert to CSR for row access
      # Actually, dgCMatrix is CSC. We want row-wise access, so transpose:
      Wt <- t(W)  # Now Wt is CSC, and column j of Wt = row j of W
      
      # For each row i of W, the nonzero columns are the neighbors
      # In Wt (CSC), column i has entries at rows = neighbor indices of i
      p <- Wt@p  # column pointers (0-based)
      idx_vec <- Wt@i + 1L  # row indices (convert to 1-based)
      
      # Vectorized grouped max/min using data.table
      # Build a table of (source_row, neighbor_row) then join with vals
      
      # Faster approach: direct C-style loop via vapply on unique patterns
      # But even better: use the CSC structure directly
      
      message("  Computing max and min...")
      
      # Process in chunks to manage memory
      chunk_size <- 500000L
      n_chunks <- ceiling(N / chunk_size)
      
      for (ch in seq_len(n_chunks)) {
        start_i <- (ch - 1L) * chunk_size + 1L
        end_i   <- min(ch * chunk_size, N)
        
        for (i in start_i:end_i) {
          nb_start <- p[i] + 1L
          nb_end   <- p[i + 1L]
          if (nb_end < nb_start) next
          
          nb_vals <- vals[idx_vec[nb_start:nb_end]]
          nb_vals <- nb_vals[!is.na(nb_vals)]
          if (length(nb_vals) == 0) next
          
          nb_max[i] <- max(nb_vals)
          nb_min[i] <- min(nb_vals)
        }
      }
      
      # Where there are no neighbors, mean should also be NA
      nb_mean[is.na(neighbor_count)] <- NA_real_
      
      # Assign to data.table
      set(dt, j = paste0(var_name, "_neighbor_max"),  value = nb_max)
      set(dt, j = paste0(var_name, "_neighbor_min"),  value = nb_min)
      set(dt, j = paste0(var_name, "_neighbor_mean"), value = nb_mean)
    }
    
    rm(W, Wt)
    gc()
    
  } else {
    message("Panel is unbalanced (", nrow(dt), " rows vs expected ", 
            expected_rows, "). Using hash-based indexing.")
    
    # ----- UNBALANCED PANEL: FAST INTEGER HASH -----
    # Use data.table keyed join instead of string paste + named vector
    
    # Create row-index lookup keyed on (cell_idx, year_idx)
    dt[, .row_idx := .I]
    lookup_dt <- dt[, .(.row_idx, cell_idx, year_idx)]
    setkey(lookup_dt, cell_idx, year_idx)
    
    # Build neighbor lookup per cell (not per cell-year!)
    message("Building cell-level neighbor index list...")
    cell_neighbors <- lapply(seq_len(n_cells), function(c_idx) {
      rook_neighbors_unique[[c_idx]]
    })
    
    # For each cell-year row, get neighbor rows
    message("Building row-level neighbor lookup via integer join...")
    
    # Expand: for each row, list (neighbor_cell_idx, year_idx) pairs
    # Then batch-join to find row indices
    
    # Build edge list: (cell_idx, neighbor_cell_idx)
    edge_from <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
    edge_to   <- unlist(rook_neighbors_unique)
    
    # Cross with years: for each edge and each year, we have a row-pair
    # But we only need edges where both endpoints exist in the data
    
    edges_dt <- data.table(from_cell = edge_from, to_cell = edge_to)
    
    # Join with data to get (from_row, to_cell, year_idx), then join to_cell+year_idx -> to_row
    from_rows <- dt[, .(from_cell = cell_idx, year_idx, from_row = .row_idx)]
    
    # Merge edges with from_rows
    setkey(edges_dt, from_cell)
    setkey(from_rows, from_cell)
    
    expanded <- edges_dt[from_rows, on = "from_cell", allow.cartesian = TRUE, nomatch = 0L]
    # expanded has columns: from_cell, to_cell, year_idx, from_row
    
    # Now find to_row by joining (to_cell, year_idx) -> lookup_dt
    setnames(expanded, "to_cell", "cell_idx_nb")
    expanded[, cell_idx := cell_idx_nb]
    
    setkey(expanded, cell_idx, year_idx)
    expanded <- lookup_dt[expanded, on = .(cell_idx, year_idx), nomatch = NA]
    # .row_idx is now the to_row
    setnames(expanded, ".row_idx", "to_row")
    
    # Remove edges where the neighbor row doesn't exist
    expanded <- expanded[!is.na(to_row)]
    
    # Now compute neighbor stats directly from this edge table
    for (var_name in neighbor_source_vars) {
      message("Computing neighbor stats for: ", var_name)
      
      expanded[, nb_val := dt[[var_name]][to_row]]
      
      stats <- expanded[!is.na(nb_val), .(
        nb_max  = max(nb_val),
        nb_min  = min(nb_val),
        nb_mean = mean(nb_val)
      ), by = from_row]
      
      # Initialize with NAs
      set(dt, j = paste0(var_name, "_neighbor_max"),  value = rep(NA_real_, nrow(dt)))
      set(dt, j = paste0(var_name, "_neighbor_min"),  value = rep(NA_real_, nrow(dt)))
      set(dt, j = paste0(var_name, "_neighbor_mean"), value = rep(NA_real_, nrow(dt)))
      
      # Fill in computed values
      set(dt, i = stats$from_row, j = paste0(var_name, "_neighbor_max"),  value = stats$nb_max)
      set(dt, i = stats$from_row, j = paste0(var_name, "_neighbor_min"),  value = stats$nb_min)
      set(dt, i = stats$from_row, j = paste0(var_name, "_neighbor_mean"), value = stats$nb_mean)
    }
  }
  
  # --- Step 4: Restore original row order ---
  setorder(dt, .original_row_order)
  dt[, c(".original_row_order", "cell_idx", "year_idx") := NULL]
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}
```

## Even Faster: Fully Vectorized Sparse-Matrix Approach (Balanced Panel)

The loop for max/min above is still per-row. Here's a fully vectorized version that eliminates all R-level loops:

```r
# =============================================================================
# FULLY VECTORIZED VERSION (recommended for balanced panel)
# =============================================================================

optimized_neighbor_features_v2 <- function(cell_data, id_order,
                                            rook_neighbors_unique,
                                            neighbor_source_vars) {
  
  library(data.table)
  library(Matrix)
  
  dt <- as.data.table(cell_data)
  dt[, .orig_order := .I]
  
  n_cells <- length(id_order)
  id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))
  dt[, cell_idx := id_to_cellidx[as.character(id)]]
  
  years_sorted <- sort(unique(dt$year))
  n_years <- length(years_sorted)
  year_to_yearidx <- setNames(seq_len(n_years), as.character(years_sorted))
  dt[, year_idx := year_to_yearidx[as.character(year)]]
  
  setorder(dt, cell_idx, year_idx)
  N <- nrow(dt)
  
  # --- Build edge table (from_row, to_row) using vectorized integer arithmetic ---
  message("Building edge table...")
  
  n_nb_per_cell <- lengths(rook_neighbors_unique)
  from_cell <- rep(seq_len(n_cells), times = n_nb_per_cell)
  to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)
  n_edges_spatial <- length(from_cell)  # ~1.37M
  
  # Expand across years: each spatial edge becomes n_years temporal edges
  # from_row = (from_cell - 1) * n_years + t
  # to_row   = (to_cell   - 1) * n_years + t
  
  # Vectorized expansion
  from_cell_exp <- rep(from_cell, each = n_years)
  to_cell_exp   <- rep(to_cell,   each = n_years)
  year_exp      <- rep(seq_len(n_years), times = n_edges_spatial)
  
  from_row <- (from_cell_exp - 1L) * n_years + year_exp
  to_row   <- (to_cell_exp   - 1L) * n_years + year_exp
  
  rm(from_cell_exp, to_cell_exp, year_exp)
  gc()
  
  message("  Total directed cell-year edges: ", length(from_row))
  
  # --- Build edge data.table for grouped operations ---
  edges <- data.table(from_row = from_row, to_row = to_row)
  rm(from_row, to_row)
  gc()
  
  # --- Compute stats per variable ---
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)
    
    # Attach neighbor values
    edges[, nb_val := dt[[var_name]][to_row]]
    
    # Grouped aggregation (data.table is extremely fast at this)
    stats <- edges[!is.na(nb_val), .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_sum  = sum(nb_val),
      nb_n    = .N
    ), keyby = from_row]
    
    stats[, nb_mean := nb_sum / nb_n]
    
    # Initialize columns with NA
    max_col  <- rep(NA_real_, N)
    min_col  <- rep(NA_real_, N)
    mean_col <- rep(NA_real_, N)
    
    # Fill values at the correct rows
    max_col[stats$from_row]  <- stats$nb_max
    min_col[stats$from_row]  <- stats$nb_min
    mean_col[stats$from_row] <- stats$nb_mean
    
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
  }
  
  rm(edges)
  gc()
  
  # Restore original order
  setorder(dt, .orig_order)
  dt[, c(".orig_order", "cell_idx", "year_idx") := NULL]
  
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}
```

## Usage (Drop-In Replacement)

```r
# Replace the original three blocks:
#   neighbor_lookup <- build_neighbor_lookup(...)
#   for (var_name in ...) { cell_data <- compute_and_add_neighbor_features(...) }
#
# With:

cell_data <- optimized_neighbor_features_v2(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is untouched â€” only the feature construction changes.
# Column names and values are identical to the original output.
```

## Memory Estimate for v2

| Object | Size |
|---|---|
| `edges` data.table (38.5M rows Ã— 2 int + 1 double) | ~620 MB |
| `dt` (6.46M rows Ã— ~125 cols) | ~6.5 GB |
| Working room | ~2-3 GB |
| **Total peak** | **~10-12 GB** (fits in 16 GB) |

## Expected Runtime

| Step | Original | Optimized (v2) |
|---|---|---|
| Build neighbor lookup | ~40-60 hours | ~30 seconds (edge table construction) |
| Compute stats (5 vars) | ~26-30 hours | ~5-10 minutes (data.table grouped aggregation) |
| **Total** | **~86+ hours** | **~10-15 minutes** |

The speedup comes from eliminating all string operations, replacing per-row R-level loops with vectorized integer arithmetic and data.table's highly optimized C-level `gforce` grouped aggregation.
END_RESPONSE>>>
