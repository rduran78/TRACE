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
run_id: Anthropic_F10_T10_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three major bottlenecks:

1. **O(nÂ²)-scale neighbor lookup construction**: `build_neighbor_lookup` iterates over all ~6.46M rows, and for each row it performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector indexing in R with millions of entries is extremely slow because R's named vector lookup is essentially a linear-hash probe on a `STRSXP` â€” repeated ~6.46M times with multiple key lookups per call.

2. **Row-wise `lapply` over 6.46M rows in `compute_neighbor_stats`**: Even though each iteration is cheap, the overhead of 6.46M R function calls, each allocating small vectors, is enormous. This runs 5 times (once per variable), producing ~32.3M R function calls total.

3. **No separation of topology from time**: The rook-neighbor graph is purely spatial (cell-to-cell). It is identical for every year. Yet the current code re-entangles topology with time by creating a single monolithic lookup over all 6.46M cell-year rows. This inflates every data structure by a factor of 28.

**Summary**: The 86+ hour runtime is dominated by R-level per-row iteration and string-based indexing across millions of rows, applied redundantly across years.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The rook-neighbor graph has **344,208 nodes** and **~1.37M directed edges** â€” it is independent of year. We should:

1. **Build the adjacency structure once** at the cell level (344K cells, not 6.46M cell-years).
2. **Convert to a sparse matrix** (`dgCMatrix`) â€” this allows vectorized sparse matrixâ€“vector multiplication for computing neighbor sums and counts in one shot.
3. **Process each year as a slice**: For each year, extract the variable column as a vector aligned with the cell order, then use sparse matrix operations to compute neighbor aggregates in bulk. `A %*% x` gives neighbor sums; use a binary version of A against indicator vectors for counts; use iterative max/min via the explicit sparse structure.

### Specific Techniques

| Bottleneck | Solution |
|---|---|
| String-key lookups in `build_neighbor_lookup` | Build a sparse adjacency matrix `A` (344K Ã— 344K) once from `rook_neighbors_unique` using integer indexing |
| Per-row `lapply` in `compute_neighbor_stats` | **Mean**: `A %*% x / A %*% 1` (sparse mat-vec). **Min/Max**: Use `data.table` grouped operations over an edge list, or a C++-level rowwise sparse extrema function |
| Redundant 28Ã— blowup | Process per-year slices; the adjacency matrix is reused |
| Memory (16GB constraint) | Only one year-slice (~344K rows) is in memory at a time for computation |

### Numerical Equivalence

- The sparse matrix `A` encodes exactly the same directed edges as `rook_neighbors_unique`.
- `A %*% x` is an exact sum of neighbor values (IEEE 754 addition in the same accumulation order as `sum()`).
- Mean = sum / count, min and max are computed over exactly the same neighbor sets.
- The resulting 15 feature columns (5 vars Ã— 3 stats) are numerically identical to the original.

### Complexity Reduction

| | Original | Optimized |
|---|---|---|
| Lookup build | ~6.46M string ops | ~1.37M integer inserts (once) |
| Stats per variable | ~6.46M R function calls | 28 sparse mat-vec ops + 28 grouped edge-list ops |
| Total R function calls | ~32.3M | ~280 (5 vars Ã— 28 years Ã— 2 ops) |
| Expected runtime | 86+ hours | **~2â€“10 minutes** |

---

## Optimized R Code

```r
###############################################################################
# OPTIMIZED SPATIAL NEIGHBOR FEATURE ENGINEERING
# 
# Preserves numerical equivalence with original compute_neighbor_stats output.
# Preserves the trained Random Forest model (no retraining).
# Designed for 16 GB RAM laptop.
###############################################################################

library(data.table)
library(Matrix)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Prepare ordered cell IDs and year structure
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table for speed (non-destructive; preserves all columns)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# id_order: the canonical ordering of cell IDs (length = 344,208)
# This must match the indexing of rook_neighbors_unique (spdep::nb object).
# id_order[i] is the cell_id for the i-th element of rook_neighbors_unique.
n_cells <- length(id_order)

# Create integer mapping: cell_id -> position in id_order (1-based)
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build sparse adjacency matrix ONCE from rook_neighbors_unique
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique is an spdep::nb object: a list of length n_cells,
# where rook_neighbors_unique[[i]] is an integer vector of neighbor indices
# (referring to positions in id_order). A value of 0L means no neighbors.

build_adjacency <- function(nb_obj, n) {
  # Build COO (coordinate) representation of directed adjacency
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) > 0) {
      from_list[[i]] <- rep.int(i, length(nbrs))
      to_list[[i]]   <- nbrs
    }
  }
  
  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list, use.names = FALSE)
  
  # Sparse matrix: A[i, j] = 1 means j is a neighbor of i
  # So A %*% x gives, for each row i, the sum of x over i's neighbors
  sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n, n),
    repr = "C"   # CSC format, efficient for mat-vec
  )
}

cat("Building sparse adjacency matrix (344K x 344K)...\n")
A <- build_adjacency(rook_neighbors_unique, n_cells)

# Precompute neighbor count per cell (used for mean calculation)
ones_vec     <- rep(1, n_cells)
neighbor_cnt <- as.numeric(A %*% ones_vec)  # length n_cells

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Build edge list for min/max (grouped operations)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Extract COO from sparse matrix for edge-list based min/max

A_T <- summary(A)  # returns data.frame with columns i, j, x
edge_dt <- data.table(
  from = A_T$i,   # the node whose feature we're computing
  to   = A_T$j    # the neighbor whose attribute we read
)
setkey(edge_dt, from)

cat(sprintf("Adjacency: %d cells, %d directed edges\n", n_cells, nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Ensure cell_data is ordered by (id_pos, year) for fast slicing
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Map each cell_id to its position in id_order
cell_data[, id_pos := id_to_pos[as.character(id)]]

# Sort by year and id_pos for efficient year-slicing
setkey(cell_data, year, id_pos)

years <- sort(unique(cell_data$year))
n_years <- length(years)

cat(sprintf("Processing %d years x %d cells = %d cell-years\n",
            n_years, n_cells, nrow(cell_data)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Compute neighbor stats (max, min, mean) per variable per year
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

# For each year, extract the value vector aligned to id_order positions,
# then compute stats using sparse matrix ops + edge-list grouped ops.

compute_neighbor_features_for_year <- function(dt_year, var_name, A, 
                                                neighbor_cnt, edge_dt, n_cells) {
  # dt_year is keyed by id_pos and contains exactly n_cells rows for this year
  # Extract values in id_order alignment
  x <- rep(NA_real_, n_cells)
  x[dt_year$id_pos] <- dt_year[[var_name]]
  
  # --- MEAN via sparse matrix ---
  # Handle NAs: we need sum of non-NA neighbors and count of non-NA neighbors
  x_nona <- x
  x_nona[is.na(x_nona)] <- 0
  
  is_valid <- as.numeric(!is.na(x))  # 1 if not NA, 0 if NA
  
  neighbor_sum     <- as.numeric(A %*% x_nona)      # sum of non-NA neighbor values
  neighbor_nvalid  <- as.numeric(A %*% is_valid)     # count of non-NA neighbors
  
  n_mean <- ifelse(neighbor_nvalid > 0, neighbor_sum / neighbor_nvalid, NA_real_)
  
  # --- MIN and MAX via edge-list grouped operation ---
  # Look up neighbor values
  edge_vals <- x[edge_dt$to]
  
  # Grouped min/max, excluding NAs
  tmp <- data.table(from = edge_dt$from, val = edge_vals)
  
  # Remove edges where neighbor value is NA
  tmp <- tmp[!is.na(val)]
  
  if (nrow(tmp) > 0) {
    agg <- tmp[, .(nmax = max(val), nmin = min(val)), by = from]
    
    n_max <- rep(NA_real_, n_cells)
    n_min <- rep(NA_real_, n_cells)
    n_max[agg$from] <- agg$nmax
    n_min[agg$from] <- agg$nmin
  } else {
    n_max <- rep(NA_real_, n_cells)
    n_min <- rep(NA_real_, n_cells)
  }
  
  # Cells with no neighbors at all also get NA (neighbor_cnt == 0)
  no_neighbors <- (neighbor_cnt == 0)
  n_mean[no_neighbors] <- NA_real_
  n_max[no_neighbors]  <- NA_real_
  n_min[no_neighbors]  <- NA_real_
  
  list(n_max = n_max, n_min = n_min, n_mean = n_mean)
}

cat("Computing neighbor features...\n")
t_start <- Sys.time()

for (yr in years) {
  cat(sprintf("  Year %d ...\n", yr))
  
  # Get row indices for this year (cell_data is keyed by year, id_pos)
  yr_rows <- which(cell_data$year == yr)
  dt_year <- cell_data[yr_rows]
  
  for (var_name in neighbor_source_vars) {
    stats <- compute_neighbor_features_for_year(
      dt_year, var_name, A, neighbor_cnt, edge_dt, n_cells
    )
    
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    
    # Write back: dt_year$id_pos maps each row to the position in the stats vectors
    set(cell_data, i = yr_rows, j = col_max,  value = stats$n_max[dt_year$id_pos])
    set(cell_data, i = yr_rows, j = col_min,  value = stats$n_min[dt_year$id_pos])
    set(cell_data, i = yr_rows, j = col_mean, value = stats$n_mean[dt_year$id_pos])
  }
}

t_end <- Sys.time()
cat(sprintf("Neighbor features computed in %.1f minutes\n", 
            difftime(t_end, t_start, units = "mins")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Clean up helper column and apply pre-trained Random Forest
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_data[, id_pos := NULL]

# The pre-trained RF model (rf_model) expects a data.frame with the ~110 
# predictor columns. The 15 new neighbor feature columns are now populated
# with numerically identical values to the original pipeline.

# Example prediction (adjust column selection to match your trained model):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of What Changed and Why

| Aspect | Original | Optimized |
|---|---|---|
| **Adjacency representation** | Per-row string-key lookup list (6.46M entries) | Sparse matrix `A` (344K Ã— 344K, ~1.37M nonzeros) built once |
| **Mean computation** | `lapply` over 6.46M rows, each calling `mean()` | Sparse matrix-vector multiply: `A %*% x` (microseconds per year) |
| **Min/Max computation** | `lapply` over 6.46M rows, each calling `max()/min()` | `data.table` grouped aggregation over ~1.37M edge rows (milliseconds per year) |
| **NA handling** | Per-row `!is.na()` filtering | Vectorized: zero-fill + valid-count sparse multiply for mean; `tmp[!is.na(val)]` for min/max |
| **Year handling** | Entangled in monolithic 6.46M-row lookup | Factored out: same `A` reused across 28 year-slices |
| **Memory** | ~6.46M-element list of integer vectors | ~1.37M nonzeros in sparse matrix + ~1.37M-row edge `data.table` (~50 MB total) |
| **Expected runtime** | 86+ hours | **2â€“10 minutes** |
| **Numerical result** | Original | Identical (same neighbor sets, same arithmetic) |
| **RF model** | Pre-trained | **Untouched â€” no retraining** |
END_RESPONSE>>>
