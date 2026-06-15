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
run_id: Anthropic_F05_T07_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. Converts an ID to a character string and looks it up in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Calls `paste(..., sep="_")` to build string keys for every neighbor of that row.
4. Performs named-vector lookup (`idx_lookup[neighbor_keys]`) â€” which is an **O(k)** hash lookup per neighbor, but the **string construction** and **repeated per-row R-level function calls** dominate.

With ~6.46M rows and an average of ~4 rook neighbors each, that's **~25.8 million `paste()` calls and hash lookups**, all inside an interpreted R loop. This alone accounts for the bulk of the 86+ hour estimate.

### Why It's a Broader Architectural Issue

The string-keyed lookup is a **workaround for the lack of a proper integer index mapping** between `(cell_id, year)` pairs and row positions. The data is a balanced panel (344,208 cells Ã— 28 years), which means:

- Every cell appears in every year.
- Neighbor relationships are **time-invariant** (spatial neighbors don't change across years).
- The neighbor row indices for cell `i` in year `t` are the **same offset pattern** as for cell `i` in year `t'`, just shifted by a fixed stride.

This regularity means we can **completely eliminate the per-row lookup** and replace it with **vectorized integer arithmetic**.

### Secondary Inefficiency

`compute_neighbor_stats` also uses an `lapply` over 6.46M elements, computing `max`, `min`, `mean` one row at a time. This can be replaced with a **sparse-matrix multiplication / vectorized grouped aggregation**.

---

## Optimization Strategy

### Strategy 1: Exploit the Balanced Panel Structure with Integer Arithmetic

If the data is sorted by `(id, year)` or `(year, id)` in a consistent order, neighbor row indices can be computed as a **fixed offset** from each cell's block of rows, eliminating all string operations entirely.

### Strategy 2: Vectorized Neighbor Stats via Sparse Matrix

Build a **sparse neighbor matrix** (6.46M Ã— 6.46M, but very sparse â€” ~4 entries per row) once, then compute neighbor sums, counts, max, and min using vectorized operations or `data.table` grouped joins.

### Strategy 3: Hybrid â€” Integer Index Map + data.table Grouped Stats

Build the neighbor lookup with pure integer indexing, then compute stats with vectorized R.

Below I implement the **full hybrid approach** that reduces runtime from ~86 hours to **minutes**.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats
# Preserves the exact same numerical output (max, min, mean of neighbors).
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 1: Build the neighbor lookup ONCE using integer arithmetic --------
# Assumptions validated below:
#   - cell_data is a balanced panel: every cell in every year
#   - We know the cell ordering and year ordering

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table for speed (non-destructive)
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # --- Validate balanced panel ---
  n_cells <- length(id_order)
  n_years <- length(unique(dt$year))
  stopifnot(
    "Data is not a balanced panel" = nrow(dt) == n_cells * n_years
  )
  
  years <- sort(unique(dt$year))
  n_years <- length(years)
  
  # --- Build a fast (id, year) -> row_idx map using integer keys ---
  # Create integer mapping: id -> cell_position (1..n_cells)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  year_to_pos <- setNames(seq_along(years), as.character(years))
  
  # Compute the position for every row: (cell_pos, year_pos)
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_pos := year_to_pos[as.character(year)]]
  
  # Build a 2D -> 1D index matrix: index_mat[cell_pos, year_pos] = row_idx
  # This is the KEY insight: replaces all string hashing with array indexing
  index_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  index_mat[cbind(dt$cell_pos, dt$year_pos)] <- dt$row_idx
  
  # --- Build edge list (from_row, to_row) for ALL cell-year pairs ---
  # neighbors is an nb object: neighbors[[cell_pos]] gives neighbor cell positions
  # We need to map from id_order positions to the nb object positions
  
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Pre-build the full edge list: for each cell, get its neighbor cell positions
  # Then expand across all years
  
  # Step A: Build cell-level edge list
  cat("Building cell-level neighbor edge list...\n")
  from_cell <- integer(0)
  to_cell   <- integer(0)
  
  # Estimate total edges for pre-allocation
  total_edges <- sum(lengths(neighbors))
  from_cell <- integer(total_edges)
  to_cell   <- integer(total_edges)
  
  offset <- 0L
  for (ci in seq_along(id_order)) {
    ref_idx <- id_to_ref[as.character(id_order[ci])]
    nb_refs <- neighbors[[ref_idx]]
    if (length(nb_refs) == 0) next
    nb_cell_ids <- id_order[nb_refs]
    nb_cell_pos <- id_to_pos[as.character(nb_cell_ids)]
    nb_cell_pos <- nb_cell_pos[!is.na(nb_cell_pos)]
    n_nb <- length(nb_cell_pos)
    if (n_nb == 0) next
    idx_range <- (offset + 1L):(offset + n_nb)
    from_cell[idx_range] <- ci
    to_cell[idx_range]   <- nb_cell_pos
    offset <- offset + n_nb
  }
  from_cell <- from_cell[1:offset]
  to_cell   <- to_cell[1:offset]
  
  cat(sprintf("  Cell-level edges: %d\n", length(from_cell)))
  
  # Step B: Expand to all years â€” pure integer arithmetic
  cat("Expanding to cell-year edge list across all years...\n")
  n_cell_edges <- length(from_cell)
  
  # For each year, map (from_cell, year) -> from_row and (to_cell, year) -> to_row
  from_rows <- integer(n_cell_edges * n_years)
  to_rows   <- integer(n_cell_edges * n_years)
  valid     <- logical(n_cell_edges * n_years)
  
  for (yi in seq_len(n_years)) {
    rng <- ((yi - 1L) * n_cell_edges + 1L):(yi * n_cell_edges)
    fr <- index_mat[cbind(from_cell, rep(yi, n_cell_edges))]
    tr <- index_mat[cbind(to_cell,   rep(yi, n_cell_edges))]
    from_rows[rng] <- fr
    to_rows[rng]   <- tr
    valid[rng]     <- !is.na(fr) & !is.na(tr)
  }
  
  # Filter to valid edges
  keep <- which(valid)
  edge_dt <- data.table(
    from_row = from_rows[keep],
    to_row   = to_rows[keep]
  )
  
  cat(sprintf("  Total cell-year edges: %d\n", nrow(edge_dt)))
  
  # Clean up large temporaries
  rm(from_rows, to_rows, valid, from_cell, to_cell, keep)
  gc()
  
  return(list(
    edge_dt   = edge_dt,
    index_mat = index_mat,
    n_rows    = nrow(dt)
  ))
}

# ---- Step 2: Compute neighbor stats vectorized via data.table ---------------

compute_neighbor_stats_fast <- function(data, edge_dt, var_name) {
  # edge_dt has columns: from_row, to_row
  # For each from_row, gather var values at to_row, compute max/min/mean
  
  n <- nrow(data)
  vals <- data[[var_name]]
  
  # Attach neighbor values to edge list
  work <- copy(edge_dt)
  work[, val := vals[to_row]]
  
  # Remove edges where the neighbor value is NA
  work <- work[!is.na(val)]
  
  # Grouped aggregation â€” this is the big speedup
  stats <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = from_row]
  
  # Initialize output with NAs
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(out) <- c("max", "min", "mean")
  
  out[stats$from_row, 1] <- stats$nb_max
  out[stats$from_row, 2] <- stats$nb_min
  out[stats$from_row, 3] <- stats$nb_mean
  
  return(out)
}

# ---- Step 3: Wrapper that adds columns (matches original interface) ---------

compute_and_add_neighbor_features_fast <- function(data, var_name, edge_dt) {
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))
  stats <- compute_neighbor_stats_fast(data, edge_dt, var_name)
  
  data[[paste0(var_name, "_max_neighbor")]]  <- stats[, 1]
  data[[paste0(var_name, "_min_neighbor")]]  <- stats[, 2]
  data[[paste0(var_name, "_mean_neighbor")]] <- stats[, 3]
  
  return(data)
}

# ---- Step 4: Main execution (drop-in replacement for outer loop) -----------

cat("=== OPTIMIZED NEIGHBOR FEATURE PIPELINE ===\n")
t0 <- Sys.time()

# Build the edge list ONCE (replaces build_neighbor_lookup)
cat("Phase 1: Building neighbor edge list...\n")
nb_structure <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
edge_dt <- nb_structure$edge_dt

cat(sprintf("  Edge list built in %.1f seconds\n", 
            as.numeric(Sys.time() - t0, units = "secs")))

# Compute features for all 5 variables (replaces the for loop)
cat("Phase 2: Computing neighbor features...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  t1 <- Sys.time()
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, edge_dt)
  cat(sprintf("  %s done in %.1f seconds\n", var_name,
              as.numeric(Sys.time() - t1, units = "secs")))
}

cat(sprintf("=== TOTAL TIME: %.1f minutes ===\n",
            as.numeric(Sys.time() - t0, units = "mins")))

# Clean up
rm(nb_structure, edge_dt)
gc()

# The cell_data now has the same 15 new columns (5 vars Ã— 3 stats) as the
# original code. The trained Random Forest model can be used directly for
# prediction without any changes.
```

---

## Why This Is Correct and Equivalent

| Concern | Guarantee |
|---|---|
| **Same numerical output** | For each row, we gather the exact same set of neighbor row indices (same cell IDs, same year), extract the same values, and compute the same `max`, `min`, `mean`. The `data.table` grouped aggregation is algebraically identical. |
| **Same column names** | The wrapper uses the same `paste0(var_name, "_max_neighbor")` naming convention. |
| **RF model untouched** | No model retraining â€” we only reconstruct the feature columns that already existed. |
| **Handles NAs identically** | Edges with `NA` neighbor values are dropped before aggregation, and rows with no valid neighbors get `NA` â€” same as the original `!is.na()` filtering. |

## Performance Estimate

| Phase | Original | Optimized | Speedup |
|---|---|---|---|
| Build lookup | ~60+ hours (6.46M `paste`+hash in `lapply`) | ~30â€“90 seconds (integer matrix indexing + vectorized expansion) | ~2000â€“7000Ã— |
| Compute stats (Ã—5 vars) | ~25+ hours (6.46M `lapply` per var) | ~30â€“60 seconds per var (~5 min total) (`data.table` grouped agg) | ~300â€“1500Ã— |
| **Total** | **~86+ hours** | **~5â€“10 minutes** | **~500â€“1000Ã—** |

### Memory Footprint

The edge list is ~38.4M rows Ã— 2 integer columns â‰ˆ **~600 MB**. The index matrix is 344,208 Ã— 28 integers â‰ˆ **~39 MB**. Both fit comfortably in 16 GB RAM alongside the 6.46M-row dataset.
END_RESPONSE>>>
