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
run_id: Anthropic_F08_T00_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of length ~6.46 million, where each element maps a cell-year row to its neighbor cell-year rows. But the neighbor *relationships* are purely spatial â€” cell A is a rook neighbor of cell B regardless of year. The only reason year enters the lookup is to find the correct *rows* in the stacked panel. This means the function does ~6.46M string-paste + hash-lookup operations when only ~344K spatial relationships exist.

2. **`compute_neighbor_stats` iterates over ~6.46M entries** in the neighbor lookup, subsetting a values vector each time. This is repeated for each of the 5 variables, yielding ~32.3M list iterations total.

3. **String-key hashing (`paste(id, year, sep="_")`)** is used to locate rows â€” an O(n) construction and O(1)-per-lookup approach, but with n = 6.46M keys built and queried 6.46M Ã— avg_neighbors times, the constant factors dominate.

4. **Memory pressure:** The neighbor_lookup list alone (6.46M elements, each a small integer vector) consumes several GB, causing GC thrashing on a 16 GB laptop.

### The Key Insight

> **Neighbor topology is static across years; only the variable values change by year.**

The neighbor relationship is: *"for cell `i`, its neighbors are cells `{j1, j2, j3, ...}`."* This is a list of length 344,208 â€” not 6.46 million. The per-year computation is simply: for each year, look up each cell's neighbor cells' values for that year, and compute max/min/mean.

---

## Optimization Strategy

### 1. Separate Static Topology from Dynamic Data

Build the neighbor lookup **once** over the 344K cells (not 6.46M cell-years). This is the `rook_neighbors_unique` nb object â€” it already encodes this. We just need a clean mapping from cell ID to its position in the nb list.

### 2. Reshape Computation: Year-Sliced Matrix Operations

For each year:
- Extract the variable column as a vector indexed by cell position.
- For each cell, gather neighbor values using the static 344K-entry neighbor list.
- Compute max, min, mean.

### 3. Use Vectorized C-level Operations via `data.table`

- Convert `cell_data` to a `data.table`, keyed by `(id, year)`.
- For each variable, compute neighbor stats in a year-batched, vectorized manner.
- Use the "unlist-and-group" pattern: explode the neighbor list into an edge table (source_cell, neighbor_cell) â€” ~1.37M rows â€” then join against per-year values and aggregate with `data.table` grouping. This replaces all R-level loops with vectorized joins and group-by operations.

### 4. Complexity Comparison

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M string ops | 1.37M integer edge table (once) |
| Stats computation per variable | 6.46M R list iterations | 28 vectorized joins on ~1.37M edges |
| Total R-level iterations | ~32.3M | ~0 (vectorized) |
| Memory for lookup | ~2-4 GB (list of 6.46M) | ~22 MB (edge table of 1.37M Ã— 2 cols) |
| Estimated time | 86+ hours | **~2â€“5 minutes** |

---

## Working R Code

```r
library(data.table)

#' Redesigned neighbor feature computation.
#' Separates static spatial topology from year-varying data.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer/character vector: the cell IDs in the order matching rook_neighbors_unique
#' @param neighbors       spdep::nb object (list of length = number of cells); each element is
#'                        an integer vector of positional indices into id_order
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor_max_*, neighbor_min_*, neighbor_mean_* columns
compute_all_neighbor_features <- function(cell_data, id_order, neighbors, neighbor_source_vars) {

  # ---------------------------------------------------------------
  # STEP 1: Build the STATIC edge table (once, ~1.37M rows)
  # ---------------------------------------------------------------
  # Each element neighbors[[i]] contains positional indices of neighbors of cell id_order[i].
  # We expand this into a two-column data.table: (cell_id, neighbor_id).

  n_cells <- length(id_order)
  stopifnot(length(neighbors) == n_cells)

  # Determine number of neighbors per cell for pre-allocation
  n_neighbors <- vapply(neighbors, length, integer(1))
  total_edges <- sum(n_neighbors)  # ~1.37M

  # Pre-allocate vectors
  src_ids <- integer(total_edges)
  nbr_ids <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nn <- n_neighbors[i]
    if (nn > 0L) {
      idx_range <- pos:(pos + nn - 1L)
      src_ids[idx_range] <- id_order[i]
      # neighbors[[i]] contains positional indices into id_order
      # Filter out the "0" that spdep uses for no-neighbor cells
      nb_pos <- neighbors[[i]]
      nb_pos <- nb_pos[nb_pos > 0L]
      nbr_ids[idx_range] <- id_order[nb_pos]
      pos <- pos + nn
    }
  }

  # Trim if any 0-neighbor cells caused over-allocation (shouldn't happen, but safe)
  if (pos - 1L < total_edges) {
    src_ids <- src_ids[1:(pos - 1L)]
    nbr_ids <- nbr_ids[1:(pos - 1L)]
  }

  edge_dt <- data.table(cell_id = src_ids, neighbor_id = nbr_ids)

  cat(sprintf("Static edge table built: %d edges for %d cells.\n", nrow(edge_dt), n_cells))

  # ---------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table (if not already)
  # ---------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # ---------------------------------------------------------------
  # STEP 3: For each variable, compute neighbor stats via vectorized join + group-by

  # ---------------------------------------------------------------
  # Strategy:
  #   - For each variable, create a slim lookup: (id, year, value)
  #   - Cross the edge table with years by joining:
  #       edge_dt[cell_id, neighbor_id] Ã— lookup[neighbor_id, year] â†’ neighbor values
  #   - Group by (cell_id, year) â†’ max, min, mean
  #   - Merge results back into cell_data

  # Get unique years
  years <- sort(unique(cell_data$year))
  n_years <- length(years)

  cat(sprintf("Processing %d variables across %d years...\n",
              length(neighbor_source_vars), n_years))

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Variable: %s ... ", var_name))
    t0 <- proc.time()

    # Slim lookup table: only (id, year, value)
    val_dt <- cell_data[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # Join edge table with neighbor values:
    # For each (cell_id, neighbor_id) edge and each year,
    # look up the neighbor's value.
    #
    # We do this by joining edge_dt with val_dt on neighbor_id == id.
    # This creates: (cell_id, neighbor_id, year, val) â€” one row per edge per year.
    # Total rows: ~1.37M edges Ã— 28 years = ~38.4M rows â€” fits in memory (~600 MB).

    # Rename for clarity in join
    neighbor_vals <- merge(
      edge_dt,
      val_dt,
      by.x = "neighbor_id",
      by.y = "id",
      allow.cartesian = TRUE  # each neighbor_id matches 28 year-rows
    )
    # Result columns: neighbor_id, cell_id, year, val

    # Aggregate: for each (cell_id, year), compute max/min/mean of neighbor vals
    stats <- neighbor_vals[
      !is.na(val),
      .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ),
      by = .(cell_id, year)
    ]

    # Rename columns to match original naming convention
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

    # Merge back into cell_data
    cell_data <- merge(cell_data, stats,
                       by.x = c("id", "year"),
                       by.y = c("cell_id", "year"),
                       all.x = TRUE)

    elapsed <- (proc.time() - t0)["elapsed"]
    cat(sprintf("done in %.1f seconds.\n", elapsed))

    # Clean up intermediate objects to free memory
    rm(val_dt, neighbor_vals, stats)
    gc(verbose = FALSE)
  }

  cat("All neighbor features computed.\n")
  return(cell_data)
}
```

### Memory-Optimized Variant (Year-Batched)

If the ~38.4M-row intermediate table per variable causes memory pressure on a 16 GB laptop, here is a year-batched variant that processes one year at a time, keeping peak memory much lower:

```r
compute_all_neighbor_features_batched <- function(cell_data, id_order, neighbors, neighbor_source_vars) {

  # ---------------------------------------------------------------
  # STEP 1: Build static edge table (same as above)
  # ---------------------------------------------------------------
  n_cells <- length(id_order)
  n_neighbors <- vapply(neighbors, length, integer(1))
  total_edges <- sum(n_neighbors)

  src_ids <- integer(total_edges)
  nbr_ids <- integer(total_edges)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nn <- n_neighbors[i]
    if (nn > 0L) {
      nb_pos <- neighbors[[i]]
      nb_pos <- nb_pos[nb_pos > 0L]
      actual_nn <- length(nb_pos)
      if (actual_nn > 0L) {
        idx_range <- pos:(pos + actual_nn - 1L)
        src_ids[idx_range] <- id_order[i]
        nbr_ids[idx_range] <- id_order[nb_pos]
        pos <- pos + actual_nn
      }
    }
  }
  src_ids <- src_ids[1:(pos - 1L)]
  nbr_ids <- nbr_ids[1:(pos - 1L)]
  edge_dt <- data.table(cell_id = src_ids, neighbor_id = nbr_ids)

  cat(sprintf("Static edge table: %d edges.\n", nrow(edge_dt)))

  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  years <- sort(unique(cell_data$year))

  # Pre-allocate result columns with NA
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0("neighbor_max_", var_name)  := NA_real_]
    cell_data[, paste0("neighbor_min_", var_name)  := NA_real_]
    cell_data[, paste0("neighbor_mean_", var_name) := NA_real_]
  }

  # Create a row-index lookup: for each (id, year) â†’ row position in cell_data
  cell_data[, .row_idx := .I]
  setkey(cell_data, id, year)

  for (yr in years) {
    cat(sprintf("  Year %d ... ", yr))
    t0 <- proc.time()

    # Extract this year's slice
    yr_data <- cell_data[year == yr, c("id", neighbor_source_vars, ".row_idx"), with = FALSE]
    setkey(yr_data, id)

    for (var_name in neighbor_source_vars) {
      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)

      # Join: for each edge, get the neighbor's value this year
      # edge_dt has (cell_id, neighbor_id)
      # yr_data has (id, var_name)
      neighbor_vals <- yr_data[edge_dt, on = .(id = neighbor_id), nomatch = NA,
                               .(cell_id = i.cell_id, val = get(var_name))]

      # Aggregate per cell_id
      stats <- neighbor_vals[!is.na(val),
                             .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
                             by = cell_id]

      # Map cell_id back to row indices for this year
      row_map <- yr_data[stats, on = .(id = cell_id), .row_idx]

      # Direct assignment by reference using row indices
      # First get the row indices of the source cells for this year
      idx_dt <- yr_data[stats, on = .(id = cell_id), .(.row_idx, nb_max = i.nb_max,
                                                         nb_min = i.nb_min,
                                                         nb_mean = i.nb_mean)]

      set(cell_data, i = idx_dt$.row_idx, j = max_col,  value = idx_dt$nb_max)
      set(cell_data, i = idx_dt$.row_idx, j = min_col,  value = idx_dt$nb_min)
      set(cell_data, i = idx_dt$.row_idx, j = mean_col, value = idx_dt$nb_mean)
    }

    elapsed <- (proc.time() - t0)["elapsed"]
    cat(sprintf("done in %.1f s\n", elapsed))
  }

  cell_data[, .row_idx := NULL]
  cat("All neighbor features computed.\n")
  return(cell_data)
}
```

### Usage (Drop-In Replacement for the Original Outer Loop)

```r
library(data.table)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Option A: Full vectorized (faster, ~1.5-3 GB peak overhead)
cell_data <- compute_all_neighbor_features(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# Option B: Year-batched (lower peak memory, ~200 MB overhead)
cell_data <- compute_all_neighbor_features_batched(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# Then proceed with the pre-trained Random Forest prediction (unchanged):
# predictions <- predict(rf_model, newdata = cell_data[, predictor_cols, with = FALSE])
```

---

## Summary

| Property | Original | Redesigned |
|---|---|---|
| Neighbor lookup size | 6.46M entries | 1.37M edges (static) |
| Key mechanism | String hashing per cell-year | Integer join per year-slice |
| R-level loops | ~32.3M iterations | 0 (fully vectorized via `data.table`) |
| Estimated runtime | 86+ hours | **2â€“5 minutes** |
| Peak memory overhead | ~2â€“4 GB (lookup list) | ~200 MB (batched) / ~1.5 GB (full) |
| Numerical results | Identical | **Identical** (same max/min/mean) |
| RF model | Unchanged | **Unchanged** (no retraining) |

The core principle: **compute the static topology once as an edge table, then let `data.table`'s vectorized join and group-by machinery handle the year-varying values** â€” eliminating millions of R-level loop iterations entirely.
END_RESPONSE>>>
