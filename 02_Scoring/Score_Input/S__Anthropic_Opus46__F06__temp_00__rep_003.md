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
run_id: Anthropic_F06_T00_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows
For each of the ~6.46M cell-year rows, the function:
1. Looks up the cell's rook neighbors from the `nb` object (keyed by cell ID).
2. Constructs character paste-keys (`"id_year"`) for each neighbor in that year.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) â€” **named vector lookup in R is hash-based but has high per-call overhead when done 6.46M times**.

The result is a **list of 6.46M integer vectors**. This is memory-heavy (~hundreds of MB) and the character key construction + lookup is extremely slow at this scale.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over ~6.46 million rows, repeated 5 times
For each variable, another `lapply` iterates over all 6.46M rows, subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per source variable), totaling ~32.3M R-level function calls with per-element subsetting.

### Why it takes 86+ hours
- **Character key construction**: `paste()` over 6.46M rows Ã— ~4 neighbors each â‰ˆ 25.8M string operations.
- **Named vector lookup**: R's named vector lookup is O(1) amortized but has significant constant overhead; 25.8M lookups is very slow.
- **`lapply` over 6.46M elements**: R's `lapply` has non-trivial per-iteration overhead. Doing this 6 times (1 for lookup + 5 for stats) is devastating.
- **No vectorization**: The core operations (subsetting, max/min/mean) are done element-by-element rather than in bulk.

### Why raster focal/kernel operations are NOT directly applicable
Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed rectangular kernel. While the cells here are on a grid, the rook-neighbor structure is already precomputed as an `nb` object, and the panel has a time dimension. Focal operations would require reshaping each variable Ã— year into a raster, applying the focal function, then extracting back â€” which is feasible but introduces complexity around boundary cells, missing data, and ensuring exact numerical equivalence. The **sparse-matrix approach below is more direct, faster, and guarantees identical results**.

---

## 2. Optimization Strategy

### Core Idea: Replace row-wise `lapply` with sparse matrix multiplication

The neighbor relationships can be encoded as a **sparse adjacency matrix** `W` of dimension `N Ã— N` (where `N` â‰ˆ 6.46M). Each row `i` has non-zero entries at columns `j` where `j` is a rook neighbor of `i` **in the same year**.

Then:
- **Neighbor mean** = `(W %*% x) / (W %*% 1_valid)` (where `1_valid` is an indicator of non-NA)
- **Neighbor max** and **min** require a different trick since sparse matrix multiplication computes sums, not extrema.

For **max** and **min**, we use a `data.table` group-by approach: expand the neighbor pairs into an edge list `(i, j)`, join the variable values, and compute `max`/`min`/`mean` grouped by `i`.

### Specific steps:

1. **Build an edge list once** (vectorized, no `lapply` over 6.46M rows):
   - Expand the `nb` object into a cell-level edge list (cell_from, cell_to).
   - Cross-join with years using `data.table` to get (row_from, row_to) in the panel.

2. **Compute all three stats via `data.table` grouped aggregation**:
   - For each variable, join the edge list with the variable's values, then `group by row_from` to compute `max`, `min`, `mean`.

3. **This replaces both `build_neighbor_lookup` and `compute_neighbor_stats`** with fully vectorized operations.

### Expected speedup:
- Edge list construction: seconds (vectorized).
- Per-variable stats: the edge list has ~6.46M Ã— ~4 â‰ˆ ~27M rows. A `data.table` group-by over 27M rows computing max/min/mean is **seconds to low minutes** per variable.
- **Total: ~5â€“15 minutes** instead of 86+ hours.

### Memory check:
- Edge list: ~27M rows Ã— 2 integer columns â‰ˆ 216 MB.
- Panel data: 6.46M rows Ã— 110 columns â‰ˆ manageable within 16 GB.
- Feasible on a 16 GB laptop.

---

## 3. Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the cell-level edge list from the nb object
# ============================================================
# rook_neighbors_unique is an nb object: a list of length = number of cells.
# rook_neighbors_unique[[i]] gives integer indices of neighbors of cell i
# (in the ordering defined by id_order).
# id_order is a vector mapping position -> cell id.

build_cell_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer vectors)
  n_cells <- length(neighbors)
  
  # Pre-compute total number of edges for memory pre-allocation
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  from_ref <- integer(n_edges)
  to_ref   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors[[i]]
    len  <- length(nb_i)
    if (len > 0L) {
      from_ref[pos:(pos + len - 1L)] <- i
      to_ref[pos:(pos + len - 1L)]   <- nb_i
      pos <- pos + len
    }
  }
  
  data.table(
    from_cell_id = id_order[from_ref],
    to_cell_id   = id_order[to_ref]
  )
}

# ============================================================
# STEP 2: Expand cell edges to panel-row edges (cross with years)
# ============================================================
build_panel_edge_list <- function(cell_data_dt, cell_edges) {
  # cell_data_dt must have columns: id, year, and a row index .row_idx
  # We join edges with the panel to get row indices for (from, year) and (to, year)
  
  # Create lookup: (id, year) -> row index
  cell_data_dt[, .row_idx := .I]
  
  lookup <- cell_data_dt[, .(.row_idx, id, year)]
  setkey(lookup, id, year)
  
  # Get unique years
  years <- sort(unique(cell_data_dt$year))
  
  # Cross join cell_edges with years
  # cell_edges has ~1.37M rows, years has 28 -> ~38.4M rows
  # This is the directed edge list across all years
  panel_edges <- cell_edges[, .(from_cell_id, to_cell_id, year = list(years)), 
                            by = .(from_cell_id, to_cell_id)]
  
  # More memory-efficient: use CJ-like expansion
  panel_edges <- cell_edges[, CJ(edge_idx = .I, year = years)]
  panel_edges[, `:=`(
    from_cell_id = cell_edges$from_cell_id[edge_idx],
    to_cell_id   = cell_edges$to_cell_id[edge_idx]
  )]
  panel_edges[, edge_idx := NULL]
  
  # Join to get row indices for 'from' side
  setkey(panel_edges, from_cell_id, year)
  panel_edges <- lookup[panel_edges, 
                        .(from_row = .row_idx, to_cell_id, year),
                        on = .(id = from_cell_id, year), 
                        nomatch = NULL]
  
  # Join to get row indices for 'to' side
  setkey(panel_edges, to_cell_id, year)
  panel_edges <- lookup[panel_edges,
                        .(from_row, to_row = .row_idx, year),
                        on = .(id = to_cell_id, year),
                        nomatch = NULL]
  
  # Keep only the row index pairs
  panel_edges[, year := NULL]
  panel_edges
}

# ============================================================
# STEP 3: Compute neighbor stats for one variable
# ============================================================
compute_neighbor_stats_fast <- function(cell_data_dt, panel_edges, var_name) {
  # panel_edges: data.table with columns from_row, to_row
  # Extract neighbor values
  vals <- cell_data_dt[[var_name]]
  
  # Build working table
  work <- panel_edges[, .(from_row, neighbor_val = vals[to_row])]
  
  # Remove NAs in neighbor values
  work <- work[!is.na(neighbor_val)]
  
  # Group by from_row and compute stats
  stats <- work[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = from_row]
  
  # Create output columns (NA for rows with no valid neighbors)
  n <- nrow(cell_data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)
  
  out_max[stats$from_row]  <- stats$nb_max
  out_min[stats$from_row]  <- stats$nb_min
  out_mean[stats$from_row] <- stats$nb_mean
  
  # Naming convention: match original feature names
  max_name  <- paste0("neighbor_max_", var_name)
  min_name  <- paste0("neighbor_min_", var_name)
  mean_name <- paste0("neighbor_mean_", var_name)
  
  cell_data_dt[, (max_name)  := out_max]
  cell_data_dt[, (min_name)  := out_min]
  cell_data_dt[, (mean_name) := out_mean]
  
  invisible(cell_data_dt)
}

# ============================================================
# MAIN PIPELINE
# ============================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  cat("Converting to data.table...\n")
  cell_data_dt <- as.data.table(cell_data)
  cell_data_dt[, .row_idx := .I]
  
  # Step 1: Cell-level edge list (fast, ~1.37M rows)
  cat("Building cell-level edge list...\n")
  cell_edges <- build_cell_edge_list(id_order, rook_neighbors_unique)
  cat(sprintf("  Cell edges: %d rows\n", nrow(cell_edges)))
  
  # Step 2: Expand to panel-level edge list
  cat("Building panel-level edge list...\n")
  
  # Memory-efficient approach: iterate over years to avoid massive CJ
  years <- sort(unique(cell_data_dt$year))
  
  # Build lookup
  lookup <- cell_data_dt[, .(.row_idx, id, year)]
  setkey(lookup, id, year)
  
  # For each year, join cell_edges with lookup to get row indices
  panel_edge_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    lookup_yr <- lookup[year == yr]
    setkey(lookup_yr, id)
    
    # Join from side
    edges_yr <- cell_edges[lookup_yr, 
                           .(from_row = i..row_idx, to_cell_id),
                           on = .(from_cell_id = id),
                           nomatch = NULL]
    
    # Join to side
    edges_yr <- edges_yr[lookup_yr,
                         .(from_row, to_row = i..row_idx),
                         on = .(to_cell_id = id),
                         nomatch = NULL]
    
    panel_edge_list[[yi]] <- edges_yr
    
    if (yi %% 7 == 0) cat(sprintf("  Processed year %d (%d/%d)\n", yr, yi, length(years)))
  }
  
  panel_edges <- rbindlist(panel_edge_list)
  rm(panel_edge_list)
  gc()
  cat(sprintf("  Panel edges: %d rows\n", nrow(panel_edges)))
  
  # Step 3: Compute neighbor features for each variable
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    compute_neighbor_stats_fast(cell_data_dt, panel_edges, var_name)
  }
  
  cat("Done. Converting back to data.frame...\n")
  cell_data_dt[, .row_idx := NULL]
  
  as.data.frame(cell_data_dt)
}

# ============================================================
# USAGE (drop-in replacement for the original outer loop)
# ============================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed with prediction using the pre-trained Random Forest:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Correctness Guarantee & Memory-Safer Year-by-Year Variant

The code above builds the panel edge list **year by year** to stay within 16 GB RAM. However, if even `rbindlist` of all years is too large (~38M rows Ã— 2 int cols â‰ˆ 305 MB, which is fine), you can alternatively compute stats year-by-year and avoid holding the full edge list:

```r
# ============================================================
# MEMORY-MINIMAL VARIANT: compute stats year-by-year
# ============================================================
run_neighbor_features_yearwise <- function(cell_data, id_order, rook_neighbors_unique) {
  
  cell_data_dt <- as.data.table(cell_data)
  cell_data_dt[, .row_idx := .I]
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    cell_data_dt[, paste0("neighbor_max_", var_name)  := NA_real_]
    cell_data_dt[, paste0("neighbor_min_", var_name)  := NA_real_]
    cell_data_dt[, paste0("neighbor_mean_", var_name) := NA_real_]
  }
  
  # Cell-level edge list
  cell_edges <- build_cell_edge_list(id_order, rook_neighbors_unique)
  
  years <- sort(unique(cell_data_dt$year))
  lookup <- cell_data_dt[, .(.row_idx, id, year)]
  setkey(lookup, id, year)
  
  for (yr in years) {
    cat(sprintf("Year %d...\n", yr))
    
    lookup_yr <- lookup[year == yr]
    setkey(lookup_yr, id)
    
    # Build row-index edge list for this year
    edges_yr <- cell_edges[lookup_yr,
                           .(from_row = i..row_idx, to_cell_id),
                           on = .(from_cell_id = id),
                           nomatch = NULL]
    edges_yr <- edges_yr[lookup_yr,
                         .(from_row, to_row = i..row_idx),
                         on = .(to_cell_id = id),
                         nomatch = NULL]
    
    for (var_name in neighbor_source_vars) {
      vals <- cell_data_dt[[var_name]]
      work <- edges_yr[, .(from_row, nv = vals[to_row])]
      work <- work[!is.na(nv)]
      
      if (nrow(work) > 0) {
        stats <- work[, .(nb_max = max(nv), nb_min = min(nv), nb_mean = mean(nv)),
                      by = from_row]
        
        set(cell_data_dt, i = stats$from_row, 
            j = paste0("neighbor_max_", var_name), value = stats$nb_max)
        set(cell_data_dt, i = stats$from_row, 
            j = paste0("neighbor_min_", var_name), value = stats$nb_min)
        set(cell_data_dt, i = stats$from_row, 
            j = paste0("neighbor_mean_", var_name), value = stats$nb_mean)
      }
    }
  }
  
  cell_data_dt[, .row_idx := NULL]
  as.data.frame(cell_data_dt)
}
```

---

## 5. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste`/named-vector lookup | Vectorized `data.table` join, year-by-year |
| **Stats computation** | `lapply` over 6.46M rows Ã— 5 vars | `data.table` grouped aggregation on edge list |
| **Estimated runtime** | 86+ hours | **~5â€“20 minutes** |
| **Peak RAM** | ~6â€“10 GB (6.46M-element list of integer vectors) | ~4â€“6 GB (edge list + panel data) |
| **Numerical results** | Baseline | **Identical** (same max/min/mean, same NA handling) |
| **RF model** | **Preserved** (not retrained) | **Preserved** (not retrained) |

The optimized code is a **drop-in replacement** for the original `build_neighbor_lookup` + `compute_neighbor_stats` + outer loop block. The pre-trained Random Forest model is never touched; only the feature-engineering step is accelerated.
END_RESPONSE>>>
