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
run_id: Anthropic_F07_T00_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates a list of ~6.46 million elements, each produced by an `lapply` call that performs character-based key lookups (`paste` + named-vector indexing) for every single row. This is O(N) string operations with N â‰ˆ 6.46M, and the named-vector lookup (`idx_lookup[neighbor_keys]`) is itself O(k) per row where k is the number of neighbors â€” but the constant factor of `paste` + character matching on a 6.46M-length named vector is enormous.

Then **`compute_neighbor_stats`** iterates over the same 6.46M-element list *five times* (once per variable), each time extracting neighbor values via R-level indexing in a loop.

**Root causes:**

1. **Character key lookups on a 6.46M named vector** â€” R's named vector lookup is hash-based but the overhead of creating ~6.46M paste keys and looking them up is massive.
2. **Row-level `lapply` over 6.46M rows** â€” pure R loop overhead.
3. **Redundant work** â€” the neighbor topology is year-invariant (same grid, same neighbors every year), but the lookup is rebuilt as if each cell-year is unique.
4. **`compute_neighbor_stats` uses R-level loops** â€” should be vectorized or pushed to C/C++ via data.table or matrix operations.

## Optimization Strategy

### Key Insight: Exploit Year-Invariance

The rook-neighbor structure is **spatial only** â€” cell A's neighbors are the same in every year. So we should:

1. **Build the neighbor lookup once at the cell level** (344K cells, not 6.46M cell-years).
2. **For each year, use vectorized (column-wise) operations** to compute max/min/mean across neighbors.

### Implementation Plan

1. **Convert `rook_neighbors_unique` (an `nb` object) into a sparse adjacency matrix** using `spdep::nb2listw` â†’ `listw2mat` or directly via `nb2mat`, or better, construct a `dgCMatrix` (sparse matrix from the Matrix package). This gives us a 344K Ã— 344K sparse matrix with ~1.37M non-zero entries.
2. **Reshape the computation**: for each year and each variable, extract the column vector of values for all cells in that year, then use **sparse matrix multiplication** to compute the sum and count of neighbors, from which mean = sum/count, and use grouped operations for max/min.
3. **For max and min**, sparse matrix multiplication doesn't directly help, but we can use a **data.table join-based approach** that is far faster than R-level lapply.

### Chosen Approach: data.table Edge-List Join

- Convert the `nb` object to an **edge list** (from_cell, to_cell) â€” ~1.37M rows.
- Cross-join with years to get (from_cell, year, to_cell, year) â€” but since years are shared, we just join on (to_cell, year) to pull in the neighbor's value.
- Then group by (from_cell, year) and compute max, min, mean.

This replaces 6.46M R-level iterations with a single vectorized data.table grouped aggregation on ~1.37M Ã— 28 â‰ˆ 38.4M rows â€” easily handled in memory and extremely fast.

**Estimated speedup**: from 86+ hours to **~2â€“5 minutes**.

## Working R Code

```r
library(data.table)
library(spdep)

# ============================================================
# Step 1: Convert nb object to edge list (one-time, fast)
# ============================================================
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (indices into id_order)
  # Build a data.table edge list: from_id -> to_id
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  
  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

# ============================================================
# Step 2: Compute neighbor stats for all variables at once
# ============================================================
compute_all_neighbor_features <- function(cell_data, id_order, nb_obj, 
                                          neighbor_source_vars) {
  
  # Convert to data.table if not already (by reference if possible)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Build edge list: ~1.37M rows
  edges <- build_edge_list(id_order, nb_obj)
  
  # We need to look up neighbor values by (to_id, year).
  # Create a keyed version of cell_data with only needed columns.
  lookup_cols <- c("id", "year", neighbor_source_vars)
  lookup_dt   <- cell_data[, ..lookup_cols]
  setnames(lookup_dt, "id", "to_id")
  setkey(lookup_dt, to_id, year)
  
  # Also need from_id -> year mapping: every from_id appears in every year
  # We get this by joining edges with the distinct (from_id, year) pairs.
  from_years <- cell_data[, .(from_id = id, year)]
  setkey(from_years, from_id)
  
  # Expand edges Ã— years: for each edge (from_id, to_id), 
  # replicate across all years.
  # More efficient: join edges onto from_years by from_id
  setkey(edges, from_id)
  
  # This creates ~1.37M * 28 â‰ˆ 38.4M rows
  edge_years <- edges[from_years, on = .(from_id), allow.cartesian = TRUE, nomatch = NULL]
  # edge_years has columns: from_id, to_id, year
  
  # Now join to get neighbor values
  setkey(edge_years, to_id, year)
  edge_years <- lookup_dt[edge_years, on = .(to_id, year), nomatch = NA]
  # edge_years now has: to_id, year, <var columns>, from_id
  
  # Group by (from_id, year) and compute stats for each variable
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    prefix <- paste0("neighbor_", v)
    agg_exprs[[paste0(prefix, "_max")]]  <- bquote(max(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0(prefix, "_min")]]  <- bquote(min(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0(prefix, "_mean")]] <- bquote(mean(.(v_sym), na.rm = TRUE))
  }
  
  # Build the aggregation call
  # We need to handle the case where all values are NA (max/min return -Inf/Inf)
  # We'll fix that after aggregation.
  
  stats_dt <- edge_years[, 
    lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(NA_real_, NA_real_, NA_real_)
      } else {
        list(max(vals), min(vals), mean(vals))
      }
    }) |> unlist() |> as.list() |> 
      setNames(unlist(lapply(neighbor_source_vars, function(v) {
        paste0("neighbor_", v, c("_max", "_min", "_mean"))
      }))),
    by = .(from_id, year)
  ]
  
  # Merge back onto cell_data
  setnames(stats_dt, "from_id", "id")
  setkey(stats_dt, id, year)
  setkey(cell_data, id, year)
  
  # Remove any pre-existing neighbor columns to avoid duplication
  new_cols <- setdiff(names(stats_dt), c("id", "year"))
  existing <- intersect(names(cell_data), new_cols)
  if (length(existing) > 0) {
    cell_data[, (existing) := NULL]
  }
  
  cell_data <- stats_dt[cell_data, on = .(id, year)]
  
  return(cell_data)
}

# ============================================================
# Optimized version of the grouped aggregation 
# (avoiding get() inside j for performance)
# ============================================================
compute_all_neighbor_features_fast <- function(cell_data, id_order, nb_obj,
                                               neighbor_source_vars) {
  
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(copy(cell_data))
  }
  
  edges <- build_edge_list(id_order, nb_obj)
  
  # --- Build edge-year table ---
  from_years <- unique(cell_data[, .(from_id = id, year)])
  setkey(edges, from_id)
  setkey(from_years, from_id)
  
  edge_years <- edges[from_years, on = .(from_id), allow.cartesian = TRUE, nomatch = NULL]
  
  # --- Join neighbor values ---
  lookup_cols <- c("id", "year", neighbor_source_vars)
  lookup_dt   <- cell_data[, ..lookup_cols]
  setnames(lookup_dt, "id", "to_id")
  setkey(lookup_dt, to_id, year)
  setkey(edge_years, to_id, year)
  
  edge_years <- lookup_dt[edge_years, on = .(to_id, year), nomatch = NA]
  
  # --- Compute stats per variable separately, then merge ---
  setkey(edge_years, from_id, year)
  
  result_dt <- unique(edge_years[, .(from_id, year)])
  setkey(result_dt, from_id, year)
  
  for (v in neighbor_source_vars) {
    cat("Computing neighbor stats for:", v, "\n")
    
    # Efficient aggregation using data.table's optimized grouping
    # Suppress warnings for max/min on empty sets
    agg <- edge_years[!is.na(get(v)), {
      vals <- get(v)
      .(
        nmax  = max(vals),
        nmin  = min(vals),
        nmean = mean(vals)
      )
    }, by = .(from_id, year)]
    
    new_names <- paste0("neighbor_", v, c("_max", "_min", "_mean"))
    setnames(agg, c("nmax", "nmin", "nmean"), new_names)
    setkey(agg, from_id, year)
    
    result_dt <- agg[result_dt, on = .(from_id, year)]
  }
  
  # --- Merge back ---
  setnames(result_dt, "from_id", "id")
  setkey(result_dt, id, year)
  
  new_cols <- setdiff(names(result_dt), c("id", "year"))
  existing <- intersect(names(cell_data), new_cols)
  if (length(existing) > 0) {
    cell_data[, (existing) := NULL]
  }
  
  setkey(cell_data, id, year)
  cell_data <- result_dt[cell_data, on = .(id, year)]
  
  return(cell_data)
}

# ============================================================
# EVEN FASTER: avoid get() entirely by melting + dcasting
# This is the recommended version.
# ============================================================
compute_all_neighbor_features_fastest <- function(cell_data, id_order, nb_obj,
                                                  neighbor_source_vars) {
  
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(copy(cell_data))
  }
  
  edges <- build_edge_list(id_order, nb_obj)
  cat("Edge list:", nrow(edges), "directed edges\n")
  
  # --- Build edge-year table ---
  from_years <- unique(cell_data[, .(from_id = id, year)])
  setkey(edges, from_id)
  
  # ~38.4M rows
  edge_years <- edges[from_years, on = .(from_id), allow.cartesian = TRUE, nomatch = NULL]
  cat("Edge-year table:", nrow(edge_years), "rows\n")
  
  # --- Process one variable at a time to control memory ---
  setkey(cell_data, id, year)
  
  for (v in neighbor_source_vars) {
    cat("Processing:", v, "... ")
    t0 <- proc.time()
    
    # Extract only needed columns for lookup
    val_dt <- cell_data[, .(to_id = id, year, val = get(v))]
    setkey(val_dt, to_id, year)
    
    # Join neighbor values onto edge-year table
    ey <- val_dt[edge_years, on = .(to_id, year), nomatch = NA]
    # ey has: to_id, year, val, from_id
    
    # Aggregate: group by (from_id, year)
    agg <- ey[!is.na(val), .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ), keyby = .(from_id, year)]
    
    # Rename columns
    new_names <- paste0("neighbor_", v, c("_max", "_min", "_mean"))
    setnames(agg, c("nmax", "nmin", "nmean"), new_names)
    setnames(agg, "from_id", "id")
    
    # Remove old columns if they exist
    existing <- intersect(names(cell_data), new_names)
    if (length(existing) > 0) cell_data[, (existing) := NULL]
    
    # Merge
    cell_data <- agg[cell_data, on = .(id, year)]
    setkey(cell_data, id, year)
    
    elapsed <- (proc.time() - t0)[3]
    cat(round(elapsed, 1), "seconds\n")
    
    # Clean up
    rm(val_dt, ey, agg)
    gc(verbose = FALSE)
  }
  
  return(cell_data)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

# Original code (86+ hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement (~2-5 minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_fastest(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” 
# the output columns have identical names and identical numerical values.
# No retraining is needed.
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Data structure** | 6.46M-element R list of index vectors | ~1.37M-row edge list (data.table) |
| **Lookup method** | `paste` + named-vector character match | Keyed data.table binary-search join |
| **Aggregation** | R-level `lapply` over 6.46M rows Ã— 5 vars | data.table grouped `max/min/mean` on ~38.4M rows |
| **Estimated time** | 86+ hours | ~2â€“5 minutes |
| **Peak RAM** | ~6â€“10 GB (list overhead) | ~4â€“6 GB (data.table, one var at a time) |
| **Numerical result** | Exact | **Identical** (same max, min, mean) |
| **RF model** | â€” | **Preserved, no retraining** |

**Why it's exact**: The edge list is constructed directly from the same `nb` object (`rook_neighbors_unique`) and the same `id_order`. The join on `(to_id, year)` replicates exactly the same neighbor-value retrieval as the original `idx_lookup[neighbor_keys]`. The `max`, `min`, and `mean` functions are applied to the same sets of non-NA neighbor values. The numerical estimand is preserved bit-for-bit (up to floating-point associativity of `mean`, which data.table computes in the same summation order within each group).
END_RESPONSE>>>
