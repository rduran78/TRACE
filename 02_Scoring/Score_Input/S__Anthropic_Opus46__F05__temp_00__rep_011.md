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
run_id: Anthropic_F05_T00_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” repeated string allocation and concatenation.
2. **Looks up those keys** in `idx_lookup`, a named character vector of length 6.46M â€” each lookup is O(N) in the worst case because R named vector lookup uses linear hashing that degrades at this scale.

This means `build_neighbor_lookup` alone performs ~6.46M Ã— (avg ~8 neighbors) â‰ˆ **51.7 million string constructions and hash lookups against a 6.46M-entry table**. That's the 86+ hour bottleneck.

### Why It's a Broader Algorithmic Problem

The string-keying approach **conflates two independent indexing dimensions** (cell ID and year) into a single flat string key. But the neighbor structure is **time-invariant** â€” cell A's neighbors don't change from year to year. The algorithm re-discovers the same spatial relationships 28 times (once per year), just with different string suffixes.

The correct algorithmic insight: **separate the spatial lookup from the temporal lookup**. Build the neighbor graph once over the 344K cells, then for each year, use integer indexing to gather neighbor rows.

Similarly, `compute_neighbor_stats` is fine algorithmically but can be vectorized using matrix operations instead of per-row `lapply`.

---

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Neighbor resolution | String paste + named-vector lookup per row (6.46M iterations) | Integer-indexed spatial neighbor list (344K cells), broadcast across years via offset arithmetic |
| Per-variable stats | `lapply` over 6.46M rows, each extracting a small vector | Vectorized sparse-matrix multiplication or `data.table` grouped aggregation |
| Complexity | O(rows Ã— avg_neighbors Ã— string_ops) | O(cells Ã— avg_neighbors) one-time setup + O(rows Ã— avg_neighbors) integer arithmetic |
| Estimated time | 86+ hours | **Minutes** |

### Key Invariant Preserved

The numerical estimand is identical: for each cell-year row, we compute `max`, `min`, and `mean` of each neighbor source variable across the rook neighbors that exist in that year. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 1: Build a fast integer-indexed spatial neighbor list (done ONCE)
# =============================================================================
# Inputs:
#   cell_data    â€” data.frame/data.table with columns: id, year, ntl, ec, ...
#   id_order     â€” integer vector of cell IDs in the order matching rook_neighbors_unique
#   rook_neighbors_unique â€” spdep nb object (list of integer index vectors)

build_neighbor_lookup_fast <- function(cell_data, id_order, rook_neighbors) {
  # Convert to data.table if needed (non-destructive)
  dt <- as.data.table(cell_data)
  
  # --- Spatial neighbor list keyed by cell id (time-invariant) ---
  # rook_neighbors[[i]] gives neighbor indices into id_order for cell id_order[i]
  # Convert to a list keyed by cell ID -> vector of neighbor cell IDs
  spatial_neighbors <- setNames(
    lapply(seq_along(id_order), function(i) {
      nb_idx <- rook_neighbors[[i]]
      # spdep nb: 0 means no neighbors
      nb_idx <- nb_idx[nb_idx != 0L]
      if (length(nb_idx) == 0L) return(integer(0))
      id_order[nb_idx]
    }),
    as.character(id_order)
  )
  
  # --- Build a fast (id, year) -> row-index lookup using data.table ---
  dt[, .row_idx := .I]
  setkey(dt, id, year)
  
  # --- For each row, find the row indices of its spatial neighbors in the same year ---
  # Strategy: expand the neighbor relationships and join, all vectorized.
  
  # Create an edge table: for each cell, its neighbor cell IDs
  # This is time-invariant, so we build it once over unique cells
  unique_ids <- id_order
  
  # Build edge list from spatial_neighbors
  from_ids <- rep(
    unique_ids,
    times = vapply(spatial_neighbors[as.character(unique_ids)], length, integer(1))
  )
  to_ids <- unlist(spatial_neighbors[as.character(unique_ids)], use.names = FALSE)
  
  edges <- data.table(from_id = from_ids, to_id = to_ids)
  
  list(dt = dt, edges = edges)
}

# =============================================================================
# STEP 2: Compute neighbor stats for all variables at once (vectorized)
# =============================================================================
compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors,
                                          neighbor_source_vars) {
  dt <- as.data.table(copy(cell_data))
  dt[, .row_idx := .I]
  
  # --- Build spatial edge list (time-invariant) ---
  cat("Building spatial edge list...\n")
  n_cells <- length(id_order)
  
  from_ids <- integer(0)
  to_ids   <- integer(0)
  
  # Vectorized construction of edge list
  nb_lengths <- vapply(rook_neighbors, function(x) {
    sum(x != 0L)
  }, integer(1))
  
  total_edges <- sum(nb_lengths)
  from_ids <- rep(id_order, times = nb_lengths)
  
  to_ids <- unlist(lapply(seq_len(n_cells), function(i) {
    nb_idx <- rook_neighbors[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(integer(0))
    id_order[nb_idx]
  }), use.names = FALSE)
  
  edges <- data.table(from_id = from_ids, to_id = to_ids)
  cat(sprintf("  Edge list: %d directed edges\n", nrow(edges)))
  
  # --- Expand edges across all years via join ---
  # For each (from_id, year) row, find the row indices of (to_id, year)
  # 
  # Instead of expanding edges Ã— years (which would be ~38M rows),
  # we join edges with the data twice:
  #   1. Join edges with dt on from_id to get the year and the "source row"
  #   2. Join the result with dt on (to_id, year) to get the "neighbor row"
  
  cat("Building row-index lookup...\n")
  # Lookup table: (id, year) -> row_idx
  lookup <- dt[, .(id, year, .row_idx)]
  setkey(lookup, id, year)
  
  # Step A: For each edge, cross with all years that from_id appears in
  cat("Joining edges with source rows...\n")
  # from_lookup: all (from_id, year, from_row_idx)
  from_lookup <- lookup[, .(from_id = id, year, from_row = .row_idx)]
  setkey(from_lookup, from_id)
  setkey(edges, from_id)
  
  # Merge: for each edge (from_id -> to_id), get all years from_id appears
  # This gives us (from_id, to_id, year, from_row)
  edge_year <- edges[from_lookup, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year columns: from_id, to_id, year, from_row
  
  cat(sprintf("  Edge-year pairs: %d\n", nrow(edge_year)))
  
  # Step B: Join with neighbor rows to get to_row
  cat("Joining with neighbor rows...\n")
  to_lookup <- lookup[, .(to_id = id, year, to_row = .row_idx)]
  setkey(to_lookup, to_id, year)
  setkey(edge_year, to_id, year)
  
  edge_year <- edge_year[to_lookup, on = c("to_id", "year"), nomatch = 0L]
  # Now edge_year has: from_id, to_id, year, from_row, to_row
  
  cat(sprintf("  Matched edge-year pairs: %d\n", nrow(edge_year)))
  
  # --- Compute stats per variable ---
  cat("Computing neighbor statistics...\n")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    
    # Extract neighbor values via integer indexing (vectorized)
    edge_year[, nbr_val := dt[[var_name]][to_row]]
    
    # Remove NAs for aggregation
    valid <- edge_year[!is.na(nbr_val)]
    
    # Aggregate by from_row (= the source cell-year row)
    stats <- valid[, .(
      nb_max  = max(nbr_val),
      nb_min  = min(nbr_val),
      nb_mean = mean(nbr_val)
    ), by = from_row]
    
    # Initialize new columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign results via integer indexing
    dt[stats$from_row, (max_col)  := stats$nb_max]
    dt[stats$from_row, (min_col)  := stats$nb_min]
    dt[stats$from_row, (mean_col) := stats$nb_mean]
  }
  
  # Clean up helper column
  dt[, .row_idx := NULL]
  edge_year[, nbr_val := NULL]  # free memory
  
  cat("Done.\n")
  return(dt)
}

# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# The result is a data.table with the same columns as before:
#   ntl_nb_max, ntl_nb_min, ntl_nb_mean,
#   ec_nb_max,  ec_nb_min,  ec_nb_mean,
#   ... etc.
#
# The trained Random Forest model can be used directly on this output
# with no retraining â€” the features are numerically identical.
```

---

## Memory-Conscious Variant

If the `edge_year` table (~38.5M rows Ã— 5 columns) strains the 16 GB RAM, process **year-by-year** to keep peak memory low:

```r
compute_all_neighbor_features_lowmem <- function(cell_data, id_order, rook_neighbors,
                                                  neighbor_source_vars) {
  dt <- as.data.table(copy(cell_data))
  dt[, .row_idx := .I]
  
  # Build spatial edge list (time-invariant)
  cat("Building spatial edge list...\n")
  n_cells <- length(id_order)
  nb_lengths <- vapply(rook_neighbors, function(x) sum(x != 0L), integer(1))
  from_ids <- rep(id_order, times = nb_lengths)
  to_ids <- unlist(lapply(seq_len(n_cells), function(i) {
    nb_idx <- rook_neighbors[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(integer(0))
    id_order[nb_idx]
  }), use.names = FALSE)
  edges <- data.table(from_id = from_ids, to_id = to_ids)
  
  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_nb_max")  := NA_real_]
    dt[, paste0(var_name, "_nb_min")  := NA_real_]
    dt[, paste0(var_name, "_nb_mean") := NA_real_]
  }
  
  # Process year by year to limit memory
  years <- sort(unique(dt$year))
  setkey(dt, year, id)
  
  for (yr in years) {
    cat(sprintf("  Year %d...\n", yr))
    
    # Subset rows for this year
    yr_dt <- dt[year == yr, c("id", ".row_idx", neighbor_source_vars), with = FALSE]
    setkey(yr_dt, id)
    
    # Join edges with this year's data
    # from side
    from_yr <- yr_dt[, .(from_id = id, from_row = .row_idx)]
    setkey(from_yr, from_id)
    matched <- edges[from_yr, on = c("from_id"), nomatch = 0L]
    
    # to side
    to_yr <- yr_dt[, .(to_id = id, to_row = .row_idx)]
    setkey(to_yr, to_id)
    setkey(matched, to_id)
    matched <- matched[to_yr, on = c("to_id"), nomatch = 0L]
    # matched: from_id, to_id, from_row, to_row
    
    for (var_name in neighbor_source_vars) {
      matched[, nbr_val := dt[[var_name]][to_row]]
      valid <- matched[!is.na(nbr_val)]
      
      stats <- valid[, .(
        nb_max  = max(nbr_val),
        nb_min  = min(nbr_val),
        nb_mean = mean(nbr_val)
      ), by = from_row]
      
      if (nrow(stats) > 0L) {
        set(dt, i = stats$from_row, j = paste0(var_name, "_nb_max"),  value = stats$nb_max)
        set(dt, i = stats$from_row, j = paste0(var_name, "_nb_min"),  value = stats$nb_min)
        set(dt, i = stats$from_row, j = paste0(var_name, "_nb_mean"), value = stats$nb_mean)
      }
      matched[, nbr_val := NULL]
    }
  }
  
  dt[, .row_idx := NULL]
  cat("Done.\n")
  return(dt)
}
```

---

## Performance Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| String constructions | ~51.7M inside loop | **0** |
| Named-vector lookups (6.46M entries) | ~51.7M | **0** (integer joins) |
| R-level loop iterations | 6.46M (`lapply`) | 28 (year loop) or 0 |
| Core operation | Character hashing | `data.table` keyed join (radix-based) |
| Estimated wall time | **86+ hours** | **5â€“15 minutes** |
| RAM (peak, low-mem variant) | ~2â€“4 GB | ~3â€“5 GB |
| Numerical output | Baseline | **Identical** |
| RF model impact | â€” | **None (no retraining)** |

The fundamental fix is **eliminating the per-row string-key construction entirely** by recognizing that the neighbor graph is time-invariant and can be expressed as an integer edge list, then joined against each year's data using `data.table`'s optimized keyed merge.
END_RESPONSE>>>
