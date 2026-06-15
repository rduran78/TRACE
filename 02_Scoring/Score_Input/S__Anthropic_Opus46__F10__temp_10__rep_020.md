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
run_id: Anthropic_F10_T10_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

### 1. **O(NÂ²)-Scale Neighbor Lookup Construction**
`build_neighbor_lookup` iterates over all ~6.46 million cell-year rows and, for each row, performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a vector of length n, making this effectively O(NÂ²) where N â‰ˆ 6.46M. The `paste(..., sep="_")` key construction for ~6.46M rows is also expensive.

### 2. **Row-Wise `lapply` in `compute_neighbor_stats`**
For each of the ~6.46M rows, the function subsets a numeric vector by index, removes NAs, and computes three summary statistics. The `lapply` + `do.call(rbind, ...)` pattern over 6.46M elements is extremely slow due to R's per-call overhead and memory allocation.

### 3. **Redundant Topology Replication Across Years**
The spatial graph topology (rook neighbors) is **identical across all 28 years**. Yet the current code builds a lookup that embeds the year dimension into every key, duplicating the adjacency structure 28 times. The neighbor relationships are between **cells**, not cell-years â€” only the **attribute values** change by year.

### Summary
- ~6.46M string-paste + named-vector lookups â†’ hours of string hashing
- ~6.46M Ã— 5 variables Ã— `lapply` calls with per-element R overhead â†’ hours of row-wise iteration
- Adjacency structure duplicated 28Ã— unnecessarily

---

## Optimization Strategy

### Principle: Separate Topology from Temporal Attributes

1. **Build the sparse adjacency matrix once** (344,208 Ã— 344,208) from the `nb` object. This is a sparse matrix with ~1.37M non-zero entries â€” trivial in memory (~16 MB).

2. **Reshape each variable into a cell Ã— year matrix** (344,208 Ã— 28). This is ~77 MB per variable in dense form.

3. **Use sparse matrix multiplication** to compute neighbor sums and counts in one shot, then derive max/min/mean via vectorized operations. For **mean**: `A %*% X / A %*% (non-NA indicator)`. For **max and min**: iterate over the sparse structure but in C++ via `Rcpp`, or use grouped operations with `data.table`.

4. **For max and min** (which are not expressible as linear algebra), use `data.table` grouped operations on an edge-list representation â€” this replaces 6.46M `lapply` calls with a single vectorized grouped aggregation.

### Expected Speedup
- Adjacency built once: seconds instead of hours
- `data.table` grouped aggregation: ~seconds per variable-year
- Total: **minutes** instead of 86+ hours

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Sparse graph neighborhood aggregation via data.table edge-list joins
# Numerically equivalent to the original implementation
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, 
                                        id_order, 
                                        rook_neighbors_unique, 
                                        neighbor_source_vars) {
  
  # -------------------------------------------------------------------------
  # STEP 1: Convert cell_data to data.table if needed (in-place, no copy)
  # -------------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    setDT(cell_data)
  }
  
  # -------------------------------------------------------------------------
  # STEP 2: Build the edge list ONCE from the nb object
  #
  # rook_neighbors_unique is an nb object: a list of length = length(id_order),

  # where each element is an integer vector of neighbor indices (into id_order).
  # We convert this to an edge list of (from_id, to_id) in terms of cell IDs.
  # -------------------------------------------------------------------------
  cat("Building edge list from nb object...\n")
  
  n_cells <- length(id_order)
  
  # Pre-compute the number of neighbors for each cell to pre-allocate
  n_neighbors <- vapply(rook_neighbors_unique, length, integer(1))
  total_edges <- sum(n_neighbors)
  
  # Pre-allocate edge list vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- rook_neighbors_unique[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      idx_range <- pos:(pos + n_nb - 1L)
      from_id[idx_range] <- id_order[i]
      to_id[idx_range]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  
  # Edge list: for each directed edge, from_id -> to_id means
  # "to_id is a rook neighbor of from_id"
  # When computing neighbor stats for from_id, we need attributes of to_id.
  edges <- data.table(from_id = from_id, to_id = to_id)
  
  cat(sprintf("  Edge list: %d directed edges across %d cells\n", 
              nrow(edges), n_cells))
  
  # -------------------------------------------------------------------------
  # STEP 3: Create a cell-year keyed lookup for fast joins
  # -------------------------------------------------------------------------
  # We need: for each (from_id, year), gather the variable values of all 
  # (to_id, year) neighbors, then compute max, min, mean.
  #
  # Strategy: 
  #   - For each variable, join edges with cell_data on (to_id, year)
  #     to get neighbor values
  #   - Group by (from_id, year) to compute aggregates
  #   - Join results back to cell_data
  # -------------------------------------------------------------------------
  
  # Ensure id and year columns exist and set key for fast joins
  # The neighbor values come from looking up (to_id, year) in cell_data
  # So we key cell_data by (id, year)
  setkeyv(cell_data, c("id", "year"))
  
  # Get all unique years
  all_years <- sort(unique(cell_data$year))
  cat(sprintf("  Years: %d (%d to %d)\n", 
              length(all_years), min(all_years), max(all_years)))
  
  # -------------------------------------------------------------------------
  # STEP 4: For each variable, compute neighbor stats via vectorized joins
  # -------------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    
    cat(sprintf("Processing variable: %s\n", var_name))
    t0 <- proc.time()
    
    # Column names for the output (must match original pipeline's naming)
    col_max  <- paste0("n_max_", var_name)
    col_min  <- paste0("n_min_", var_name)
    col_mean <- paste0("n_mean_", var_name)
    
    # Extract only (id, year, var_name) for the join target â€” minimal memory
    # This is the "node attribute" table
    attr_dt <- cell_data[, .(id, year, val = get(var_name))]
    setkey(attr_dt, id, year)
    
    # -----------------------------------------------------------------------
    # Cross edges with years: for each year, the same edge list applies.
    # Instead of replicating edges Ã— years (which would be ~38M rows),
    # we join edges to the attribute table directly.
    #
    # For each edge (from_id, to_id), and for each year in the data,
    # we need val[to_id, year]. We do this by:
    #   1. Cross-join edges with all_years â†’ edge_year table
    #   2. Join edge_year with attr_dt on (to_id = id, year) to get neighbor val
    #   3. Group by (from_id, year) â†’ compute max, min, mean
    #
    # Optimization: instead of cross-joining (which creates ~38M rows),
    # we process year-by-year in a loop. Each year has ~1.37M edges.
    # This keeps peak memory low and is still very fast with data.table.
    # -----------------------------------------------------------------------
    
    # Pre-allocate result list
    result_list <- vector("list", length(all_years))
    
    for (yi in seq_along(all_years)) {
      yr <- all_years[yi]
      
      # Get attribute values for this year: (id, val)
      attr_year <- attr_dt[year == yr, .(id, val)]
      setkey(attr_year, id)
      
      # Join: for each edge, look up the neighbor's value
      # edges[, .(from_id, to_id)] joined with attr_year on to_id = id
      edge_vals <- attr_year[edges, on = .(id = to_id), nomatch = NA,
                             .(from_id = i.from_id, val = x.val)]
      
      # Group by from_id, compute stats (excluding NAs, matching original)
      stats <- edge_vals[!is.na(val), 
                         .(n_max  = max(val),
                           n_min  = min(val),
                           n_mean = mean(val)),
                         by = from_id]
      
      # Add year for later joining
      stats[, year := yr]
      
      result_list[[yi]] <- stats
    }
    
    # Combine all years
    all_stats <- rbindlist(result_list, use.names = TRUE)
    setkey(all_stats, from_id, year)
    
    # Rename columns to match expected output
    setnames(all_stats, 
             c("n_max", "n_min", "n_mean"),
             c(col_max, col_min, col_mean))
    setnames(all_stats, "from_id", "id")
    
    # -----------------------------------------------------------------------
    # Join results back to cell_data
    # For cells with no valid neighbors in a given year, they won't appear
    # in all_stats â†’ they get NA, matching the original c(NA, NA, NA) behavior.
    # -----------------------------------------------------------------------
    
    # Remove old columns if they exist (for idempotency)
    for (cc in c(col_max, col_min, col_mean)) {
      if (cc %in% names(cell_data)) {
        cell_data[, (cc) := NULL]
      }
    }
    
    # Merge
    cell_data <- all_stats[cell_data, on = .(id, year)]
    setkeyv(cell_data, c("id", "year"))
    
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  Done in %.1f seconds\n", elapsed))
  }
  
  cat("All neighbor features computed.\n")
  return(cell_data)
}


# =============================================================================
# USAGE
# =============================================================================

# # Load pre-existing objects:
# # - cell_data: data.frame/data.table with columns id, year, ntl, ec, 
# #              pop_density, def, usd_est_n2, ... (~6.46M rows)
# # - id_order: integer vector of cell IDs (length 344,208)
# # - rook_neighbors_unique: nb object (list of integer neighbor indices)
# # - rf_model: pre-trained randomForest model object (DO NOT retrain)
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- optimize_neighbor_features(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # Predict with the existing trained Random Forest (no retraining)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` subsets neighbor values | `attr_year[edges, on = .(id = to_id)]` joins neighbor values | Same values selected |
| `neighbor_vals[!is.na(neighbor_vals)]` removes NAs | `edge_vals[!is.na(val), ...]` filters NAs before grouping | Identical NA handling |
| `max(neighbor_vals)` | `max(val)` in `data.table` grouped `j` | Identical (IEEE 754) |
| `min(neighbor_vals)` | `min(val)` in `data.table` grouped `j` | Identical (IEEE 754) |
| `mean(neighbor_vals)` | `mean(val)` in `data.table` grouped `j` | Identical (sum/n, same precision) |
| No neighbors â†’ `c(NA, NA, NA)` | No match in `all_stats` â†’ `NA` from join | Identical |

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Adjacency construction | ~6.46M string-paste + named lookups | 1 loop over 344K cells â†’ edge list |
| Per-variable aggregation | 6.46M `lapply` calls | 28 Ã— `data.table` grouped-agg on ~1.37M edges |
| Memory for adjacency | ~6.46M-element list of integer vectors | One 2-column integer `data.table` (~11 MB) |
| Estimated total time | **86+ hours** | **~5â€“15 minutes** |
| Random Forest model | Preserved (not retrained) | Preserved (not retrained) |
| Numerical results | Original | **Identical** |
END_RESPONSE>>>
