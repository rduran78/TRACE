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
run_id: Anthropic_F05_T07_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Layer 1: Redundant string construction inside `build_neighbor_lookup`
- `idx_lookup` is built once (good), but `paste(neighbor_cell_ids, data$year[i], sep = "_")` is called **6.46 million times** inside `lapply`. Each call constructs string keys, hashes them, and probes a named vector. With ~4 neighbors per cell on average, that's ~25.8 million `paste` + hash-probe operations.

### Layer 2: The real bottleneck â€” an R-level row-by-row `lapply` over 6.46M rows
- `build_neighbor_lookup` iterates row-by-row in R (not vectorized, not compiled). Even if each iteration is fast, 6.46M iterations in interpreted R with hash lookups is catastrophically slow.

### Layer 3: The neighbor lookup is year-invariant but recomputed per cell-year
- Rook neighbors are a **spatial** relationship: cell A's neighbors are the same in 1992 as in 2019. The current code re-derives the neighbor mapping for every cell-year row. For 344,208 cells Ã— 28 years, the same spatial lookup is repeated 28 times per cell.

### Layer 4: `compute_neighbor_stats` also loops row-by-row
- After building the lookup, stats are computed via another 6.46M-iteration `lapply`.

### Summary
| Problem | Scale |
|---|---|
| String-key hashing per row | 6.46M Ã— ~4 neighbors |
| R-level `lapply` in `build_neighbor_lookup` | 6.46M iterations |
| Redundant year duplication of spatial topology | 28Ã— overhead |
| R-level `lapply` in `compute_neighbor_stats` | 6.46M iterations Ã— 5 variables |

## Optimization Strategy

**Core insight:** Separate the spatial topology (which cell neighbors which) from the temporal panel (which rows correspond to which year). Then use vectorized/`data.table` operations instead of row-by-row R loops.

1. **Build the neighbor edge list once** â€” a two-column integer matrix `(cell_i, cell_j)` with ~1.37M directed edges. This is year-invariant.
2. **Join panel data onto the edge list by year** â€” for each variable, use `data.table` keyed joins to pull neighbor values, then compute grouped `max/min/mean` in one vectorized pass.
3. **No string keys, no row-by-row `lapply`, no 28Ã— redundancy.**

Expected speedup: from ~86 hours to **minutes**.

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build the directed neighbor edge list ONCE (year-invariant)
#
#   rook_neighbors_unique : spdep nb object (list of integer index vectors)
#   id_order              : integer vector mapping nb-list position -> cell id
#
#   Output: data.table with columns  focal_id, neighbor_id
# ===========================================================================

build_neighbor_edge_list <- function(id_order, neighbors) {
  # neighbors[[k]] gives the nb-list indices of the neighbors of id_order[k]
  n <- length(neighbors)
  
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)
  
  pos <- 1L

  for (k in seq_len(n)) {
    nb_idx <- neighbors[[k]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    len <- length(nb_idx)
    focal_id[pos:(pos + len - 1L)]    <- id_order[k]
    neighbor_id[pos:(pos + len - 1L)] <- id_order[nb_idx]
    pos <- pos + len
  }
  
  # Trim if any nb entries were empty (0-neighbor cells)
  if (pos <= n_edges) {
    focal_id    <- focal_id[1:(pos - 1L)]
    neighbor_id <- neighbor_id[1:(pos - 1L)]
  }
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

# ===========================================================================
# STEP 2: Vectorized neighbor-stat computation via data.table joins
#
#   For each variable, join the edge list with the panel on
#   (neighbor_id, year) to retrieve neighbor values, then group by
#   (focal_id, year) to compute max, min, mean.
# ===========================================================================

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_names) {
  # Ensure data.table
  if (!is.data.table(cell_dt)) cell_dt <- as.data.table(cell_dt)
  
  # Key the panel for fast joins
  setkey(cell_dt, id, year)
  
  # We need (focal_id, year) pairs. Get the unique years from the panel.
  # Cross-join edges Ã— years to get the full (focal, neighbor, year) set.
  # BUT: that would be 1.37M Ã— 28 = 38.5M rows â€” manageable.
  #
  # More memory-efficient: join edges onto the panel's (id, year) rows.
  
  # Build a lookup: for each row in cell_dt, get its (id, year)
  # Then join to edge_dt to expand to neighbor rows.
  
  # Panel keyed on id, year â€” we join neighbor values directly.
  
  # For each variable:
  for (var_name in var_names) {
    message("Processing neighbor stats for: ", var_name)
    
    # Subset to needed columns for the join (minimize memory)
    # neighbor_id will be looked up in the panel by (neighbor_id, year)
    val_dt <- cell_dt[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)
    
    # Also need (focal_id, year) from the panel to know which years exist
    focal_years <- cell_dt[, .(focal_id = id, year)]
    
    # Merge focal_years with edge_dt to get (focal_id, neighbor_id, year)
    # This is the cross of edges Ã— years, filtered to existing focal rows.
    expanded <- merge(focal_years, edge_dt, by = "focal_id", allow.cartesian = TRUE)
    # expanded has columns: focal_id, year, neighbor_id
    # ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows (fits in 16GB easily)
    
    # Now join to get the neighbor's value in that year
    expanded[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]
    
    # Group by (focal_id, year) and compute stats, dropping NAs
    stats <- expanded[!is.na(neighbor_val),
                      .(nb_max  = max(neighbor_val),
                        nb_min  = min(neighbor_val),
                        nb_mean = mean(neighbor_val)),
                      by = .(focal_id, year)]
    
    # Rename columns to match original naming convention
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))
    
    # Merge back into cell_dt
    # First remove old columns if they exist (idempotent re-runs)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    }
    
    cell_dt <- merge(cell_dt, stats,
                     by.x = c("id", "year"),
                     by.y = c("focal_id", "year"),
                     all.x = TRUE)
    
    # Clean up
    rm(val_dt, focal_years, expanded, stats)
    gc()
  }
  
  cell_dt
}

# ===========================================================================
# STEP 3: Main execution â€” drop-in replacement for the original outer loop
# ===========================================================================

# Convert to data.table if needed
cell_data <- as.data.table(cell_data)

# Build edge list once (~1.37M rows, a few MB)
edge_dt <- build_neighbor_edge_list(id_order, rook_neighbors_unique)

# Compute all neighbor features (vectorized)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_fast(cell_data, edge_dt, neighbor_source_vars)

# cell_data now has the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns as the original code, with identical numerical values.
# The trained Random Forest model can be used directly for prediction.
```

## Memory-Optimized Variant (if 38.5M-row expanded table is tight on 16GB)

If the full expanded join strains memory, process one variable at a time **and** chunk by year:

```r
compute_neighbor_features_chunked <- function(cell_dt, edge_dt, var_names) {
  if (!is.data.table(cell_dt)) cell_dt <- as.data.table(cell_dt)
  setkey(cell_dt, id, year)
  
  all_years <- sort(unique(cell_dt$year))
  
  for (var_name in var_names) {
    message("Processing: ", var_name)
    
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # Pre-allocate result columns
    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]
    
    for (yr in all_years) {
      # Subset to this year
      yr_dt <- cell_dt[year == yr, .(id, val = get(var_name))]
      setkey(yr_dt, id)
      
      # Expand edges: every edge applies to this year
      # edge_dt has ~1.37M rows â€” very manageable
      expanded <- copy(edge_dt)
      expanded[yr_dt, neighbor_val := i.val, on = .(neighbor_id = id)]
      
      stats <- expanded[!is.na(neighbor_val),
                        .(nb_max  = max(neighbor_val),
                          nb_min  = min(neighbor_val),
                          nb_mean = mean(neighbor_val)),
                        by = .(focal_id)]
      
      # Write back into cell_dt
      idx <- cell_dt[year == yr, which = TRUE]
      match_pos <- match(cell_dt$id[idx], stats$focal_id)
      
      set(cell_dt, i = idx, j = max_col,  value = stats$nb_max[match_pos])
      set(cell_dt, i = idx, j = min_col,  value = stats$nb_min[match_pos])
      set(cell_dt, i = idx, j = mean_col, value = stats$nb_mean[match_pos])
      
      rm(yr_dt, expanded, stats)
    }
    gc()
  }
  
  cell_dt
}

# Usage (drop-in replacement):
cell_data <- compute_neighbor_features_chunked(cell_data, edge_dt, neighbor_source_vars)
```

## Complexity Comparison

| | Original | Optimized (full) | Optimized (chunked) |
|---|---|---|---|
| **Outer iterations** | 6.46M rows Ã— 5 vars | 5 vars | 28 years Ã— 5 vars |
| **Inner work per iter** | ~4 paste + hash ops | Vectorized join | Vectorized join |
| **String hashing** | ~129M paste calls | **Zero** | **Zero** |
| **Total operations** | O(N Ã— k) interpreted R | O(E Ã— T) vectorized C | O(E) vectorized C Ã— T |
| **Estimated time** | ~86 hours | **~2â€“5 minutes** | **~5â€“10 minutes** |
| **Peak RAM** | Moderate (but slow) | ~3â€“4 GB for expanded | ~200 MB per chunk |

## Numerical Equivalence Guarantee

The optimized code computes **exactly** `max`, `min`, and `mean` of the same neighbor values selected by the same spatial topology and year matching. The only change is algorithmic: replacing interpreted R loops with vectorized `data.table` joins and grouped aggregations. The trained Random Forest model requires no modification â€” the output columns are numerically identical.
END_RESPONSE>>>
