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
run_id: Anthropic_F06_T00_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine â€” O(1) hash lookup).
2. Constructs `paste(neighbor_id, year)` key strings for every neighbor (expensive string allocation Ã— ~4 neighbors Ã— 6.46M rows â‰ˆ 25.8M string constructions).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` â€” named vector indexing in R is an O(n) hash probe per call, but the real cost is that `idx_lookup` is a named vector of length 6.46M, and this is done 6.46M times.

**Result:** `neighbor_lookup` is a list of 6.46M integer vectors. Building it takes hours and consumes enormous memory (each list element has R overhead of ~200+ bytes, so 6.46M elements â‰ˆ 1.2â€“2 GB just in list overhead, plus the integer vectors).

### Bottleneck B: `compute_neighbor_stats` â€” `lapply` over 6.46M elements, repeated 5 times

For each of the 5 variables, the code iterates over 6.46M list elements, subsets a numeric vector, removes NAs, and computes `max/min/mean`. The per-element R interpreter overhead (function call, subsetting, `is.na`, three summary functions) dominates. 5 variables Ã— 6.46M rows Ã— ~5 R function calls each â‰ˆ 160M interpreted R operations.

### Why raster focal/kernel operations are the right analogy but wrong implementation

The data is a **panel on an irregular spatial grid** (not a regular raster), so `terra::focal()` or `raster::focal()` cannot be applied directly. However, the *concept* is identical: for each cell, aggregate neighbor values. The efficient implementation is **sparse matrix multiplication / aggregation**, which is the generalization of focal operations to irregular grids.

---

## 2. Optimization Strategy

### Key Insight: Replace per-row R loops with vectorized sparse-matrix operations

A rook-neighbor aggregation (max, min, mean) can be computed as follows:

1. **Build a sparse adjacency matrix `W`** of dimension `N_cells Ã— N_cells` (344,208 Ã— 344,208) from the `nb` object â€” this is a one-time cost and the matrix is very sparse (~1.37M non-zero entries, i.e., density â‰ˆ 0.000012).

2. **Reshape each variable into a matrix** of dimension `N_cells Ã— N_years` (344,208 Ã— 28).

3. **Compute neighbor stats using sparse matrix operations:**
   - **Mean:** `W_row_normalized %*% X` gives the mean of neighbors for every cell-year in one matrix multiply.
   - **Sum of neighbors:** `W %*% X` (useful for mean = sum / count).
   - **Max and Min:** These are not linear, so sparse matrix multiply doesn't directly work. Instead, use an efficient C++-level grouped operation via `data.table` or, better, iterate over the *sparse matrix entries* in a vectorized way.

4. **For max/min specifically:** Expand the sparse adjacency into a long `data.table` of `(cell_i, neighbor_j)` pairs (~1.37M rows), join with the variable values by `(neighbor_j, year)`, then do a grouped `max/min/mean` by `(cell_i, year)`. This is a `data.table` grouped aggregation over ~1.37M Ã— 28 â‰ˆ 38.5M rows â€” `data.table` handles this in seconds.

### Expected speedup

| Step | Current | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string ops) | ~seconds (sparse matrix from nb) |
| Compute stats (per var) | ~15+ hours (6.46M R-level lapply) | ~5â€“15 seconds (data.table grouped agg) |
| Total (5 vars) | 86+ hours | **< 5 minutes** |

### Memory

- Sparse matrix: ~1.37M entries Ã— 12 bytes â‰ˆ 16 MB
- Long neighbor table: ~38.5M rows Ã— 3 cols Ã— 8 bytes â‰ˆ 925 MB (fits in 16 GB)
- All operations fit comfortably in 16 GB RAM.

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact same numerical results (max, min, mean of rook neighbors)
# Preserves: trained Random Forest model (no retraining)
# ==============================================================================

library(data.table)
library(spdep)    # for nb object handling

# --------------------------------------------------------------------------
# Step 1: Build a long edge table from the nb object (one-time, ~seconds)
# --------------------------------------------------------------------------
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object (list of integer index vectors)
  # id_order is the vector mapping position -> cell id
  
  from_idx <- rep(seq_along(neighbors_nb), lengths(neighbors_nb))
  to_idx   <- unlist(neighbors_nb)
  
  # Convert positional indices to actual cell IDs
  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

# --------------------------------------------------------------------------
# Step 2: Compute all neighbor features for all variables at once
# --------------------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, neighbors_nb,
                                          source_vars) {
  
  # Convert to data.table if not already (non-destructive copy)
  dt <- as.data.table(cell_data)
  
  # --- Build edge table ---
  cat("Building edge table from nb object...\n")
  edges <- build_edge_table(id_order, neighbors_nb)
  cat(sprintf("  Edge table: %s directed neighbor pairs\n",
              format(nrow(edges), big.mark = ",")))
  
  # --- Create a keyed lookup of cell-year rows ---
  # We need: for each (from_id, year), find all to_id neighbors,
  #          look up their variable values in that year,
  #          compute max/min/mean.
  
  # Subset to only the columns we need for the join
  id_year_cols <- c("id", "year", source_vars)
  dt_sub <- dt[, ..id_year_cols]
  
  # For each variable, do the grouped aggregation
  for (var_name in source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    # Build the neighbor-value table:
    # For every (from_id, year), get the var_name value of each neighbor (to_id)
    
    # Prepare neighbor values: keyed by (to_id = id, year)
    neighbor_vals <- dt_sub[, .(id, year, val = get(var_name))]
    setkey(neighbor_vals, id, year)
    
    # Expand edges Ã— years: for each edge (from_id, to_id), 
    # we need all years. But instead of a full cross join (expensive),
    # we join edges with the data on to_id.
    
    # Join: for each edge, get all (to_id, year, val) combinations
    # edges has (from_id, to_id); neighbor_vals has (id, year, val)
    # We want: (from_id, to_id, year, val) where to_id = id
    
    setnames(neighbor_vals, "id", "to_id")
    setkey(edges, to_id)
    setkey(neighbor_vals, to_id)
    
    # This is the key join: ~1.37M edges Ã— 28 years = ~38.5M rows
    joined <- neighbor_vals[edges, on = "to_id", allow.cartesian = TRUE,
                            nomatch = NA]
    # joined has columns: to_id, year, val, from_id
    
    # Remove NA values (matching original behavior: neighbor_vals[!is.na(...)])
    joined <- joined[!is.na(val)]
    
    # Grouped aggregation by (from_id, year)
    stats <- joined[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = .(from_id, year)]
    
    # Rename columns to match expected output naming convention
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))
    setnames(stats, "from_id", "id")
    
    # Merge back into dt
    setkey(stats, id, year)
    setkey(dt, id, year)
    dt <- stats[dt, on = .(id, year)]
    
    # Reset neighbor_vals name for next iteration
    setnames(neighbor_vals, "to_id", "id")
    
    cat(sprintf("  Done. Added: %s, %s, %s\n", max_col, min_col, mean_col))
  }
  
  # Convert back to data.frame if the original was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    setDF(dt)
  }
  
  return(dt)
}

# --------------------------------------------------------------------------
# Step 3: Run it (replaces the entire original outer loop)
# --------------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data        = cell_data,
  id_order         = id_order,
  neighbors_nb     = rook_neighbors_unique,
  source_vars      = neighbor_source_vars
)

# --------------------------------------------------------------------------
# Step 4: Predict with the pre-trained Random Forest (unchanged)
# --------------------------------------------------------------------------
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Exact Numerical Results

| Original operation | Optimized equivalent | Identical? |
|---|---|---|
| `vals[idx]` â€” subset neighbor values | `join on to_id, year` â€” same values retrieved | âœ… Yes |
| `neighbor_vals[!is.na()]` | `joined[!is.na(val)]` | âœ… Yes |
| `max(neighbor_vals)` | `joined[, max(val), by=.(from_id, year)]` | âœ… Yes |
| `min(neighbor_vals)` | `joined[, min(val), by=.(from_id, year)]` | âœ… Yes |
| `mean(neighbor_vals)` | `joined[, mean(val), by=.(from_id, year)]` | âœ… Yes |
| Cells with no valid neighbors â†’ `NA` | Left join back to `dt` â†’ `NA` for unmatched | âœ… Yes |

The `data.table` grouped aggregation computes the identical `max`, `min`, and `mean` over the identical set of non-NA neighbor values for each `(cell, year)` pair. The Random Forest model is never retouched â€” only the feature-engineering step is accelerated.

---

## 5. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Architecture** | Per-row R `lapply` (6.46M iterations) | Vectorized `data.table` join + grouped agg |
| **Neighbor lookup** | 6.46M string-paste + named-vector hash | Sparse edge table (~1.37M rows), built once |
| **Stats computation** | 5 Ã— 6.46M `lapply` calls with R-level `max/min/mean` | 5 Ã— one `data.table` grouped aggregation (~38.5M rows) |
| **Estimated runtime** | 86+ hours | **< 5 minutes** |
| **Peak memory** | ~2â€“4 GB (6.46M-element list) | ~1â€“2 GB (38.5M-row join table, temporary) |
| **Numerical results** | Baseline | **Identical** |
| **RF model** | Unchanged | **Unchanged** |
END_RESPONSE>>>
