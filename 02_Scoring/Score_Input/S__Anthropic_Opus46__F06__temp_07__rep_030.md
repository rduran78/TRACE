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
run_id: Anthropic_F06_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows

For every single cell-year row, the function:
1. Looks up the spatial cell's reference index.
2. Retrieves neighbor cell IDs.
3. Pastes together `id_year` keys (character allocation).
4. Performs named-vector lookups via `idx_lookup[neighbor_keys]`.

This is ~6.46 million iterations of character paste + named-vector subset. Named-vector lookup in R is **O(n)** in the worst case (hash collisions aside, the overhead of repeated character matching on a vector of 6.46M names is enormous). The result is a list of 6.46M integer vectors â€” itself a large memory object.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46 million rows, repeated 5 times

For each of the 5 variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by index, removing NAs, and computing max/min/mean. This is 5 Ã— 6.46M = ~32.3 million R-level function calls. The overhead per call is small but the sheer count is devastating.

### Why raster focal/kernel operations are a useful analogy but not directly applicable

Focal operations (e.g., `terra::focal`) work on regular grids with fixed rectangular kernels. The rook-neighbor structure here *is* on a regular grid, so in principle a 3Ã—3 cross-shaped kernel (`matrix(c(0,1,0,1,0,1,0,1,0), 3, 3)`) would compute the same thing. However:
- The data is in **long panel format** (cell-year rows), not raster layers.
- Boundary cells and missing cells would need careful handling.
- The existing `spdep::nb` object already encodes the topology.

The better strategy is to **vectorize the neighbor computation entirely** using sparse matrix multiplication / sparse adjacency operations, eliminating both `lapply` loops.

---

## 2. Optimization Strategy

### Key Insight: Separate the spatial dimension from the temporal dimension

Neighbor relationships are **purely spatial** â€” they don't change across years. The current code redundantly re-discovers the same spatial neighbors for every year. We should:

1. **Build a sparse spatial adjacency matrix** (344,208 Ã— 344,208) from the `nb` object â€” done once.
2. **Reshape each variable into a matrix** of shape (344,208 cells Ã— 28 years).
3. **Use sparse matrixâ€“dense matrix multiplication** to compute neighbor sums and counts, then derive max/min/mean.

For **mean**: `W %*% X / W %*% 1` (where `W` is the binary adjacency matrix, `X` is the cellÃ—year matrix, and `1` is a matrix of non-NA indicators) gives exact neighbor means via a single sparse matrix multiply.

For **max and min**: Sparse matrix multiplication doesn't directly give max/min. But we can use a **grouped operation** approach: for each cell, gather neighbor values and compute max/min. We vectorize this using `data.table` grouped operations on an edge list, which is far faster than 6.46M individual R function calls.

### Expected speedup

| Component | Current | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M paste+match) | Eliminated (use sparse matrix or edge list directly) |
| Mean computation (Ã—5 vars) | ~hours (32.3M lapply calls) | ~seconds (5 sparse matrix multiplications) |
| Max/Min computation (Ã—5 vars) | included above | ~minutes (data.table grouped ops on ~1.37M edges Ã— 28 years) |
| **Total** | **86+ hours** | **~5â€“15 minutes** |

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# ==============================================================================
# 
# Requirements: data.table, Matrix, spdep (already available since nb object exists)
# 
# Inputs:
#   cell_data              â€” data.frame/data.table with columns: id, year, 
#                            and the 5 neighbor source variables
#   id_order               â€” integer/character vector of unique cell IDs 
#                            (same order as rook_neighbors_unique)
#   rook_neighbors_unique  â€” spdep::nb object (list of neighbor index vectors)
#
# Output:
#   cell_data with 15 new columns: {var}_nb_max, {var}_nb_min, {var}_nb_mean
#   for each of the 5 neighbor source variables.
#
# The trained Random Forest model is NOT touched.
# Numerical results are identical to the original implementation.
# ==============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # ---- Convert to data.table for speed ----
  was_df <- !is.data.table(cell_data)
  if (was_df) cell_data <- as.data.table(cell_data)
  
  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(cell_data)))
  
  # ---- Step 1: Build edge list from nb object (one-time, spatial only) ----
  cat("Building edge list from nb object...\n")
  
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb encodes no-neighbor as 0L in a length-1 vector
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) == 0L) return(NULL)
    data.table(from_ref = i, to_ref = nb_i)
  }))
  
  # Map ref indices to actual cell IDs
  edges[, from_id := id_order[from_ref]]
  edges[, to_id   := id_order[to_ref]]
  
  cat(sprintf("Edge list: %d directed edges\n", nrow(edges)))
  
  # ---- Step 2: Build a fast row-lookup for cell_data ----
  # Create a mapping: (id, year) -> row index in cell_data
  cell_data[, .row_idx := .I]
  
  # Create integer keys for fast joining
  setkey(cell_data, id, year)
  
  # ---- Step 3: Expand edge list across all years ----
  # Each spatial edge applies to every year
  cat("Expanding edge list across years...\n")
  
  # Cross join edges Ã— years
  year_dt <- data.table(year = years)
  edge_year <- edges[, .(from_id, to_id)][, CJ_key := 1L]
  year_dt[, CJ_key := 1L]
  
  # More memory-efficient: use rep
  n_edges <- nrow(edges)
  edge_year <- data.table(
    from_id = rep(edges$from_id, times = n_years),
    to_id   = rep(edges$to_id,   times = n_years),
    year    = rep(years, each = n_edges)
  )
  
  cat(sprintf("Edge-year table: %d rows (%.1f M)\n", 
              nrow(edge_year), nrow(edge_year) / 1e6))
  
  # ---- Step 4: For each variable, join neighbor values and compute stats ----
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Pre-build a lookup table: (id, year) -> row index + variable values
  # We'll join on (to_id, year) to get the neighbor's value
  # and then group by (from_id, year) to compute max, min, mean
  
  # Create a lean lookup keyed by (id, year)
  lookup_cols <- c("id", "year", neighbor_source_vars)
  lookup <- cell_data[, ..lookup_cols]
  setnames(lookup, "id", "to_id")
  setkey(lookup, to_id, year)
  
  # Join neighbor values onto edge_year
  cat("Joining neighbor values...\n")
  setkey(edge_year, to_id, year)
  edge_year <- lookup[edge_year, on = .(to_id, year), nomatch = NA]
  
  # Now edge_year has columns: to_id, year, ntl, ec, ..., from_id
  # Group by (from_id, year) to get stats
  
  cat("Computing neighbor statistics...\n")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    # Compute grouped stats, removing NAs as in original code
    stats <- edge_year[
      !is.na(get(var_name)),
      .(
        nb_max  = max(get(var_name), na.rm = TRUE),
        nb_min  = min(get(var_name), na.rm = TRUE),
        nb_mean = mean(get(var_name), na.rm = TRUE)
      ),
      by = .(from_id, year)
    ]
    
    setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
    setnames(stats, "from_id", "id")
    
    # Join back to cell_data
    setkey(stats, id, year)
    setkey(cell_data, id, year)
    
    # Remove columns if they already exist (idempotency)
    for (cc in c(max_col, min_col, mean_col)) {
      if (cc %in% names(cell_data)) cell_data[, (cc) := NULL]
    }
    
    cell_data <- stats[cell_data, on = .(id, year)]
  }
  
  # Clean up helper column
  cell_data[, .row_idx := NULL]
  
  # Cells with no neighbors (or all-NA neighbors) will have NA for the stats,
  # exactly matching the original implementation's behavior.
  
  cat("Done.\n")
  
  if (was_df) cell_data <- as.data.frame(cell_data)
  return(cell_data)
}

# ==============================================================================
# USAGE
# ==============================================================================
# 
# cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# # Then proceed with prediction using the pre-trained Random Forest:
# predictions <- predict(rf_model, newdata = cell_data)
#
```

---

## 4. Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | We read from the identical `rook_neighbors_unique` nb object. The edge list is a lossless representation of the same adjacency. |
| **Same statistics** | `max`, `min`, `mean` are computed on the same sets of non-NA neighbor values, grouped by `(from_id, year)` â€” identical to the original per-row `lapply`. |
| **NA handling** | `!is.na(get(var_name))` in the filter + the `nomatch = NA` in the join replicates the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]` logic. Cells with zero valid neighbors get `NA` for all three stats (via the left join). |
| **Random Forest model** | Untouched. We only add the same 15 feature columns the model expects. |
| **Numerical estimand** | Preserved exactly â€” no approximation, no rounding, no kernel substitution. |

## 5. Memory Considerations (16 GB laptop)

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M Ã— ~125 cols) | ~6â€“8 GB |
| `edge_year` (1.37M edges Ã— 28 years Ã— 7 cols) | ~2.5 GB |
| `lookup` (6.46M Ã— 7 cols) | ~0.4 GB |
| Temporary stats tables | ~0.3 GB |
| **Total peak** | **~11â€“12 GB** âœ“ fits in 16 GB |

If memory is tight, the edge-year expansion can be chunked by year (process 4â€“5 years at a time), reducing peak usage to ~7â€“8 GB with minimal code change:

```r
# Memory-conservative variant: chunk by year
chunk_size <- 5  # years per chunk
year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))

for (var_name in neighbor_source_vars) {
  stats_list <- lapply(year_chunks, function(yr_chunk) {
    ey_chunk <- data.table(
      from_id = rep(edges$from_id, times = length(yr_chunk)),
      to_id   = rep(edges$to_id,   times = length(yr_chunk)),
      year    = rep(yr_chunk, each = nrow(edges))
    )
    setkey(ey_chunk, to_id, year)
    ey_chunk <- lookup[ey_chunk, on = .(to_id, year), nomatch = NA]
    ey_chunk[!is.na(get(var_name)),
      .(nb_max = max(get(var_name)), 
        nb_min = min(get(var_name)), 
        nb_mean = mean(get(var_name))),
      by = .(from_id, year)]
  })
  stats <- rbindlist(stats_list)
  # ... join back as before
}
```

## 6. Expected Runtime

| Step | Time |
|---|---|
| Edge list construction | ~5 seconds |
| Edge-year expansion | ~10 seconds |
| Join + grouped stats (Ã—5 vars) | ~3â€“8 minutes |
| **Total** | **~5â€“10 minutes** (vs. 86+ hours) |

This represents a **~500â€“1000Ã— speedup** by eliminating 32+ million R-level function calls and replacing them with vectorized `data.table` grouped operations on a pre-joined edge table.
END_RESPONSE>>>
