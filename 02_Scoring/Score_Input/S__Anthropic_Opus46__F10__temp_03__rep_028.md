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
run_id: Anthropic_F10_T03_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookup, and `NA` filtering. This is an O(n) loop over millions of rows with expensive string operations (`paste`, named vector indexing). The named-vector `idx_lookup` with 6.46M entries makes each lookup O(n) due to R's linear hashing on large named vectors.

2. **`compute_neighbor_stats` iterates over 6.46M list entries** for each of 5 variables (32.3M iterations total), extracting subsets of a numeric vector, removing NAs, and computing `max/min/mean`. Each iteration has R-level overhead (function call, subsetting, `is.na`, three aggregations).

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property â€” they don't change across years. But the lookup is built over the full cell-year panel, duplicating the same spatial adjacency structure 28 times and entangling it with year-matching via string keys. This inflates the problem from ~344K spatial lookups to ~6.46M spatiotemporal lookups.

**Root cause:** The implementation treats the problem as a flat panel operation instead of exploiting the separability of spatial topology (fixed) and temporal alignment (trivially parallel by year).

---

## Optimization Strategy

### Key Insight: Separate Space from Time

The rook neighbor graph is **time-invariant**. For any cell `i` in year `t`, its neighbors' attributes come from the same year `t`. Therefore:

1. **Build the spatial adjacency structure once** over 344K cells (not 6.46M cell-years).
2. **For each year, subset the data, and use vectorized sparse matrixâ€“vector multiplication** to compute neighbor statistics.
3. **Use a sparse adjacency matrix (CSC/CSR format)** from the `Matrix` package. The `spdep::nb` object converts directly to a sparse matrix via `spdep::nb2listw` â†’ `spdep::listw2mat` or more efficiently `spdep::nb2Matrix`.

### Aggregation via Sparse Matrix Operations

For **mean**: `A %*% x / A %*% 1` where `A` is the binary adjacency matrix, `x` is the attribute vector, and `1` is a vector of ones (adjusted for NAs).

For **max** and **min**: Sparse matrix multiplication computes sums, not extremes. However, we can avoid the R-level `lapply` by using **`data.table` grouped operations** on an edge list, which is extremely fast.

### Final Approach: Edge-List + `data.table` Grouped Aggregation

- Convert the `nb` object to an edge list (from, to) â€” ~1.37M rows, built once.
- For each year, join edge list to attribute values, then compute `max/min/mean` grouped by the source node â€” all vectorized in `data.table`.
- This reduces 6.46M list iterations Ã— 5 variables to 28 years Ã— 5 variables Ã— one vectorized grouped aggregation each.

**Expected speedup:** From ~86 hours to **minutes** (roughly 1000â€“5000Ã—).

---

## Optimized R Code

```r
library(data.table)
library(Matrix)
library(spdep)

# ============================================================
# 0. Prerequisites (assumed already in environment):
#    - cell_data: data.frame/data.table with columns id, year, 
#                 ntl, ec, pop_density, def, usd_est_n2, ...
#    - rook_neighbors_unique: spdep nb object (length = 344,208)
#    - id_order: integer/character vector of cell IDs corresponding
#                to positions in rook_neighbors_unique
#    - rf_model: pre-trained Random Forest model (DO NOT retrain)
# ============================================================

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density", 
                                                                  "def", "usd_est_n2")) {
  
  # ----------------------------------------------------------
  # STEP 1: Build edge list ONCE from the nb object

# ----------------------------------------------------------
  cat("Building spatial edge list from nb object...\n")
  
  n_cells <- length(rook_neighbors_unique)
  stopifnot(n_cells == length(id_order))
  
  # Convert nb object to edge list: "from" node index -> "to" node index
  # Each entry rook_neighbors_unique[[i]] contains integer indices of neighbors of node i
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0 to indicate no neighbors
    if (length(nb_i) == 1L && nb_i[1] == 0L) next
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) == 0L) next
    from_list[[i]] <- rep.int(i, length(nb_i))
    to_list[[i]]   <- nb_i
  }
  
  edge_from <- unlist(from_list, use.names = FALSE)
  edge_to   <- unlist(to_list, use.names = FALSE)
  
  # Map positional indices to actual cell IDs
  edges <- data.table(
    source_id   = id_order[edge_from],   # the node whose features we're computing
    neighbor_id = id_order[edge_to]       # the neighbor node
  )
  
  rm(from_list, to_list, edge_from, edge_to)
  
  cat(sprintf("  Edge list: %s directed edges across %s cells.\n",
              format(nrow(edges), big.mark = ","),
              format(n_cells, big.mark = ",")))
  
  # ----------------------------------------------------------
  # STEP 2: Convert cell_data to data.table (if not already)
  # ----------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Ensure key columns exist
  stopifnot(all(c("id", "year") %in% names(cell_data)))
  stopifnot(all(neighbor_source_vars %in% names(cell_data)))
  
  # ----------------------------------------------------------
  # STEP 3: For each variable, compute neighbor stats via
  #         edge-list join + grouped aggregation, by year
  # ----------------------------------------------------------
  
  # Pre-allocate output columns with NA
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
  }
  
  # Create a row-index column for fast assignment
  cell_data[, .row_idx := .I]
  
  # Key for fast joins
  setkey(cell_data, id, year)
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  cat(sprintf("Computing neighbor features for %d variables across %d years...\n",
              length(neighbor_source_vars), length(years)))
  
  for (yr in years) {
    # Subset to this year â€” extract only needed columns
    cols_needed <- c("id", ".row_idx", neighbor_source_vars)
    yr_data <- cell_data[year == yr, ..cols_needed]
    setkey(yr_data, id)
    
    # Join edges with neighbor attributes for this year
    # edges: source_id, neighbor_id
    # We need neighbor attributes keyed by neighbor_id
    # And we need source row indices keyed by source_id
    
    # Get neighbor values: join edges to yr_data on neighbor_id
    neighbor_vals <- merge(
      edges,
      yr_data,
      by.x = "neighbor_id",
      by.y = "id",
      all.x = FALSE,  # inner join: only edges where neighbor exists this year
      allow.cartesian = FALSE
    )
    # neighbor_vals columns: neighbor_id, source_id, .row_idx (of neighbor), vars...
    # Rename .row_idx to avoid confusion â€” this is the neighbor's row index, not needed
    neighbor_vals[, .row_idx := NULL]
    
    # Now for each source_id, aggregate neighbor variable values
    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Compute grouped stats, dropping NAs in the variable
      agg <- neighbor_vals[!is.na(get(var_name)),
                           .(nb_max  = max(get(var_name)),
                             nb_min  = min(get(var_name)),
                             nb_mean = mean(get(var_name))),
                           by = source_id]
      
      if (nrow(agg) == 0L) next
      
      # Get row indices for source cells in this year
      source_rows <- yr_data[agg, on = .(id = source_id), .row_idx]
      
      # Direct assignment by row index into cell_data
      set(cell_data, i = source_rows, j = col_max,  value = agg$nb_max)
      set(cell_data, i = source_rows, j = col_min,  value = agg$nb_min)
      set(cell_data, i = source_rows, j = col_mean, value = agg$nb_mean)
    }
    
    if (yr %% 5 == 0 || yr == years[1] || yr == years[length(years)]) {
      cat(sprintf("  Year %d complete.\n", yr))
    }
  }
  
  # Clean up helper column
  cell_data[, .row_idx := NULL]
  
  cat("Neighbor feature computation complete.\n")
  return(cell_data)
}


# ============================================================
# USAGE
# ============================================================

# Compute all neighbor features
cell_data <- optimize_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Apply the pre-trained Random Forest (DO NOT retrain)
# The model expects the same column names as the training data
predictions <- predict(rf_model, newdata = cell_data)
cell_data$predicted <- predictions
```

---

## Further Optimization: Avoiding `get()` Overhead

The `get(var_name)` inside `data.table` grouped operations has non-trivial overhead at scale. Here is a tighter inner loop that avoids it:

```r
# REPLACEMENT for the inner var_name loop above (drop-in):
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Extract the variable as a simple vector, keyed to neighbor_vals
  v <- neighbor_vals[[var_name]]
  sid <- neighbor_vals[["source_id"]]
  
  # Remove NAs
  valid <- !is.na(v)
  v   <- v[valid]
  sid <- sid[valid]
  
  if (length(v) == 0L) next
  
  # Use data.table for grouped aggregation on plain vectors
  tmp <- data.table(source_id = sid, val = v)
  agg <- tmp[, .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
             by = source_id]
  
  # Map source_id -> row index in cell_data for this year
  source_rows <- yr_data[agg, on = .(id = source_id), .row_idx]
  
  set(cell_data, i = source_rows, j = col_max,  value = agg$nb_max)
  set(cell_data, i = source_rows, j = col_min,  value = agg$nb_min)
  set(cell_data, i = source_rows, j = col_mean, value = agg$nb_mean)
}
```

---

## Why This Preserves Numerical Equivalence

| Original | Optimized | Equivalence |
|---|---|---|
| `max(neighbor_vals)` | `data.table` `max(val)` grouped by source | Identical â€” same IEEE 754 `max` |
| `min(neighbor_vals)` | `data.table` `min(val)` grouped by source | Identical |
| `mean(neighbor_vals)` | `data.table` `mean(val)` grouped by source | Identical â€” `sum/n` on same values |
| NAs excluded via `!is.na()` | NAs excluded via `!is.na()` before grouping | Identical filtering |
| No neighbors â†’ `NA, NA, NA` | No matching edges â†’ columns remain `NA` (pre-initialized) | Identical |

The trained Random Forest model is **never modified** â€” only `predict()` is called on the enriched dataset.

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string-paste + named-vector lookups | ~1.37M edge list rows, built once |
| Per-variable computation | 6.46M R-level `lapply` iterations | 28 `data.table` grouped aggregations on ~1.37M rows |
| Total R-level iterations | ~32.3M (5 vars Ã— 6.46M) | ~140 (5 vars Ã— 28 years), each vectorized |
| Memory for lookup | ~6.46M-element list of integer vectors | ~1.37M Ã— 2 integer edge table (~11 MB) |
| Estimated runtime | 86+ hours | **5â€“15 minutes** |
| Numerical output | Reference | Identical |
END_RESPONSE>>>
