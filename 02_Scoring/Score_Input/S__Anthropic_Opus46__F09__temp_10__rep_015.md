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
run_id: Anthropic_F09_T10_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is critically slow (~86+ hours) for three interconnected reasons:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates over every cell-year row, performing per-row string pasting (`paste(id, year)`), hash lookups (`id_to_ref`, `idx_lookup`), and subsetting. With ~6.46 million iterations, each involving character coercion, string concatenation, named-vector lookups, and `NA` filtering, this is extremely expensive. Named-vector lookups in R are O(n) in the worst case when the vector is large (6.46M keys), and the function creates ~6.46 million small integer vectors stored in a list.

### 2. Row-Level `lapply` Over 6.46 Million Rows in `compute_neighbor_stats`
For each of 5 variables, the function loops over all 6.46M rows again, extracting neighbor values by index, removing `NA`s, and computing `max`/`min`/`mean`. That's ~32.3 million R-level function calls total (5 vars Ã— 6.46M rows), each with overhead from anonymous function dispatch, subsetting, and `NA` handling.

### 3. Redundant Recomputation of the Neighbor Topology Per Cell-Year
The key insight in the prompt is correct: **the neighbor topology is time-invariant**. Cell A's rook neighbors are the same in 1992 as in 2019. Yet the current code embeds the year into the lookup, effectively rebuilding a "who are my neighbors and where are they in the data frame" mapping for every cell-year combination. The topology only needs to be defined once across the 344,208 cells, then joined to yearly attributes.

### Memory Profile
The `neighbor_lookup` list alone stores ~6.46 million integer vectors. With an average of ~4 rook neighbors per cell, that's ~25.8 million integers (~200 MB) plus R list overhead (~50 bytes per element â†’ ~320 MB). Total: ~520 MB just for the lookup. This is feasible on 16 GB RAM but wasteful.

---

## Optimization Strategy

**Core idea:** Separate the spatial topology (which cells neighbor which cells) from the temporal panel attributes (what values those cells have in a given year). Build the topology table once (344K cells Ã— ~4 neighbors = ~1.37M rows), then use a vectorized merge/join to attach yearly attribute values, then compute grouped statistics.

### Step-by-step:

1. **Build a cell-level neighbor edge table once** â€” a `data.table` with columns `(id, neighbor_id)` derived from the `spdep::nb` object. This has ~1.37 million rows and never changes.

2. **For each variable, join yearly attributes onto the edge table** â€” For a given variable, join `cell_data[, .(id, year, var)]` onto the edge table by `(neighbor_id, year)` to get each neighbor's value for each cell-year.

3. **Compute grouped `max`, `min`, `mean`** â€” Group by `(id, year)` and compute the three statistics in one vectorized `data.table` operation.

4. **Join results back** to the main `cell_data`.

This replaces ~32.3 million R-level function calls with ~5 vectorized `data.table` grouped joins. Expected runtime: **minutes, not days**.

### Complexity comparison:

| | Current | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M R-level iterations | 1 vectorized table build (1.37M rows) |
| Per-variable stats | 6.46M R-level iterations Ã— 5 | 5 keyed joins + grouped aggregations |
| Total R-level loop iterations | ~38.8M | ~0 (vectorized) |
| Expected wall time | 86+ hours | 5â€“15 minutes |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame or data.table with columns: id, year, ntl, ec, 
#                pop_density, def, usd_est_n2, ... (6.46M rows)
#   - id_order: integer/character vector of cell IDs matching the spdep::nb object
#   - rook_neighbors_unique: spdep::nb object (list of integer index vectors)
#   - rf_model: the pre-trained Random Forest model (untouched)
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table if needed ----------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- Step 1: Build time-invariant cell neighbor edge table (once) -----------
build_neighbor_edge_table <- function(id_order, nb_object) {
  # nb_object is a list of length length(id_order).
  # nb_object[[i]] is an integer vector of indices into id_order for the
  # neighbors of the i-th cell. A value of 0L (or integer(0)) means no neighbors.
  
  n <- length(id_order)
  
  # Pre-allocate: count total edges
  edge_counts <- vapply(nb_object, function(x) {
    # spdep::nb encodes "no neighbors" as a single 0L
    valid <- x[x > 0L]
    length(valid)
  }, integer(1))
  
  total_edges <- sum(edge_counts)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_object[[i]]
    nbrs <- nbrs[nbrs > 0L]
    k <- length(nbrs)
    if (k > 0L) {
      from_id[pos:(pos + k - 1L)] <- id_order[i]
      to_id[pos:(pos + k - 1L)]   <- id_order[nbrs]
      pos <- pos + k
    }
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

cat("Building neighbor edge table...\n")
edge_table <- build_neighbor_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %d rows (directed neighbor pairs)\n", nrow(edge_table)))

# ---- Step 2: Compute neighbor stats for all variables (vectorized) ----------
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_names) {
  # cell_dt: data.table with columns id, year, and all var_names
  # edge_dt: data.table with columns id, neighbor_id
  # Returns: cell_dt with new columns appended
  
  # Create a minimal lookup table: (id, year, var1, var2, ...)
  lookup_cols <- c("id", "year", var_names)
  lookup <- cell_dt[, ..lookup_cols]
  setnames(lookup, "id", "neighbor_id")
  
  # Key the lookup for fast join
  setkey(lookup, neighbor_id, year)
  
  # Key the edge table
  setkey(edge_dt, neighbor_id)
  
  for (var_name in var_names) {
    cat(sprintf("  Processing variable: %s\n", var_name))
    
    # Subset lookup to just this variable for memory efficiency
    var_lookup <- lookup[, .(neighbor_id, year, val = get(var_name))]
    setkey(var_lookup, neighbor_id, year)
    
    # Join: for each (id, neighbor_id) edge, attach the neighbor's year+value
    # This creates a table: (id, neighbor_id, year, val) â€” one row per 
    # edge per year = ~1.37M edges Ã— 28 years = ~38.4M rows
    #
    # To avoid materializing 38.4M rows at once, we join edge_table with
    # the yearly data via a cross approach. But actually, the correct 
    # approach is: for each cell-year row, find its neighbors' values.
    #
    # Efficient approach: 
    #   1. Join edge_table (id, neighbor_id) with var_lookup (neighbor_id, year, val)
    #      This gives (id, neighbor_id, year, val) â€” the neighbor's value
    #   2. Aggregate by (id, year) to get max, min, mean
    
    # Join: edge_table[var_lookup] or merge
    # We want all combinations of edges Ã— years where the neighbor has data
    joined <- var_lookup[edge_dt, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = NA]
    # Result columns: neighbor_id, year, val, id
    # This has nrow(edge_dt) Ã— n_years rows â‰ˆ 1.37M Ã— 28 = 38.4M
    # At ~24 bytes/row this is ~920 MB â€” tight on 16 GB but feasible
    
    # Aggregate by (id, year), ignoring NA values
    agg <- joined[!is.na(val), 
                   .(nmax  = max(val), 
                     nmin  = min(val), 
                     nmean = mean(val)), 
                   by = .(id, year)]
    
    # Rename columns to match expected feature names
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    setnames(agg, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
    
    # Join aggregated stats back to cell_data
    setkey(agg, id, year)
    setkey(cell_dt, id, year)
    
    # Remove old columns if they exist (idempotent reruns)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    }
    
    cell_dt <- agg[cell_dt, on = .(id, year)]
    setkey(cell_dt, id, year)
    
    # Clean up to free memory
    rm(var_lookup, joined, agg)
    gc()
  }
  
  cell_dt
}

# ---- Step 3: Run the optimized pipeline -------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorized)...\n")
t_start <- Sys.time()

cell_data <- compute_neighbor_features_fast(cell_data, edge_table, neighbor_source_vars)

t_end <- Sys.time()
cat(sprintf("  Done in %.1f minutes.\n", as.numeric(difftime(t_end, t_start, units = "mins"))))

# ---- Step 4: Predict with existing Random Forest model (unchanged) ----------
# The rf_model is the pre-trained model and is NOT retrained.
# Ensure the column names match what the model expects.

cat("Generating predictions with pre-trained Random Forest model...\n")

# Identify the predictor columns the model expects
rf_pred_vars <- if (inherits(rf_model, "ranger")) {
  rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  rownames(rf_model$importance)
} else {
  stop("Unknown RF model class. Extract predictor names manually.")
}

# Verify all expected predictors are present
missing_vars <- setdiff(rf_pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  warning(sprintf("Missing predictor columns: %s", paste(missing_vars, collapse = ", ")))
}

# Generate predictions
if (inherits(rf_model, "ranger")) {
  pred_result <- predict(rf_model, data = cell_data[, ..rf_pred_vars])
  cell_data[, prediction := pred_result$predictions]
} else if (inherits(rf_model, "randomForest")) {
  pred_result <- predict(rf_model, newdata = as.data.frame(cell_data[, ..rf_pred_vars]))
  cell_data[, prediction := pred_result]
}

cat("Pipeline complete.\n")
```

---

## Memory-Constrained Variant

If the ~38.4M-row intermediate join causes memory pressure on 16 GB RAM, process years in chunks:

```r
compute_neighbor_features_chunked <- function(cell_dt, edge_dt, var_names, years) {
  # Process year-by-year to cap peak memory at ~1.37M rows per join
  
  lookup_cols <- c("id", "year", var_names)
  results_list <- vector("list", length(years))
  
  for (var_name in var_names) {
    cat(sprintf("  Processing variable: %s (chunked by year)\n", var_name))
    
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    agg_all <- rbindlist(lapply(years, function(yr) {
      # Extract this year's values
      yr_vals <- cell_dt[year == yr, .(neighbor_id = id, val = get(var_name))]
      setkey(yr_vals, neighbor_id)
      
      # Join with edge table: for each edge, get neighbor's value this year
      joined <- yr_vals[edge_dt, on = .(neighbor_id), nomatch = NA]
      
      # Aggregate
      joined[!is.na(val), 
             .(nmax = max(val), nmin = min(val), nmean = mean(val), year = yr), 
             by = .(id)]
    }))
    
    setnames(agg_all, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
    
    # Remove old columns if present
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    }
    
    setkey(agg_all, id, year)
    setkey(cell_dt, id, year)
    cell_dt <- agg_all[cell_dt, on = .(id, year)]
    
    rm(agg_all)
    gc()
  }
  
  cell_dt
}

# Usage:
all_years <- sort(unique(cell_data$year))  # 1992:2019
cell_data <- compute_neighbor_features_chunked(cell_data, edge_table, neighbor_source_vars, all_years)
```

**Peak memory per join iteration:** ~1.37M rows Ã— ~32 bytes â‰ˆ 44 MB (trivial).

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor topology | Rebuilt per cell-year (6.46M iterations) | Built once as edge table (1.37M rows) |
| Stats computation | 6.46M R `lapply` calls Ã— 5 vars | 5 vectorized `data.table` join + group-by |
| String hashing | ~12.9M `paste()` + named vector lookups | Zero â€” integer keys only |
| Estimated runtime | 86+ hours | 5â€“15 minutes |
| RAM peak | ~520 MB (lookup list) + data | ~1â€“2 GB (join intermediate) + data |
| RF model | Unchanged | Unchanged |
| Numerical output | Identical `max`, `min`, `mean` | Identical `max`, `min`, `mean` |
END_RESPONSE>>>
