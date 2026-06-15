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
run_id: Anthropic_F09_T07_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-Level `lapply` in `build_neighbor_lookup` (Primary Bottleneck)
The function iterates over **~6.46 million rows**, and for each row it:
- Looks up the cell's reference index in `id_to_ref`.
- Extracts neighbor cell IDs from the `nb` object.
- Pastes cell IDs and years together to form string keys.
- Looks up row indices via named-vector string matching (`idx_lookup[neighbor_keys]`).

This means ~6.46 million iterations of string concatenation (`paste`) and named-vector lookups, which in R are O(n) or hash-based but with enormous per-call overhead. The `idx_lookup` named vector has ~6.46 million entries, and each call does substring matching against it. This alone likely takes tens of hours.

### 2. Row-Level `lapply` in `compute_neighbor_stats` (Secondary Bottleneck)
For each of the ~6.46 million rows, it subsets `vals[idx]`, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per neighbor source variable), yielding ~32.3 million R-level function calls with per-element overhead.

### 3. Redundant Recomputation Across Years
The **neighbor topology is purely spatial** â€” cell A's rook neighbors are the same in every year. Yet the current code rebuilds the full row-level lookup across all 6.46M cell-year rows, effectively re-deriving the same spatial adjacency for each of the 28 years. This is a 28Ã— redundancy.

---

## Optimization Strategy

The key insight: **separate topology (spatial, static) from attributes (yearly, dynamic).**

1. **Build a cell-level adjacency table once** â€” a `data.table` with columns `(cell_id, neighbor_id)`, derived from `rook_neighbors_unique`. This has ~1.37 million rows and is built in seconds.

2. **Join yearly attributes onto the adjacency table** â€” for each year and variable, join `cell_data` onto the adjacency table by `neighbor_id` and `year`. This gives each cell-year row the attribute values of all its neighbors.

3. **Aggregate with `data.table` grouping** â€” group by `(cell_id, year)` and compute `max`, `min`, `mean` in a single vectorized pass.

4. **Join aggregates back** to the main `cell_data` table.

This replaces millions of R-level iterations with vectorized `data.table` joins and grouped aggregations, reducing runtime from ~86 hours to **minutes**.

**The trained Random Forest model is never touched.** The output columns are numerically identical (same `max`, `min`, `mean` of the same neighbor values), preserving the original estimand.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ============================================================
setDT(cell_data)

# Ensure id and year are keyed for fast joins
if (!("id" %in% names(cell_data)) || !("year" %in% names(cell_data))) {
  stop("cell_data must contain 'id' and 'year' columns.")
}

# ============================================================
# STEP 1: Build the static cell-level adjacency table ONCE
#          from the precomputed spdep::nb object.
# ============================================================
# rook_neighbors_unique is a list of length = number of cells.
# rook_neighbors_unique[[i]] gives integer indices of neighbors
# of the i-th cell (in id_order).
# id_order is the vector mapping index position -> cell id.

build_adjacency_dt <- function(id_order, nb_obj) {
  # Pre-allocate vectors
  n_cells <- length(id_order)
  # Count total neighbor pairs for pre-allocation
  n_pairs <- sum(lengths(nb_obj))
  
  from_id <- integer(n_pairs)
  to_id   <- integer(n_pairs)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_indices <- nb_obj[[i]]
    # spdep::nb uses 0 to denote no neighbors; filter those out
    nb_indices <- nb_indices[nb_indices > 0L]
    n_nb <- length(nb_indices)
    if (n_nb > 0L) {
      from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
      to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_indices]
      pos <- pos + n_nb
    }
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  if (pos - 1L < n_pairs) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  adj_dt <- data.table(cell_id = from_id, neighbor_id = to_id)
  return(adj_dt)
}

cat("Building adjacency table...\n")
system.time({
  adj_dt <- build_adjacency_dt(id_order, rook_neighbors_unique)
})
# adj_dt has ~1,373,394 rows: (cell_id, neighbor_id)
cat(sprintf("Adjacency table: %d directed neighbor pairs.\n", nrow(adj_dt)))

# ============================================================
# STEP 2: For each neighbor source variable, compute neighbor
#          max, min, mean via data.table join + group-by.
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

compute_neighbor_features_dt <- function(cell_data, adj_dt, var_name) {
  cat(sprintf("  Computing neighbor stats for: %s\n", var_name))
  
  # Extract only the columns we need: (id, year, var_name)
  # This avoids copying the entire 110-column table into the join.
  cols_needed <- c("id", "year", var_name)
  attr_dt <- cell_data[, ..cols_needed]
  setnames(attr_dt, c("id", var_name), c("neighbor_id", "nb_val"))
  setkey(attr_dt, neighbor_id, year)
  
  # Create the join table: for every year, expand adjacency with that year.
  # Instead of a massive cross-join, we join adj_dt with the attribute table.
  # 
  # For each (cell_id, neighbor_id) pair in adj_dt, we need the neighbor's
  # value in EVERY year that cell_id appears. Since every cell appears in
  # every year (balanced panel), we can expand adj_dt Ã— years, then join.
  #
  # More efficient: join cell_data's (id, year) with adj_dt on cell_id = id,
  # then join neighbor attributes on (neighbor_id, year).
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  # Expand adjacency table by years: ~1.37M pairs Ã— 28 years = ~38.5M rows
  # This fits in memory: 38.5M Ã— 3 int columns â‰ˆ ~460 MB (manageable on 16GB)
  
  # Use CJ-like expansion efficiently
  adj_expanded <- adj_dt[, .(year = years), by = .(cell_id, neighbor_id)]
  # adj_expanded has columns: cell_id, neighbor_id, year
  
  # Join neighbor attribute values
  setkey(adj_expanded, neighbor_id, year)
  adj_expanded[attr_dt, nb_val := i.nb_val, on = .(neighbor_id, year)]
  
  # Aggregate: group by (cell_id, year), compute max/min/mean of nb_val
  stats <- adj_expanded[
    !is.na(nb_val),
    .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ),
    by = .(cell_id, year)
  ]
  
  # Rename columns to match expected feature names
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Clean up the large intermediate table
  rm(adj_expanded)
  gc()
  
  return(stats)
}

# ============================================================
# STEP 3: Compute and merge all neighbor features
# ============================================================

# Remove any previously computed neighbor columns to avoid duplication
old_nb_cols <- grep("^neighbor_(max|min|mean)_", names(cell_data), value = TRUE)
if (length(old_nb_cols) > 0) {
  cell_data[, (old_nb_cols) := NULL]
}

cat("Computing neighbor features for all variables...\n")
total_time <- system.time({
  for (var_name in neighbor_source_vars) {
    stats_dt <- compute_neighbor_features_dt(cell_data, adj_dt, var_name)
    
    # Merge back onto cell_data
    # Use cell_id = id, year = year
    setkey(stats_dt, cell_id, year)
    new_cols <- setdiff(names(stats_dt), c("cell_id", "year"))
    cell_data[stats_dt, (new_cols) := mget(paste0("i.", new_cols)),
              on = .(id = cell_id, year = year)]
    
    rm(stats_dt)
    gc()
    cat(sprintf("    Done: %s\n", var_name))
  }
})
cat(sprintf("Total neighbor feature computation time: %.1f seconds\n", total_time["elapsed"]))

# ============================================================
# STEP 4: Predict with the existing trained Random Forest
#          (model object is unchanged)
# ============================================================

# cell_data now has the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns as the original pipeline, with identical numerical values.
# Proceed with prediction using the pre-trained model:

# predictions <- predict(trained_rf_model, newdata = cell_data)

cat("Neighbor features computed. Ready for Random Forest prediction.\n")
```

---

## Memory-Optimized Variant (If 38.5M-Row Expansion Is Too Large)

If the ~38.5 million row `adj_expanded` table strains the 16 GB RAM (especially with the 110-column `cell_data` already in memory), process **year-by-year** instead:

```r
compute_neighbor_features_dt_chunked <- function(cell_data, adj_dt, var_name) {
  cat(sprintf("  Computing neighbor stats (chunked) for: %s\n", var_name))
  
  years <- sort(unique(cell_data$year))
  cols_needed <- c("id", "year", var_name)
  attr_dt <- cell_data[, ..cols_needed]
  setnames(attr_dt, c("id", var_name), c("neighbor_id", "nb_val"))
  setkey(attr_dt, neighbor_id, year)
  
  results_list <- vector("list", length(years))
  
  for (j in seq_along(years)) {
    yr <- years[j]
    
    # Get neighbor values for this year only
    yr_vals <- attr_dt[year == yr, .(neighbor_id, nb_val)]
    setkey(yr_vals, neighbor_id)
    
    # Join onto adjacency table
    adj_yr <- adj_dt[yr_vals, on = .(neighbor_id), nomatch = 0L, allow.cartesian = TRUE]
    # adj_yr has: cell_id, neighbor_id, nb_val
    
    # Aggregate
    stats_yr <- adj_yr[
      !is.na(nb_val),
      .(
        nb_max  = max(nb_val),
        nb_min  = min(nb_val),
        nb_mean = mean(nb_val)
      ),
      by = .(cell_id)
    ]
    stats_yr[, year := yr]
    results_list[[j]] <- stats_yr
  }
  
  stats <- rbindlist(results_list)
  
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  return(stats)
}
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Adjacency** | Rebuilt per-row via string keys across 6.46M rows | Built once as a ~1.37M-row `data.table` |
| **Neighbor lookup** | `lapply` over 6.46M rows with `paste`/named-vector lookup | `data.table` keyed join (vectorized C code) |
| **Stats computation** | `lapply` over 6.46M rows Ã— 5 vars = 32.3M R function calls | `data.table` grouped aggregation (vectorized) |
| **Estimated runtime** | ~86+ hours | **~2â€“10 minutes** |
| **RAM peak** | Moderate (but slow) | ~3â€“6 GB for intermediates (fits in 16 GB) |
| **RF model** | Unchanged âœ“ | Unchanged âœ“ |
| **Numerical output** | Original estimand | Identical values âœ“ |
END_RESPONSE>>>
