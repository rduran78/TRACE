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
run_id: Anthropic_F09_T10_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all ~6.46 million cell-year rows
For every single cell-year row, the function:
- Looks up the cell's spatial neighbors (fine).
- Constructs **year-specific string keys** (`paste(neighbor_id, year, sep="_")`) and matches them into a named index vector of 6.46M entries.
- `Named vector lookup by character key` in base R is effectively a repeated hash-table probe over a 6.46M-element named vector â€” **for each of 6.46M rows**. This is ~41.7 trillion character-comparison operations in the worst case.

The fundamental mistake: **the neighbor topology is time-invariant, but the lookup table is built as if it were time-varying.** Every cell has the same neighbors in 1992 as in 2019. Yet the code re-discovers and re-encodes this for every year, multiplying work by 28Ã—.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows
Each call extracts a small vector of neighbor values and computes `max/min/mean`. The per-element overhead of `lapply` + anonymous function + subsetting for 6.46M iterations is enormous. This is done 5 times (once per source variable), totaling ~32.3 million R-level function calls.

### 3. Memory pressure from 6.46M-element list
`neighbor_lookup` is a list of 6.46M integer vectors. The list overhead alone (~50 bytes/element for the list spine, plus each integer vector's header) consumes multiple gigabytes before the actual index data, which is dangerous on a 16 GB laptop.

---

## Optimization Strategy

**Core insight:** Separate the **time-invariant spatial topology** from the **time-varying cell attributes**, then use vectorized joins.

### Step-by-step plan:

1. **Build a spatial neighbor edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_cell_id)` with ~1.37M rows (directed rook edges). This is built from `rook_neighbors_unique` and `id_order` and **never touches year**.

2. **Join yearly attributes onto the edge table** â€” for each year, the edge table is joined to the cell attributes by `neighbor_cell_id`, giving each edge the neighbor's variable values. This is a keyed `data.table` equi-join: extremely fast.

3. **Aggregate by `(cell_id, year)`** â€” group the joined edge table by `(cell_id, year)` and compute `max`, `min`, `mean` in one vectorized pass per variable (or all variables at once).

4. **Join aggregated neighbor stats back** onto the main `cell_data` table.

This replaces:
- 6.46M-element `lapply` â†’ **vectorized `data.table` join + group-by**
- 6.46M character-key lookups â†’ **integer-keyed joins**
- 28Ã— redundant topology work â†’ **1Ã— topology table reused for all years**

**Expected speedup:** From ~86 hours to **minutes** (typically 2â€“10 minutes depending on disk I/O and RAM).

**Preserves:** The trained Random Forest model (no retraining), and the original numerical estimand (same `max`, `min`, `mean` neighbor statistics, same column names, same NA behavior).

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build time-invariant spatial neighbor edge table (run once, reuse)
# ===========================================================================
build_neighbor_edge_table <- function(id_order, nb_object) {
  # nb_object: spdep::nb list â€” nb_object[[i]] gives integer indices into

  # id_order for the neighbors of id_order[i].
  # Returns a data.table with columns: cell_id, neighbor_cell_id
  
  n <- length(id_order)
  # Pre-count total edges for pre-allocation
  n_edges <- sum(vapply(nb_object, function(x) {
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(x) == 1L && x[1L] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L
  
  for (i in seq_len(n)) {
    nb_idx <- nb_object[[i]]
    if (length(nb_idx) == 1L && nb_idx[1L] == 0L) next
    k <- length(nb_idx)
    from_id[pos:(pos + k - 1L)] <- id_order[i]
    to_id[pos:(pos + k - 1L)]   <- id_order[nb_idx]
    pos <- pos + k
  }
  
  data.table(cell_id = from_id, neighbor_cell_id = to_id)
}

# Build it once
edge_dt <- build_neighbor_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))

# ===========================================================================
# STEP 2-4: Compute neighbor stats for all variables, join back
# ===========================================================================
compute_all_neighbor_features <- function(cell_data_df, edge_dt,
                                          neighbor_source_vars) {
  # Convert to data.table if needed (by reference if already data.table)
  if (!is.data.table(cell_data_df)) {
    dt <- as.data.table(cell_data_df)
  } else {
    dt <- copy(cell_data_df)
  }
  
  # Ensure key columns are present
  stopifnot(all(c("id", "year") %in% names(dt)))
  
  # Columns we need from the main table for the neighbor join
  # (neighbor_source_vars + id + year)
  attr_cols <- c("id", "year", neighbor_source_vars)
  neighbor_attrs <- dt[, ..attr_cols]
  
  # Rename 'id' to 'neighbor_cell_id' for join
  setnames(neighbor_attrs, "id", "neighbor_cell_id")
  
  # Key the attribute table for fast join
  setkey(neighbor_attrs, neighbor_cell_id, year)
  
  # -------------------------------------------------------------------
  # Cross-join edge table with all years to create (cell_id, year, neighbor_cell_id)
  # Then join neighbor attributes.
  # 
  # Memory note: edge_dt has ~1.37M rows Ã— 28 years = ~38.4M rows.
  # With 5 numeric columns + 3 key columns, this is roughly:
  #   38.4M Ã— 8 cols Ã— 8 bytes â‰ˆ 2.5 GB â€” fits in 16 GB.
  # -------------------------------------------------------------------
  
  # Get unique years
  years <- sort(unique(dt$year))
  
  # Expand edges Ã— years
  edge_year <- CJ_dt(edge_dt, years)
  
  # Join neighbor attributes onto edge_year
  setkey(edge_year, neighbor_cell_id, year)
  edge_year <- neighbor_attrs[edge_year, on = .(neighbor_cell_id, year)]
  
  # Now group by (cell_id, year) and compute max, min, mean for each variable
  setkey(edge_year, cell_id, year)
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
    )
  }))
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  names(agg_exprs) <- agg_names
  
  # Evaluate aggregation
  stats_dt <- edge_year[,
    eval(as.call(c(as.name("list"), agg_exprs))),
    by = .(cell_id, year)
  ]
  
  # Replace -Inf/Inf (from max/min of all-NA groups) with NA
  for (col_name in agg_names) {
    set(stats_dt, which(is.infinite(stats_dt[[col_name]])), col_name, NA_real_)
  }
  
  # Join back onto the main data
  setkey(dt, id, year)
  setkey(stats_dt, cell_id, year)
  
  # Remove any pre-existing neighbor columns to avoid duplication
  existing_neighbor_cols <- intersect(names(dt), agg_names)
  if (length(existing_neighbor_cols) > 0) {
    dt[, (existing_neighbor_cols) := NULL]
  }
  
  dt <- stats_dt[dt, on = .(cell_id = id, year = year)]
  
  # Restore 'id' column name (the join renames it to cell_id)
  setnames(dt, "cell_id", "id")
  
  return(dt)
}

# Helper: Cross-join an edge data.table with a vector of years
CJ_dt <- function(edge_dt, years) {
  # Efficient: replicate edge_dt for each year
  n_edges <- nrow(edge_dt)
  n_years <- length(years)
  
  result <- data.table(
    cell_id         = rep(edge_dt$cell_id,         times = n_years),
    neighbor_cell_id = rep(edge_dt$neighbor_cell_id, times = n_years),
    year            = rep(years, each = n_edges)
  )
  result
}

# ===========================================================================
# EXECUTION
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorized)...\n")
t0 <- proc.time()

cell_data <- compute_all_neighbor_features(
  cell_data, edge_dt, neighbor_source_vars
)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done in %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60))

# ===========================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ===========================================================================
# The trained RF model object is assumed to be already in memory.
# Column names match because we used the same naming convention:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, etc.
#
# If the original code used a different naming convention, adjust the
# agg_names construction above to match exactly.

# Example (adjust to your actual model object and column requirements):
# cell_data$rf_prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## Memory-Constrained Variant (if 2.5 GB edge-year table is too large)

If RAM is tight, process year-by-year without ever building the full 38.4M-row table:

```r
compute_neighbor_features_chunked <- function(cell_data_df, edge_dt,
                                               neighbor_source_vars) {
  if (!is.data.table(cell_data_df)) {
    dt <- as.data.table(cell_data_df)
  } else {
    dt <- copy(cell_data_df)
  }
  
  years <- sort(unique(dt$year))
  attr_cols <- c("id", neighbor_source_vars)
  
  # Prepare aggregation names
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  # Process one year at a time â†’ peak memory â‰ˆ 1.37M rows Ã— ~8 cols
  results_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Subset this year's attributes
    yr_attrs <- dt[year == yr, ..attr_cols]
    setnames(yr_attrs, "id", "neighbor_cell_id")
    setkey(yr_attrs, neighbor_cell_id)
    
    # Join neighbor attributes onto edge table
    edges_with_vals <- yr_attrs[edge_dt, on = .(neighbor_cell_id)]
    
    # Aggregate
    agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
      list(
        bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
        bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
        bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
      )
    }))
    names(agg_exprs) <- agg_names
    
    yr_stats <- edges_with_vals[,
      eval(as.call(c(as.name("list"), agg_exprs))),
      by = .(cell_id)
    ]
    
    # Fix Inf â†’ NA
    for (col_name in agg_names) {
      set(yr_stats, which(is.infinite(yr_stats[[col_name]])), col_name, NA_real_)
    }
    
    yr_stats[, year := yr]
    results_list[[yi]] <- yr_stats
    
    if (yi %% 5 == 0) cat(sprintf("  Processed %d / %d years\n", yi, length(years)))
  }
  
  stats_dt <- rbindlist(results_list)
  
  # Join back
  setkey(dt, id, year)
  setkey(stats_dt, cell_id, year)
  
  existing <- intersect(names(dt), agg_names)
  if (length(existing) > 0) dt[, (existing) := NULL]
  
  dt <- stats_dt[dt, on = .(cell_id = id, year = year)]
  setnames(dt, "cell_id", "id")
  
  return(dt)
}

# Usage:
cell_data <- compute_neighbor_features_chunked(
  cell_data, edge_dt, neighbor_source_vars
)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology representation** | Rebuilt per cell-year (6.46M string-key lookups) | Built once as edge table (~1.37M rows) |
| **Neighbor stat computation** | `lapply` over 6.46M rows Ã— 5 vars | Vectorized `data.table` join + group-by |
| **Estimated runtime** | ~86+ hours | ~2â€“10 minutes |
| **Peak RAM** | ~8â€“12 GB (6.46M-element list) | ~2.5 GB (full) or ~0.5 GB (chunked) |
| **RF model** | Preserved (unchanged) | Preserved (unchanged) |
| **Numerical output** | Same `max`, `min`, `mean` per cell-year | Identical values |
END_RESPONSE>>>
