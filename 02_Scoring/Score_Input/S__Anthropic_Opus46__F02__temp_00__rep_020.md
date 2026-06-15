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
run_id: Anthropic_F02_T00_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries â€” an O(nÂ²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again per variable, subsetting numeric vectors by index lists. With 5 variables, that's ~32.3 million list iterations total.

**Specific problems:**

1. **`build_neighbor_lookup`:** `paste()` and named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely slow. Each call to `idx_lookup[neighbor_keys]` does a linear-time hash probe on a 6.46M-element named character vector. The resulting `neighbor_lookup` list of 6.46M integer vectors also consumes enormous memory (estimated 10â€“15 GB with overhead).
2. **`compute_neighbor_stats`:** `lapply` over 6.46M list elements with per-element subsetting, NA removal, and summary stats is slow but secondary to problem #1.
3. **Memory:** Storing 6.46M list elements (each a variable-length integer vector) plus the 6.46M-row data frame with 110+ columns pushes well past 16 GB.

---

## Optimization Strategy

### Key Insight: Separate Space and Time

The neighbor structure is **purely spatial** â€” neighbors don't change across years. So instead of building a 6.46M-row lookup, build a **344,208-cell spatial lookup once**, then join by year using vectorized operations.

### Strategy Summary

| Step | What | Why |
|------|------|-----|
| 1 | Use `data.table` throughout | Vectorized grouped operations, memory-efficient |
| 2 | Build a flat edge table (cell â†’ neighbor) from the `nb` object â€” only 344K cells, ~1.37M edges | Eliminates per-row string pasting and named-vector lookups |
| 3 | Join edge table to data by (neighbor_id, year) to get neighbor values | Vectorized equi-join, no per-row R function calls |
| 4 | Compute grouped max/min/mean over (cell, year) | Single `data.table` grouped aggregation â€” extremely fast |
| 5 | Process all 5 variables in one pass | Reduces join overhead from 5Ã— to 1Ã— |

**Expected speedup:** From 86+ hours to **~2â€“10 minutes**. Memory peak drops to ~3â€“5 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert to data.table (if not already) and ensure key columns
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order is the vector mapping position index â†’ cell id
# rook_neighbors_unique is the spdep::nb object (list of integer vectors)

setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a flat spatial edge table from the nb object
#         This replaces build_neighbor_lookup entirely.
#         Only 344,208 cells Ã— ~4 neighbors each â‰ˆ 1.37M rows.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for cell i's neighbors
  # We expand this into a two-column data.table: (cell_id, neighbor_id)
  n_cells <- length(id_order)
  
  # Pre-compute lengths for pre-allocation
  lens <- vapply(neighbors, length, integer(1))
  total_edges <- sum(lens)
  
  # Pre-allocate vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    ni <- lens[i]
    if (ni > 0L) {
      idx_range <- pos:(pos + ni - 1L)
      from_id[idx_range] <- id_order[i]
      to_id[idx_range]   <- id_order[neighbors[[i]]]
      pos <- pos + ni
    }
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Join edges with data to retrieve neighbor variable values,
#         then compute grouped statistics â€” all in one pass.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {
  # Subset only the columns we need for the join (saves memory)
  join_cols <- c("id", "year", source_vars)
  neighbor_data <- cell_data[, ..join_cols]
  
  # Key the neighbor data for fast join on (id, year)
  setnames(neighbor_data, "id", "neighbor_id")
  setkey(neighbor_data, neighbor_id, year)
  
  # Get unique years from cell_data
  # We need to cross-join edges Ã— years, then look up neighbor values.
  # But it's more efficient to:
  #   1. Take cell_data's (id, year) pairs
  #   2. Join to edge_dt to get (id, year, neighbor_id)
  #   3. Join to neighbor_data to get neighbor values
  #   4. Aggregate by (id, year)
  
  # Step 2a: Create (cell_id, year, neighbor_id) by joining cell_data's 
  #          unique (id, year) with edge_dt on cell_id = id
  # Since every cell appears for every year (balanced panel), we can 
  # cross-join edges with years directly â€” this is more memory-efficient.
  
  unique_years <- sort(unique(cell_data$year))
  
  # Cross join: each edge Ã— each year
  # ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows â€” fits in memory (~1-2 GB)
  edge_year <- CJ_dt_edges(edge_dt, unique_years)
  
  # Join to get neighbor values
  setkeyv(edge_year, c("neighbor_id", "year"))
  edge_year <- neighbor_data[edge_year, on = .(neighbor_id, year), nomatch = NA]
  
  # Aggregate: compute max, min, mean for each source var, grouped by (cell_id, year)
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  # Build the aggregation call
  # For robustness with NA handling (max/min of empty set), we use a 
  # safe wrapper
  safe_max  <- function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x) }
  safe_min  <- function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else min(x) }
  safe_mean <- function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else mean(x) }
  
  agg_list <- lapply(source_vars, function(v) {
    vsym <- as.name(v)
    parse(text = sprintf(
      'list(neighbor_max_%s = safe_max(%s), neighbor_min_%s = safe_min(%s), neighbor_mean_%s = safe_mean(%s))',
      v, v, v, v, v, v
    ))
  })
  
  # Simpler and cleaner approach: melt is unnecessary; just aggregate directly
  stats_dt <- edge_year[, {
    out <- list()
    for (v in source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      n <- length(vals)
      out[[paste0("neighbor_max_", v)]]  <- if (n == 0L) NA_real_ else max(vals)
      out[[paste0("neighbor_min_", v)]]  <- if (n == 0L) NA_real_ else min(vals)
      out[[paste0("neighbor_mean_", v)]] <- if (n == 0L) NA_real_ else mean(vals)
    }
    out
  }, by = .(cell_id, year)]
  
  stats_dt
}

# Helper: cross-join edges with years
CJ_dt_edges <- function(edge_dt, years) {
  # Memory-efficient cross join
  yr_dt <- data.table(year = years)
  result <- edge_dt[, .(neighbor_id, year = list(years)), by = cell_id]
  result <- result[, .(neighbor_id = rep(neighbor_id, each = length(years)),
                        year = rep(years, times = .N / length(years))), 
                   by = cell_id]
  # Cleaner approach:
  result <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  result[, cell_id     := edge_dt$cell_id[edge_idx]]
  result[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
  result[, edge_idx := NULL]
  result
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Run the optimized pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
stats_dt <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Merge back into cell_data
setkey(cell_data, id, year)
setkey(stats_dt, cell_id, year)
setnames(stats_dt, "cell_id", "id")

cell_data <- stats_dt[cell_data, on = .(id, year)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained RF model object is preserved as-is. 
# cell_data now contains the same neighbor_max_*, neighbor_min_*, 
# neighbor_mean_* columns the model expects.
# predictions <- predict(rf_model, newdata = cell_data)
```

---

However, the cross-join approach above (~38.4M rows) may still be memory-heavy and the `by`-group aggregation with a `for` loop inside `j` is not maximally efficient. Here is a **cleaner, more memory-efficient final version** that processes one year at a time:

---

## Recommended Final Version (Memory-Safe, Fast)

```r
library(data.table)
setDT(cell_data)

# â”€â”€ Step 1: Build flat spatial edge table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  lens <- lengths(nb_obj)
  total <- sum(lens)
  from_id <- rep(id_order, times = lens)
  to_id   <- id_order[unlist(nb_obj)]
  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€ Step 2: Compute neighbor features year-by-year to control memory â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
keep_cols <- c("id", "year", neighbor_source_vars)

all_years <- sort(unique(cell_data$year))
result_list <- vector("list", length(all_years))

for (yi in seq_along(all_years)) {
  yr <- all_years[yi]
  
  # Subset this year's data
  yr_data <- cell_data[year == yr, ..keep_cols]
  
  # Join: for each edge, get the neighbor's variable values this year
  # edge_dt has (cell_id, neighbor_id); yr_data has (id, year, vars...)
  setkey(yr_data, id)
  merged <- edge_dt[yr_data, on = .(neighbor_id = id), nomatch = NA, allow.cartesian = TRUE]
  # merged now has columns: cell_id, neighbor_id, year, ntl, ec, ...
  # Each row = one (focal cell, neighbor) pair for this year
  # But we need to aggregate by cell_id (the focal cell)
  
  # Wait â€” the join direction matters. We want: for each focal cell, 
  # find its neighbors' values. So:
  # Start from edge_dt, join neighbor values from yr_data on neighbor_id = id
  merged <- merge(edge_dt, yr_data, by.x = "neighbor_id", by.y = "id", 
                  all.x = FALSE, allow.cartesian = FALSE)
  # merged: (cell_id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2)
  
  # Aggregate by cell_id
  agg_expr <- lapply(neighbor_source_vars, function(v) {
    call_list <- list(
      as.name(":="),
      setNames(list(
        substitute(if (.N == 0L || all(is.na(x))) NA_real_ else max(x, na.rm = TRUE), list(x = as.name(v))),
        substitute(if (.N == 0L || all(is.na(x))) NA_real_ else min(x, na.rm = TRUE), list(x = as.name(v))),
        substitute(if (.N == 0L || all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE), list(x = as.name(v)))
      ), paste0("neighbor_", c("max_", "min_", "mean_"), v))
    )
  })
  
  # Simpler: use a direct aggregation
  stats_yr <- merged[, {
    out <- vector("list", length(neighbor_source_vars) * 3L)
    k <- 0L
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      n <- length(vals)
      k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else max(vals)
      k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else min(vals)
      k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else mean(vals)
    }
    names(out) <- paste0("neighbor_", 
                         rep(c("max_", "min_", "mean_"), times = length(neighbor_source_vars)),
                         rep(neighbor_source_vars, each = 3L))
    out
  }, by = .(cell_id), .SDcols = neighbor_source_vars]
  
  stats_yr[, year := yr]
  result_list[[yi]] <- stats_yr
  
  if (yi %% 5 == 0) cat(sprintf("  Completed year %d (%d/%d)\n", yr, yi, length(all_years)))
}

all_stats <- rbindlist(result_list)
rm(result_list); gc()

# â”€â”€ Step 3: Merge back into cell_data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setnames(all_stats, "cell_id", "id")
setkey(all_stats, id, year)
setkey(cell_data, id, year)

# Add new columns to cell_data
new_cols <- setdiff(names(all_stats), c("id", "year"))
cell_data[all_stats, (new_cols) := mget(paste0("i.", new_cols)), on = .(id, year)]

# â”€â”€ Step 4: Predict with existing RF model (unchanged) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and What Changed

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Lookup construction** | 6.46M `lapply` iterations with `paste` + named-vector lookup | Vectorized `rep`/`unlist` on 344K-cell `nb` object â†’ flat edge table in <1 sec |
| **Neighbor value retrieval** | Per-row index subsetting via 6.46M-element list | `data.table` keyed merge (vectorized C-level hash join) |
| **Aggregation** | `lapply` over 6.46M list elements per variable (Ã—5) | `data.table` grouped `[, ..., by=cell_id]` â€” single pass per year |
| **Memory** | 6.46M-element list of integer vectors (~10â€“15 GB) | ~1.37M-row edge table (~22 MB) + one year's merge at a time (~5.5M rows, ~250 MB) |
| **Estimated time** | 86+ hours | **~5â€“15 minutes** |
| **RF model** | Unchanged âœ“ | Unchanged âœ“ |
| **Numerical output** | Same max/min/mean per (cell, year) | Identical âœ“ |

The trained Random Forest model is never touched â€” only the feature-engineering step is optimized. The computed `neighbor_max_*`, `neighbor_min_*`, and `neighbor_mean_*` columns are numerically identical to the original implementation, preserving the estimand.
END_RESPONSE>>>
