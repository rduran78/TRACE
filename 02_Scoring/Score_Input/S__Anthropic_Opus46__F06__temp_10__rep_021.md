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
run_id: Anthropic_F06_T10_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row lookups via `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs string keys by pasting neighbor IDs with the current year.
4. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) â€” these are **hash lookups on a character vector of 6.46M entries**, repeated for every row.

The result is a list of ~6.46M integer vectors. This is extremely slow and memory-hungry because:
- **String pasting and hash lookups** inside a loop over millions of rows is inherently slow in R.
- The `idx_lookup` named vector of length 6.46M has poor cache performance.
- The output list itself consumes significant memory.

### Bottleneck B: `compute_neighbor_stats` â€” per-row `lapply` over 6.46M entries

For each of the 5 variables, this iterates over 6.46M list elements, subsets a numeric vector, removes NAs, and computes max/min/mean. While each operation is trivial, doing it 6.46M Ã— 5 = ~32.3M times in interpreted R is very slow.

### Why raster focal/kernel operations are not directly applicable

The comment in the docstring asks about raster focal operations. Focal operations (e.g., `terra::focal`) assume a **regular grid with uniform kernel**. If the 344K cells form a regular rectangular grid, focal operations *could* work, but:
- Panel data means each "layer" is a year â€” focal operations work spatially within a single layer.
- Rook neighbors from `spdep::nb` may encode irregular boundaries (coastal cells, edge cells with fewer than 4 neighbors), which focal operations handle via `na.rm=TRUE` padding.
- The critical issue is that **the `nb` object is already computed and may reflect an irregular subset of a grid** (e.g., only land cells). A focal approach would require reconstructing the full rectangular grid and mapping cells back, which risks altering the estimand.

**Conclusion:** The correct approach is to **vectorize the neighbor computation using sparse matrix multiplication and grouped operations via `data.table`**, not focal raster operations. This preserves the exact `nb` structure and numerical results.

---

## 2. Optimization Strategy

### Step 1: Replace `build_neighbor_lookup` with a sparse adjacency matrix

Construct a sparse **N_cells Ã— N_cells** adjacency matrix `W` from `rook_neighbors_unique`. This is done once, costs negligible time, and uses the `Matrix` package.

### Step 2: Vectorize neighbor stats computation per year

For each year, extract the column vector `x` of length N_cells for a given variable. Then:
- `W %*% x` gives the **sum** of neighbor values.
- `W %*% (x != NA)` (with proper NA handling) gives the **count**.
- Mean = sum / count.
- For **max** and **min**, use a row-wise sparse iteration or, better, use `data.table` joins on the edge list.

Since sparse matrix multiplication doesn't directly yield row-wise max/min, the most efficient general approach is:

### Step 3: Edge-list + `data.table` grouped aggregation

Convert the `nb` object to an **edge list** (from_id, to_id) â€” about 1.37M rows. Then for each year and each variable:
1. Join the edge list to the data to get neighbor values.
2. Group by `from_id` and compute `max`, `min`, `mean`.
3. Join results back.

This replaces 6.46M R-level list iterations with **vectorized `data.table` grouped operations** over ~1.37M edges Ã— 28 years = ~38.5M rows, which `data.table` handles in seconds.

**Expected speedup:** From 86+ hours to **~2â€“5 minutes**.

### Numerical equivalence

The edge list is derived from the identical `nb` object, the same grouping (cell Ã— year), and the same `max`, `min`, `mean` functions. The results are **numerically identical** to the original implementation. The trained Random Forest model is never touched.

---

## 3. Working R Code

```r
library(data.table)
library(spdep)

# ---------------------------------------------------------------
# 0. Load pre-existing objects (assumed already in environment)
#    - cell_data        : data.frame with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#    - id_order         : integer/character vector of cell IDs (the ordering used by the nb object)
#    - rook_neighbors_unique : an nb object (list of integer index vectors)
#    - rf_model         : the pre-trained Random Forest model (untouched)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 1. Build edge list from the nb object (once, ~1.37M rows)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, nb_obj) {
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove zero-neighbor placeholders (spdep uses 0L for no-neighbor entries)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  data.table(
    from_id = id_order[from],
    to_id   = id_order[to]
  )
}

edge_list <- build_edge_list(id_order, rook_neighbors_unique)

cat("Edge list rows:", nrow(edge_list), "\n")

# ---------------------------------------------------------------
# 2. Convert cell_data to data.table and set keys
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Ensure id and year are keyed for fast joins
setkey(cell_dt, id, year)

# ---------------------------------------------------------------
# 3. Vectorised neighbor feature computation
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_dt, edge_list, source_vars) {
  
  # Create a long table of (from_id, year) Ã— neighbor value
  # by joining edge_list to cell_dt on to_id = id
  
  # We need: for each (from_id, year), the values of each source_var at to_id
  # Step: expand edge_list Ã— years via join
  
  # Prepare a slim lookup: id, year, and source_vars only
  lookup_cols <- c("id", "year", source_vars)
  neighbor_vals <- cell_dt[, ..lookup_cols]
  setnames(neighbor_vals, "id", "to_id")
  setkey(neighbor_vals, to_id)
  
  # Join edge_list to neighbor_vals: for each (from_id, to_id) get all years of to_id
  # This creates ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
  cat("Joining edge list with panel data...\n")
  
  # Keyed join: edge_list[to_id] â†’ neighbor_vals[to_id, year, vars]
  setkey(edge_list, to_id)
  expanded <- neighbor_vals[edge_list, on = "to_id", allow.cartesian = TRUE]
  # Result columns: to_id, year, <source_vars>, from_id
  
  cat("Expanded edge-year table rows:", nrow(expanded), "\n")
  
  # Group by (from_id, year), compute max/min/mean for each variable
  cat("Computing grouped neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  names(agg_exprs) <- agg_names
  
  # Compute all aggregations in one grouped operation
  stats <- expanded[,
    lapply(agg_exprs, eval),
    by = .(from_id, year)
  ]
  
  # Handle Inf/-Inf from max/min on all-NA groups â†’ set to NA
  for (v in source_vars) {
    max_col <- paste0("neighbor_max_", v)
    min_col <- paste0("neighbor_min_", v)
    stats[is.infinite(get(max_col)), (max_col) := NA_real_]
    stats[is.infinite(get(min_col)), (min_col) := NA_real_]
  }
  
  setnames(stats, "from_id", "id")
  return(stats)
}

neighbor_stats <- compute_all_neighbor_features(cell_dt, edge_list, neighbor_source_vars)

# ---------------------------------------------------------------
# 4. Merge neighbor features back to cell_dt
# ---------------------------------------------------------------
cat("Merging neighbor features back to main table...\n")

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- grep("^neighbor_", names(cell_dt), value = TRUE)
if (length(existing_neighbor_cols) > 0) {
  cell_dt[, (existing_neighbor_cols) := NULL]
}

setkey(neighbor_stats, id, year)
setkey(cell_dt, id, year)
cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

# ---------------------------------------------------------------
# 5. Convert back to data.frame if needed downstream
# ---------------------------------------------------------------
cell_data <- as.data.frame(cell_dt)

cat("Done. New columns added:\n")
print(grep("^neighbor_", names(cell_data), value = TRUE))

# ---------------------------------------------------------------
# 6. Predict with the untouched pre-trained RF model
# ---------------------------------------------------------------
# predictions <- predict(rf_model, newdata = cell_data)
```

### Alternative: More Memory-Efficient Chunked Version

If the ~38.5M-row expanded table risks exceeding 16 GB RAM (each row with 5 doubles â‰ˆ 1.5 GB + keys), process **year-by-year**:

```r
compute_neighbor_features_chunked <- function(cell_dt, edge_list, source_vars) {
  
  years <- sort(unique(cell_dt$year))
  lookup_cols <- c("id", "year", source_vars)
  
  results_list <- vector("list", length(years))
  
  for (i in seq_along(years)) {
    yr <- years[i]
    cat("Processing year", yr, "(", i, "/", length(years), ")\n")
    
    # Subset to this year
    yr_data <- cell_dt[year == yr, ..lookup_cols]
    setnames(yr_data, "id", "to_id")
    setkey(yr_data, to_id)
    
    # Join: each edge gets the to_id's values for this year
    expanded_yr <- yr_data[edge_list, on = "to_id", nomatch = NA]
    
    # Aggregate by from_id
    agg_list <- list()
    for (v in source_vars) {
      vcol <- expanded_yr[[v]]
      agg_list[[paste0("neighbor_max_", v)]] <- bquote(
        {tmp <- .(as.name(v)); tmp <- tmp[!is.na(tmp)];
         if(length(tmp)==0) NA_real_ else max(tmp)}
      )
    }
    
    # Simpler approach: direct computation
    agg_result <- expanded_yr[, {
      res <- list()
      for (vv in source_vars) {
        vals <- .SD[[vv]]
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0) {
          res[[paste0("neighbor_max_", vv)]]  <- NA_real_
          res[[paste0("neighbor_min_", vv)]]  <- NA_real_
          res[[paste0("neighbor_mean_", vv)]] <- NA_real_
        } else {
          res[[paste0("neighbor_max_", vv)]]  <- max(vals)
          res[[paste0("neighbor_min_", vv)]]  <- min(vals)
          res[[paste0("neighbor_mean_", vv)]] <- mean(vals)
        }
      }
      res
    }, by = .(from_id), .SDcols = source_vars]
    
    agg_result[, year := yr]
    setnames(agg_result, "from_id", "id")
    results_list[[i]] <- agg_result
  }
  
  rbindlist(results_list)
}

neighbor_stats <- compute_neighbor_features_chunked(cell_dt, edge_list, neighbor_source_vars)
```

### Highest-Performance Version (Recommended)

This version avoids `.SD` overhead and uses **pre-melted** edge+value joins with native `data.table` aggregation, one variable at a time:

```r
compute_neighbor_features_fast <- function(cell_dt, edge_list, source_vars) {
  
  all_stats <- list()
  setkey(edge_list, to_id)
  
  for (v in source_vars) {
    cat("Computing neighbor stats for:", v, "\n")
    
    # Slim lookup: id, year, value
    lkp <- cell_dt[, .(to_id = id, year, val = get(v))]
    setkey(lkp, to_id)
    
    # Join edges â†’ values: (from_id, to_id, year, val)
    joined <- lkp[edge_list, on = "to_id", allow.cartesian = TRUE, nomatch = NA]
    # Columns: to_id, year, val, from_id
    
    # Aggregate by (from_id, year)
    agg <- joined[!is.na(val), .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ), by = .(from_id, year)]
    
    setnames(agg, c("nmax", "nmin", "nmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))
    setnames(agg, "from_id", "id")
    setkey(agg, id, year)
    
    all_stats[[v]] <- agg
  }
  
  # Merge all variable stats together
  result <- all_stats[[1]]
  for (k in seq_along(all_stats)[-1]) {
    result <- result[all_stats[[k]], on = .(id, year)]
  }
  
  result
}

# --- Full pipeline ---
library(data.table)

cell_dt <- as.data.table(cell_data)
edge_list <- build_edge_list(id_order, rook_neighbors_unique)

neighbor_stats <- compute_neighbor_features_fast(cell_dt, edge_list, neighbor_source_vars)

# Merge back
setkey(cell_dt, id, year)
setkey(neighbor_stats, id, year)
existing_neighbor_cols <- grep("^neighbor_", names(cell_dt), value = TRUE)
if (length(existing_neighbor_cols) > 0) cell_dt[, (existing_neighbor_cols) := NULL]

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]
cell_data <- as.data.frame(cell_dt)

# Predict (RF model is untouched)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string-paste + hash lookups in R loop | One-time edge list from `nb` object (~ms) |
| **Stats computation** | 6.46M Ã— 5 `lapply` iterations | 5 vectorized `data.table` group-by operations |
| **Estimated runtime** | 86+ hours | **2â€“10 minutes** |
| **Peak RAM** | Large (6.46M-element list of vectors) | ~2â€“3 GB for largest join table |
| **Numerical equivalence** | Baseline | Identical (same edges, same max/min/mean) |
| **RF model** | Untouched | Untouched |
END_RESPONSE>>>
