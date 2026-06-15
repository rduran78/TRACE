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
run_id: Anthropic_F09_T10_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all 6.46M cell-year rows with expensive per-row operations
The function iterates over every cell-year row, performs character-based key lookups (`paste`, named-vector indexing) for each row's neighbors for each year. Because the spatial neighbor topology is **time-invariant** (a cell's rook neighbors don't change from year to year), this recomputes the same spatial relationships 28 times (once per year). That's ~6.46M iterations with string manipulation and named-vector lookups inside each iteration.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows
Each call iterates over every row, subsetting a numeric vector by index, removing NAs, and computing max/min/mean. This is called 5 times (once per source variable), totaling ~32.3M individual R-level function calls.

### 3. The fundamental design flaw: the lookup conflates spatial topology with temporal panel structure
The neighbor lookup embeds *both* the spatial adjacency *and* the year-matching into one monolithic list of length 6.46M. This means you cannot exploit the factored structure: **neighbors are spatial, attributes are spatiotemporal**. A join-based strategy on a static adjacency table is dramatically faster.

---

## Optimization Strategy

**Core insight:** Build the adjacency table **once** as a two-column `data.table` (`id`, `neighbor_id`) with ~1.37M rows. Then for each year, join the cell attributes onto both sides of this table and compute grouped summary statistics. This replaces 6.46M R-level iterations with vectorized `data.table` grouped joins, reducing runtime from ~86 hours to **minutes**.

Steps:
1. Convert `rook_neighbors_unique` (spdep nb object) into a static `data.table` edge list: `(id, neighbor_id)`.
2. Convert `cell_data` to `data.table`.
3. For each neighbor source variable, join `cell_data` onto the edge list by `(neighbor_id, year)` to attach the neighbor's value, then compute `max`, `min`, `mean` grouped by `(id, year)`.
4. Join results back onto `cell_data`.
5. Feed augmented `cell_data` into the existing trained Random Forest via `predict()` â€” no retraining.

This preserves the original numerical estimand because the same neighbor topology and the same summary statistics (max, min, mean of rook neighbors' attribute values) are computed; only the *method of computation* changes.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static spatial adjacency edge-list ONCE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique : spdep nb object (list of integer neighbor indices)
# id_order              : vector of cell IDs in the same order as the nb object

build_adjacency_table <- function(id_order, neighbors) {
  # neighbors is a list of length N; each element is an integer vector of
  # neighbor indices (referencing positions in id_order), or 0L for no neighbors.
  edges <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    # spdep encodes "no neighbors" as a single 0
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  setkey(edges, id)
  edges
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)
# adj_table has ~1,373,394 rows: (id, neighbor_id)
# This is built ONCE and can be serialized for future runs:
# fst::write_fst(adj_table, "adj_table.fst")

cat("Adjacency table rows:", nrow(adj_table), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor stats for all source variables via vectorized joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-select only the columns we need for neighbor lookups (keep it lean)
# We will process one variable at a time to limit peak memory.

# Add a year column to the adjacency table via a cross-join approach:
#   For each year, join neighbor attributes.
# But more efficient: expand adj_table Ã— years only implicitly via keyed join.

compute_all_neighbor_features <- function(cell_dt, adj_table, source_vars) {
  
  # We need: for each (id, year), the values of source_vars at all (neighbor_id, year).
  # Strategy: join adj_table with cell_dt on neighbor_id == id, then group by (id, year).
  
  # Prepare a slim lookup: just id, year, and the source variables
  lookup_cols <- c("id", "year", source_vars)
  lookup <- cell_dt[, ..lookup_cols]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id)
  
  # Join: for every edge, attach the neighbor's year-specific values
  # adj_table has (id, neighbor_id); lookup has (neighbor_id, year, var1, var2, ...)
  # This is a many-to-many join: each edge appears for each year the neighbor_id has data.
  # Result: (id, neighbor_id, year, var1, var2, ...)
  
  # Use merge for clarity; data.table makes this fast with keys
  setkey(adj_table, neighbor_id)
  setkey(lookup, neighbor_id)
  
  # This join produces nrow(adj_table) * 28 â‰ˆ 38.5M rows â€” fits in 16GB RAM
  # (1.37M edges Ã— 28 years Ã— ~7 columns Ã— 8 bytes â‰ˆ 2.4 GB)
  
  cat("Performing adjacency-attribute join...\n")
  joined <- adj_table[lookup, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NULL]
  # joined columns: id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2
  
  cat("Join complete. Rows:", nrow(joined), "\n")
  
  # Now group by (id, year) and compute max, min, mean for each source variable
  cat("Computing grouped neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
    )
  }))
  
  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  names(agg_exprs) <- agg_names
  
  # Evaluate the aggregation
  stats <- joined[, 
    lapply(agg_exprs, eval, envir = .SD), 
    by = .(id, year)
  ]
  
  # The above dynamic approach can be tricky; here's an explicit, robust version:
  stats <- joined[, {
    out <- list()
    for (v in source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_max_", v)]]  <- NA_real_
        out[[paste0("neighbor_min_", v)]]  <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_", v)]]  <- max(vals)
        out[[paste0("neighbor_min_", v)]]  <- min(vals)
        out[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    out
  }, by = .(id, year)]
  
  cat("Stats computed. Rows:", nrow(stats), "\n")
  
  setkey(stats, id, year)
  stats
}

neighbor_stats <- compute_all_neighbor_features(cell_dt, adj_table, neighbor_source_vars)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Join neighbor stats back onto the main data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Remove old neighbor columns if they exist (to avoid duplicates)
old_neighbor_cols <- grep("^neighbor_(max|min|mean)_", names(cell_dt), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_dt[, (old_neighbor_cols) := NULL]
}

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

cat("Final cell_dt rows:", nrow(cell_dt), " cols:", ncol(cell_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the existing trained Random Forest (NO retraining)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rf_model is the pre-trained randomForest / ranger object already in memory.
# Ensure column names match the training feature set.

cell_dt[, prediction := predict(rf_model, newdata = cell_dt)]

# Convert back to data.frame if downstream code expects it:
cell_data <- as.data.frame(cell_dt)
```

---

## Memory-Constrained Variant (if 38.5M-row join is too large)

If the single join exceeds available RAM, process year-by-year in a loop â€” still vastly faster than the original because each iteration is a vectorized `data.table` operation over ~1.37M edges rather than an `lapply` over 6.46M rows:

```r
compute_neighbor_features_by_year <- function(cell_dt, adj_table, source_vars) {
  
  years <- sort(unique(cell_dt$year))
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    cat("Processing year", yr, "(", yi, "/", length(years), ")\n")
    
    # Subset to this year
    dt_yr <- cell_dt[year == yr, c("id", "year", source_vars), with = FALSE]
    
    # Create neighbor lookup: rename id -> neighbor_id for join
    nb_lookup <- copy(dt_yr)
    setnames(nb_lookup, "id", "neighbor_id")
    nb_lookup[, year := NULL]
    setkey(nb_lookup, neighbor_id)
    
    # Join: attach neighbor attributes to each edge
    setkey(adj_table, neighbor_id)
    joined <- adj_table[nb_lookup, on = "neighbor_id", nomatch = NULL]
    # joined: (id, neighbor_id, ntl, ec, pop_density, def, usd_est_n2)
    
    # Aggregate by id
    stats_yr <- joined[, {
      out <- list()
      for (v in source_vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          out[[paste0("neighbor_max_", v)]]  <- NA_real_
          out[[paste0("neighbor_min_", v)]]  <- NA_real_
          out[[paste0("neighbor_mean_", v)]] <- NA_real_
        } else {
          out[[paste0("neighbor_max_", v)]]  <- max(vals)
          out[[paste0("neighbor_min_", v)]]  <- min(vals)
          out[[paste0("neighbor_mean_", v)]] <- mean(vals)
        }
      }
      out
    }, by = .(id)]
    
    stats_yr[, year := yr]
    result_list[[yi]] <- stats_yr
  }
  
  rbindlist(result_list)
}

neighbor_stats <- compute_neighbor_features_by_year(cell_dt, adj_table, neighbor_source_vars)
setkey(neighbor_stats, id, year)

# Join and predict as before (Steps 4â€“5 above)
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor topology | Rebuilt per cell-year (6.46M iterations) | Built once (344K cells â†’ 1.37M edges) |
| Key lookup method | Character paste + named vector | Integer-keyed `data.table` join |
| Stats computation | `lapply` over 6.46M rows Ã— 5 vars | Vectorized `data.table` group-by |
| Estimated runtime | ~86+ hours | **~5â€“15 minutes** |
| RAM peak | Moderate (but slow) | ~2â€“4 GB (full join) or ~0.5 GB (year-by-year) |
| Numerical output | âœ… Identical max/min/mean | âœ… Identical max/min/mean |
| RF model | Untouched | Untouched â€” `predict()` only |
END_RESPONSE>>>
