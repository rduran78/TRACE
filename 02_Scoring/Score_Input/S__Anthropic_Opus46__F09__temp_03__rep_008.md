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
run_id: Anthropic_F09_T03_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(n) over all 6.46 million cell-year rows
For every single cell-year row, the function:
- Looks up the cell's spatial neighbors.
- Pastes together `(neighbor_id, year)` string keys.
- Performs named-vector character lookups into `idx_lookup` (a named character vector of length 6.46M).

Named-vector lookups in R are **hash-based but with per-call overhead**, and doing ~6.46 million `paste` + lookup operations inside an `lapply` is brutally slow. Critically, **the spatial neighbor topology is identical across all 28 years**â€”the same cell always has the same rook neighbors regardless of year. Yet this function redundantly recomputes the neighbor-key mapping for every year of every cell.

### 2. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Each call to `compute_neighbor_stats` iterates over all 6.46M rows, subsetting a numeric vector by index and computing `max`, `min`, `mean`. This is done 5 times (once per source variable), totaling ~32.3 million R-level function calls. The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive.

### Root cause summary
The neighbor topology is **static across years** but is being resolved at the **cell-year level**. The entire approach can be restructured: build the spatial adjacency table **once** (344K cells), then use fast vectorized joins and grouped aggregations per year.

---

## Optimization Strategy

1. **Build a static neighbor edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` representing all ~1.37M directed rook-neighbor pairs. This is year-invariant.

2. **Join yearly attributes onto the edge table** â€” for each year, join the cell-year attributes onto the `neighbor_id` column. This turns the problem into a standard grouped aggregation.

3. **Compute neighbor stats via `data.table` grouped aggregation** â€” group by `(cell_id, year)` and compute `max`, `min`, `mean` for each variable in one vectorized pass.

4. **Join results back** to the main dataset.

This replaces ~6.46M R-level `lapply` iterations with vectorized `data.table` joins and `by=` aggregations, reducing runtime from ~86 hours to **minutes**.

### Complexity comparison

| Step | Current | Proposed |
|---|---|---|
| Neighbor resolution | 6.46M `paste` + named-vector lookups | 1 merge on 1.37M Ã— 28 = ~38.4M rows |
| Stats computation (per var) | 6.46M `lapply` calls | 1 grouped `data.table` aggregation |
| Total R function calls | ~38.7M | ~5 (one per variable) |

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# plus all other predictor columns. We preserve it fully.
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static spatial neighbor edge table ONCE
#
# rook_neighbors_unique is an nb object (list of integer index vectors).
# id_order is the vector of cell IDs in the same order as the nb object.
# We expand it into a two-column edge table: (cell_id, neighbor_id).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edge_table <- function(id_order, neighbors) {
  # neighbors is a list of length length(id_order);
  # neighbors[[i]] contains integer indices into id_order for cell i's neighbors.
  # We expand to a long-form edge table.
  n <- length(id_order)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    # spdep::nb objects use 0L to indicate no neighbors
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) > 0L) {
      from_list[[i]] <- rep(id_order[i], length(nb_idx))
      to_list[[i]]   <- id_order[nb_idx]
    }
  }
  
  data.table(
    cell_id     = unlist(from_list, use.names = FALSE),
    neighbor_id = unlist(to_list,   use.names = FALSE)
  )
}

# Build once â€” ~1.37M rows, year-invariant
neighbor_edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Neighbor edge table: %s directed edges for %s cells\n",
  format(nrow(neighbor_edges), big.mark = ","),
  format(length(id_order), big.mark = ",")
))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor stats for all variables via join + groupby
#
# Strategy:
#   - Cross-join neighbor_edges with years â†’ ~38.4M rows
#     (but we do it implicitly via a keyed merge to avoid materializing
#      the full cross product in memory).
#   - For each year, join cell attributes onto neighbor_id.
#   - Group by (cell_id, year) â†’ compute max, min, mean.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, neighbor_edges, source_vars) {
  
  # Extract only the columns we need for the neighbor attribute lookup
  # to minimize memory during the join.
  lookup_cols <- c("id", "year", source_vars)
  attr_dt <- cell_data[, ..lookup_cols]
  setnames(attr_dt, "id", "neighbor_id")
  
  # Key for fast join
  setkey(attr_dt, neighbor_id, year)
  
  # Expand neighbor_edges Ã— year by joining:
  #   neighbor_edges (cell_id, neighbor_id)
  #     âŸ• attr_dt (neighbor_id, year, var1, var2, ...)
  # This gives us one row per (cell_id, neighbor_id, year) with the
  # neighbor's attribute values attached.
  
  # To keep memory manageable on a 16 GB laptop, we process year-by-year.
  years <- sort(unique(cell_data$year))
  
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Subset neighbor attributes for this year
    attr_yr <- attr_dt[year == yr]
    attr_yr[, year := NULL]  # drop year column for the join; we'll add it back
    setkey(attr_yr, neighbor_id)
    
    # Join: for each edge, attach the neighbor's attribute values in this year
    # This produces ~1.37M rows (one per directed edge)
    edges_with_attrs <- neighbor_edges[attr_yr, on = "neighbor_id", nomatch = 0L, allow.cartesian = FALSE]
    
    # Now group by cell_id and compute stats
    # Build aggregation expressions dynamically
    agg_exprs <- unlist(lapply(source_vars, function(v) {
      list(
        bquote(max(.(as.name(v)),   na.rm = TRUE)),
        bquote(min(.(as.name(v)),   na.rm = TRUE)),
        bquote(mean(.(as.name(v)),  na.rm = TRUE))
      )
    }))
    
    agg_names <- unlist(lapply(source_vars, function(v) {
      paste0("neighbor_", c("max_", "min_", "mean_"), v)
    }))
    
    names(agg_exprs) <- agg_names
    
    # Evaluate the grouped aggregation
    stats_yr <- edges_with_attrs[,
      lapply(agg_exprs, eval, envir = .SD),
      by = cell_id
    ]
    
    stats_yr[, year := yr]
    result_list[[yi]] <- stats_yr
    
    if (yi %% 5 == 0 || yi == length(years)) {
      cat(sprintf("  Processed year %d (%d/%d)\n", yr, yi, length(years)))
    }
  }
  
  rbindlist(result_list, use.names = TRUE)
}

# --- Actually, the dynamic bquote approach above is fragile. ---
# Here is a cleaner, robust version:

compute_all_neighbor_features <- function(cell_data, neighbor_edges, source_vars) {
  
  lookup_cols <- c("id", "year", source_vars)
  attr_dt <- cell_data[, ..lookup_cols]
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)
  
  years <- sort(unique(cell_data$year))
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Neighbor attributes for this year only
    attr_yr <- attr_dt[J(unique(neighbor_edges$neighbor_id), yr), nomatch = 0L]
    attr_yr[, year := NULL]
    setkey(attr_yr, neighbor_id)
    
    # Attach neighbor attributes to each edge
    merged <- merge(neighbor_edges, attr_yr, by = "neighbor_id", all.x = FALSE)
    
    # Grouped aggregation: max, min, mean of each source var, grouped by cell_id
    # We build the aggregation as a single expression string for efficiency.
    agg_expr_parts <- unlist(lapply(source_vars, function(v) {
      c(
        sprintf("neighbor_max_%s  = suppressWarnings(max(%s, na.rm = TRUE))", v, v),
        sprintf("neighbor_min_%s  = suppressWarnings(min(%s, na.rm = TRUE))", v, v),
        sprintf("neighbor_mean_%s = mean(%s, na.rm = TRUE)", v, v)
      )
    }))
    
    agg_expr_str <- paste0("list(", paste(agg_expr_parts, collapse = ", "), ")")
    agg_expr <- parse(text = agg_expr_str)
    
    stats_yr <- merged[, eval(agg_expr), by = cell_id]
    
    # Replace Inf/-Inf (from max/min on all-NA) with NA
    for (col_name in names(stats_yr)) {
      if (is.numeric(stats_yr[[col_name]])) {
        set(stats_yr, which(is.infinite(stats_yr[[col_name]])), col_name, NA_real_)
      }
    }
    
    stats_yr[, year := yr]
    result_list[[yi]] <- stats_yr
    
    if (yi %% 5 == 0 || yi == length(years)) {
      cat(sprintf("  Processed year %d (%d/%d)\n", yr, yi, length(years)))
    }
  }
  
  rbindlist(result_list, use.names = TRUE)
}

cat("Computing neighbor features...\n")
t0 <- proc.time()

neighbor_features <- compute_all_neighbor_features(
  cell_data, neighbor_edges, neighbor_source_vars
)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Neighbor features computed in %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Join neighbor features back onto cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Rename cell_id back to id for the join
setnames(neighbor_features, "cell_id", "id")
setkey(neighbor_features, id, year)
setkey(cell_data, id, year)

# Remove old neighbor columns if they exist (from a prior run)
old_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

# Merge
cell_data <- neighbor_features[cell_data, on = c("id", "year")]

cat(sprintf("Final dataset: %s rows Ã— %s columns\n",
            format(nrow(cell_data), big.mark = ","),
            ncol(cell_data)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the existing trained Random Forest model
#
# The trained model object (e.g., rf_model) is preserved as-is.
# No retraining occurs. We simply call predict().
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Ensure all predictor columns expected by the model are present
# (The neighbor features now have the same column names as before.)

cell_data[, predicted := predict(rf_model, newdata = cell_data)]

cat("Prediction complete. Trained RF model was NOT retrained.\n")
```

---

## Expected Performance

| Metric | Before (current) | After (optimized) |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M string ops) | **~2 seconds** (344K-cell edge table) |
| `compute_neighbor_stats` (Ã—5 vars) | ~80+ hours (32.3M R calls) | **~3â€“8 minutes** (28 vectorized joins + groupby) |
| Peak RAM | Moderate (but slow) | ~3â€“5 GB (fits in 16 GB) |
| **Total wall time** | **~86+ hours** | **~5â€“15 minutes** |

### Why this preserves correctness
- The neighbor edge table encodes **exactly** the same rook-neighbor relationships from `rook_neighbors_unique`.
- The `max`, `min`, `mean` aggregations are computed on the same neighbor attribute values as before.
- `Inf`/`-Inf` from `max`/`min` on empty groups are converted to `NA`, matching the original `c(NA, NA, NA)` fallback.
- The trained Random Forest model is loaded and used for `predict()` onlyâ€”**no retraining**.
- The original numerical estimand (the predicted values and all input features) is preserved identically.
END_RESPONSE>>>
