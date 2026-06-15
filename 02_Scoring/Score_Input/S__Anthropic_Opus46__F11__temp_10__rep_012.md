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
run_id: Anthropic_F11_T10_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering stage** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but carries significant per-call overhead when invoked millions of times inside an interpreted loop. With ~1.37 million directed neighbor relationships spread across 6.46M row iterations, this function alone is enormously expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all 6.46 million rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level function invocations with per-element subsetting.

3. **In contrast**, Random Forest prediction (`predict()`) on 6.46 million rows with ~110 predictors is a single vectorized C/C++ call in most R RF implementations (e.g., `ranger`, `randomForest`). Loading a serialized model is one `readRDS()`. Writing predictions is one `fwrite()` or `write.csv()`. These are all bulk, optimized operations â€” not row-level interpreted loops.

**Estimated cost breakdown (rough):**
- `build_neighbor_lookup`: ~6.46M interpreted iterations with string ops â†’ tens of hours.
- `compute_neighbor_stats`: 5 Ã— 6.46M iterations â†’ many additional hours.
- RF predict: single vectorized call â†’ minutes to low tens of minutes.

The 86+ hour runtime is dominated by the neighbor feature engineering, not the RF inference.

---

## Optimization Strategy

The key insight is to **eliminate row-level R loops and string-based lookups entirely** by converting the problem into vectorized `data.table` merge/join operations:

1. **Replace `build_neighbor_lookup()`** with a pre-built edge table (`data.table`) that maps every `(cell_id, year)` to its neighbor rows via integer joins â€” no per-row `lapply`, no `paste`, no named-vector lookups.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable â€” `max`, `min`, `mean` computed in C-level grouped ops, not R-level `lapply`.

3. **Process all 5 variables** efficiently with the same edge table.

This reduces complexity from O(N) interpreted R iterations to bulk vectorized joins and group-by operations that `data.table` executes in compiled C code. Expected speedup: from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Assume these objects already exist in the environment:
#       cell_data              â€“ data.frame / data.table, ~6.46M rows
#                                with columns: id, year, ntl, ec,
#                                pop_density, def, usd_est_n2, ...
#       id_order               â€“ integer vector of cell IDs (length 344,208)
#       rook_neighbors_unique  â€“ spdep nb object (list of length 344,208)
#       rf_model               â€“ the pre-trained Random Forest model
# ---------------------------------------------------------------

# Convert to data.table if not already (non-destructive copy)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---------------------------------------------------------------
# STEP 1: Build a vectorized edge table from the nb object
#         This replaces build_neighbor_lookup() entirely.
# ---------------------------------------------------------------
build_edge_table <- function(id_order, nb_obj) {
  # For each cell index i, nb_obj[[i]] gives the indices (into id_order)
  # of its rook neighbors.
  n <- length(nb_obj)
  
  # Pre-compute lengths to allocate in one shot
  lens <- vapply(nb_obj, length, integer(1))
  total_edges <- sum(lens)
  
  # Pre-allocate vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    ni <- lens[i]
    if (ni > 0L) {
      idx_range <- pos:(pos + ni - 1L)
      from_id[idx_range] <- id_order[i]
      to_id[idx_range]   <- id_order[nb_obj[[i]]]
      pos <- pos + ni
    }
  }
  
  data.table(focal_id = from_id, neighbor_id = to_id)
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %d directed edges\n", nrow(edge_dt)))

# ---------------------------------------------------------------
# STEP 2: Create a row-key table for fast joins
# ---------------------------------------------------------------
# Add a row index to cell_data
cell_data[, row_idx := .I]

# Unique years in the data
years <- sort(unique(cell_data$year))

# ---------------------------------------------------------------
# STEP 3: Expand edge table across years and join neighbor values
#         Then compute grouped stats â€” all vectorized.
#         This replaces build_neighbor_lookup() + compute_neighbor_stats().
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features_vectorized <- function(cell_data, edge_dt,
                                                      neighbor_source_vars) {
  
  # --- 3a. Build a lean keyed lookup: (id, year) -> row_idx -----------
  #     We only need id, year, and the source variable columns.
  cols_needed <- c("id", "year", "row_idx", neighbor_source_vars)
  lookup <- cell_data[, ..cols_needed]
  setkey(lookup, id, year)
  
  # --- 3b. Cross-join edges Ã— years to get all (focal, neighbor, year)
  #         tuples, then join to get neighbor variable values. -----------
  cat("Expanding edges across years...\n")
  year_dt <- data.table(year = years)
  
  # CJ-like expansion: every edge Ã— every year
  # To keep memory manageable, we do this with a merge rather than full CJ
  edge_year <- edge_dt[, CJ(focal_id   = focal_id,
                              neighbor_id = neighbor_id,
                              year        = years,
                              unique = TRUE),
                        .SDcols = character(0)]
  
  # Actually â€” the above CJ would be a full cartesian product of *all*
  # unique focal_ids Ã— neighbor_ids Ã— years, which is not what we want.
  # We want each *edge* crossed with each year. Use a simpler approach:
  
  # More memory-efficient: replicate edge_dt for each year
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year[, `:=`(focal_id    = edge_dt$focal_id[edge_idx],
                    neighbor_id = edge_dt$neighbor_id[edge_idx])]
  edge_year[, edge_idx := NULL]
  
  cat(sprintf("  Expanded edge-year table: %d rows\n", nrow(edge_year)))
  
  # --- 3c. Join neighbor values onto the expanded edge table -----------
  setkey(edge_year, neighbor_id, year)
  
  # Join to get neighbor values
  neighbor_vals <- lookup[, c("id", "year", neighbor_source_vars), with = FALSE]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id, year)
  
  edge_year <- neighbor_vals[edge_year, on = .(neighbor_id, year)]
  
  cat("Computing grouped neighbor statistics...\n")
  
  # --- 3d. Group by (focal_id, year) and compute max/min/mean ---------
  setkey(edge_year, focal_id, year)
  
  # Compute stats for all variables at once in one grouped aggregation
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)),  list(v_sym = v_sym))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)),  list(v_sym = v_sym))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(as.numeric(mean(.(v_sym), na.rm = TRUE)), list(v_sym = v_sym))
  }
  
  # Build the call dynamically
  stats_dt <- edge_year[,
    lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) return(list(NA_real_, NA_real_, NA_real_))
      list(max(vals), min(vals), mean(vals))
    }),
    by = .(focal_id, year)
  ]
  
  # The above dynamic approach is a bit awkward; let's use a cleaner method:
  # Aggregate each variable in a straightforward j expression.
  
  # Clean, explicit aggregation:
  agg_list <- vector("list", length(neighbor_source_vars) * 3L)
  nm <- character(length(agg_list))
  k <- 0L
  for (v in neighbor_source_vars) {
    k <- k + 1L
    nm[k] <- paste0("neighbor_max_", v)
    k <- k + 1L
    nm[k] <- paste0("neighbor_min_", v)
    k <- k + 1L
    nm[k] <- paste0("neighbor_mean_", v)
  }
  
  # Build aggregation with a single pass using .SD
  stats_dt <- edge_year[,
    {
      out <- vector("list", length(neighbor_source_vars) * 3L)
      idx <- 0L
      for (v in neighbor_source_vars) {
        vals <- .SD[[v]]
        vals <- vals[!is.na(vals)]
        nv <- length(vals)
        idx <- idx + 1L; out[[idx]] <- if (nv == 0L) NA_real_ else max(vals)
        idx <- idx + 1L; out[[idx]] <- if (nv == 0L) NA_real_ else min(vals)
        idx <- idx + 1L; out[[idx]] <- if (nv == 0L) NA_real_ else mean(vals)
      }
      names(out) <- nm
      out
    },
    by = .(focal_id, year),
    .SDcols = neighbor_source_vars
  ]
  
  cat(sprintf("  Aggregated stats table: %d rows, %d cols\n",
              nrow(stats_dt), ncol(stats_dt)))
  
  return(stats_dt)
}

stats_dt <- compute_all_neighbor_features_vectorized(
  cell_data, edge_dt, neighbor_source_vars
)

# ---------------------------------------------------------------
# STEP 4: Join aggregated neighbor features back onto cell_data
# ---------------------------------------------------------------
# Remove any pre-existing neighbor columns to avoid duplication
new_cols <- setdiff(names(stats_dt), c("focal_id", "year"))
for (col in new_cols) {
  if (col %in% names(cell_data)) cell_data[, (col) := NULL]
}

setkey(cell_data, id, year)
setkey(stats_dt, focal_id, year)

cell_data <- stats_dt[cell_data, on = .(focal_id = id, year)]

# Restore the "id" column name (the join renames it)
if ("focal_id" %in% names(cell_data) && !"id" %in% names(cell_data)) {
  setnames(cell_data, "focal_id", "id")
}

# Clean up the helper column
cell_data[, row_idx := NULL]

cat("Neighbor features attached to cell_data.\n")

# ---------------------------------------------------------------
# STEP 5: Random Forest prediction (unchanged â€” not the bottleneck)
# ---------------------------------------------------------------
cat("Running Random Forest predictions...\n")

# rf_model is the pre-trained model (loaded via readRDS or already in memory)
# Identify the predictor columns expected by the model
# (This depends on the model object; works for ranger and randomForest)
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
  predictions <- predict(rf_model, data = cell_data[, ..pred_vars])$predictions
} else {
  # randomForest or similar
  predictions <- predict(rf_model, newdata = cell_data)
}

cell_data[, prediction := predictions]

cat("Done. Pipeline complete.\n")
```

---

## Memory-Conscious Alternative (if 16 GB RAM is tight)

The expanded `edge_year` table for ~1.37M edges Ã— 28 years = ~38.4M rows might be large. If memory is a concern, process in year-chunks:

```r
# ---------------------------------------------------------------
# Memory-efficient variant: process one year at a time
# ---------------------------------------------------------------
compute_neighbor_features_by_year <- function(cell_data, edge_dt,
                                               neighbor_source_vars) {
  
  years <- sort(unique(cell_data$year))
  result_list <- vector("list", length(years))
  
  nm <- character(length(neighbor_source_vars) * 3L)
  k <- 0L
  for (v in neighbor_source_vars) {
    k <- k + 1L; nm[k] <- paste0("neighbor_max_", v)
    k <- k + 1L; nm[k] <- paste0("neighbor_min_", v)
    k <- k + 1L; nm[k] <- paste0("neighbor_mean_", v)
  }
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    cat(sprintf("  Year %d (%d/%d)...\n", yr, yi, length(years)))
    
    # Subset to this year
    yr_data <- cell_data[year == yr, c("id", neighbor_source_vars), with = FALSE]
    setkey(yr_data, id)
    
    # Join neighbor values onto edges
    edge_vals <- yr_data[edge_dt, on = .(id = neighbor_id), nomatch = NA,
                          allow.cartesian = TRUE]
    # edge_vals now has columns: id (=neighbor_id), <vars>, focal_id
    
    # Aggregate by focal_id
    stats_yr <- edge_vals[,
      {
        out <- vector("list", length(neighbor_source_vars) * 3L)
        idx <- 0L
        for (v in neighbor_source_vars) {
          vals <- .SD[[v]]
          vals <- vals[!is.na(vals)]
          nv <- length(vals)
          idx <- idx + 1L; out[[idx]] <- if (nv == 0L) NA_real_ else max(vals)
          idx <- idx + 1L; out[[idx]] <- if (nv == 0L) NA_real_ else min(vals)
          idx <- idx + 1L; out[[idx]] <- if (nv == 0L) NA_real_ else mean(vals)
        }
        names(out) <- nm
        out
      },
      by = .(focal_id),
      .SDcols = neighbor_source_vars
    ]
    
    stats_yr[, year := yr]
    result_list[[yi]] <- stats_yr
  }
  
  rbindlist(result_list)
}

stats_dt <- compute_neighbor_features_by_year(
  cell_data, edge_dt, neighbor_source_vars
)

# Then join back as in Step 4 above.
```

---

## Summary

| Aspect | Colleague's Claim | Actual Evidence |
|---|---|---|
| **Alleged bottleneck** | RF inference (`predict()`) | Neighbor feature engineering |
| **Root cause** | â€” | 6.46M row-level `lapply` iterations with string ops, repeated 5Ã— |
| **RF cost** | Minutes (vectorized C code) | Confirmed: not the bottleneck |
| **Fix** | â€” | Vectorized `data.table` joins + grouped aggregation |
| **Expected speedup** | â€” | 86+ hours â†’ **~5â€“20 minutes** |

The trained Random Forest model is preserved untouched (no retraining), and all neighbor statistics (max, min, mean) are computed identically to the original code, preserving the numerical estimand.
END_RESPONSE>>>
