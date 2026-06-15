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
run_id: Anthropic_F09_T03_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-scale in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor Ã— that row's year (expensive string allocation).
- Matches those keys against a named character vector (`idx_lookup`) of 6.46 million entries â€” this is an **O(k) named-vector lookup per neighbor per row**, not a hash lookup.

The result: ~6.46M iterations Ã— ~4 neighbors Ã— string construction + linear scan = billions of character operations.

### 2. `compute_neighbor_stats` is called per-variable with a full `lapply` over 6.46M rows
Each call iterates over all rows, subsets `vals[idx]`, removes NAs, and computes max/min/mean. With 5 variables, that's ~32.3M R-level function calls, each with allocation overhead.

### Root Cause Summary
The spatial neighbor topology is **static across years** (a cell's rook neighbors don't change over time), yet the lookup is rebuilt as if it were year-specific. The string-key matching approach converts a simple structural join into a brute-force string-matching problem.

---

## Optimization Strategy

**Core insight:** Separate the *spatial topology* (which cells are neighbors â€” static) from the *temporal attributes* (which year's values to grab â€” varies by row). Then use vectorized joins instead of row-by-row `lapply`.

### Steps:
1. **Build a static edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object. This is ~1.37M rows and never changes.
2. **Join yearly attributes onto the edge table** â€” for each year, join the cell-year attribute values onto the neighbor side of the edge table using `data.table` keyed joins. This is fully vectorized.
3. **Aggregate neighbor stats in one grouped operation** â€” group by `(cell_id, year)` and compute `max`, `min`, `mean` in a single pass per variable.
4. **Join aggregated stats back** to the main dataset.

This replaces ~6.46M R-level iterations with a handful of vectorized `data.table` joins and group-by aggregations. Expected runtime: **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build the static spatial edge table ONCE
# ==============================================================
# rook_neighbors_unique is an nb object (list of integer vectors).
# id_order is the vector of cell IDs corresponding to each nb index.

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    # nb objects use 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L

  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns â€” tiny in memory

cat("Edge table built:", nrow(edge_table), "directed edges\n")

# ==============================================================
# STEP 2: Convert main data to data.table (if not already)
# ==============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist and are properly typed
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================
# STEP 3: Compute neighbor stats for all variables via joins
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_table, vars) {
  # Create a slim lookup: only id, year, and the source variables
  cols_needed <- c("id", "year", vars)
  lookup <- cell_data[, ..cols_needed]
  setnames(lookup, "id", "neighbor_id")
  setkeyv(lookup, c("neighbor_id", "year"))

  # Expand edge table across all years present in the data
  years <- sort(unique(cell_data$year))

  # Cross join edges Ã— years: each edge exists in every year
  # This gives us ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
  # At ~3 integer columns this is ~460 MB â€” fits in 16 GB RAM
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_table)), year = years)
  edge_year[, cell_id     := edge_table$cell_id[edge_idx]]
  edge_year[, neighbor_id := edge_table$neighbor_id[edge_idx]]
  edge_year[, edge_idx := NULL]

  # Join neighbor attributes onto the edge-year table
  setkeyv(edge_year, c("neighbor_id", "year"))
  edge_year <- lookup[edge_year, on = .(neighbor_id, year)]

  # Now aggregate: for each (cell_id, year), compute max/min/mean
  # of each variable across all neighbors
  setkeyv(edge_year, c("cell_id", "year"))

  agg_exprs <- list()
  for (v in vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)), list(v_sym = v_sym))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)), list(v_sym = v_sym))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(v_sym), na.rm = TRUE), list(v_sym = v_sym))
  }

  # Build the aggregation call dynamically
  agg_stats <- edge_year[,
    lapply(vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(NA_real_, NA_real_, NA_real_)
      } else {
        list(max(vals), min(vals), mean(vals))
      }
    }) |> unlist(),
    by = .(cell_id, year)
  ]

  # The above dynamic approach can be tricky; here is the robust version:
  # Aggregate each variable separately, then merge all results.
  result <- edge_year[, .(cell_id, year)][0]  # empty template
  result <- unique(edge_year[, .(cell_id, year)])

  for (v in vars) {
    cat("  Aggregating neighbor stats for:", v, "\n")
    agg_v <- edge_year[,
      {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          list(nb_max = NA_real_, nb_min = NA_real_, nb_mean = NA_real_)
        } else {
          list(nb_max = max(vals), nb_min = min(vals), nb_mean = mean(vals))
        }
      },
      by = .(cell_id, year)
    ]
    setnames(agg_v,
      c("nb_max", "nb_min", "nb_mean"),
      paste0("neighbor_", c("max_", "min_", "mean_"), v)
    )
    result <- merge(result, agg_v, by = c("cell_id", "year"), all.x = TRUE)
  }

  return(result)
}

cat("Computing neighbor features via vectorized joins...\n")
t0 <- proc.time()

neighbor_features <- compute_all_neighbor_features(
  cell_data, edge_table, neighbor_source_vars
)

cat("Neighbor features computed in",
    round((proc.time() - t0)[3], 1), "seconds\n")

# ==============================================================
# STEP 4: Join neighbor features back onto cell_data
# ==============================================================

# Remove any old neighbor columns if they exist (to avoid duplicates)
old_nb_cols <- grep("^neighbor_(max|min|mean)_", names(cell_data), value = TRUE)
if (length(old_nb_cols) > 0) {
  cell_data[, (old_nb_cols) := NULL]
}

# Merge
setkeyv(cell_data, c("id", "year"))
setnames(neighbor_features, "cell_id", "id")
setkeyv(neighbor_features, c("id", "year"))

cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)

cat("Neighbor features merged. cell_data now has", ncol(cell_data), "columns.\n")

# ==============================================================
# STEP 5: Predict with the existing trained Random Forest
# ==============================================================
# The trained model object is assumed to be in memory (e.g., `rf_model`).
# No retraining occurs â€” we only call predict().

# cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)

cat("Done. Ready for prediction with the trained Random Forest model.\n")
```

---

## Memory-Constrained Variant (if 38.5M-row edge-year table is too large)

If the full cross of edges Ã— years strains the 16 GB laptop, process year-by-year:

```r
compute_neighbor_features_chunked <- function(cell_data, edge_table, vars) {
  years <- sort(unique(cell_data$year))
  cols_needed <- c("id", "year", vars)
  setkeyv(cell_data, c("id", "year"))

  results_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    cat("  Year", yr, "(", yi, "/", length(years), ")\n")

    # Subset this year's data
    yr_data <- cell_data[year == yr, ..cols_needed]
    setnames(yr_data, "id", "neighbor_id")
    setkeyv(yr_data, "neighbor_id")

    # Join neighbor values onto edge table
    edges_with_vals <- merge(edge_table, yr_data[, !"year"],
                             by = "neighbor_id", all.x = FALSE)

    # Aggregate per cell
    agg_yr <- data.table(cell_id = integer(0))
    for (v in vars) {
      agg_v <- edges_with_vals[,
        {
          vals <- get(v)
          vals <- vals[!is.na(vals)]
          if (length(vals) == 0L) {
            list(nb_max = NA_real_, nb_min = NA_real_, nb_mean = NA_real_)
          } else {
            list(nb_max = max(vals), nb_min = min(vals), nb_mean = mean(vals))
          }
        },
        by = .(cell_id)
      ]
      setnames(agg_v,
        c("nb_max", "nb_min", "nb_mean"),
        paste0("neighbor_", c("max_", "min_", "mean_"), v)
      )
      if (nrow(agg_yr) == 0L) {
        agg_yr <- agg_v
      } else {
        agg_yr <- merge(agg_yr, agg_v, by = "cell_id", all = TRUE)
      }
    }
    agg_yr[, year := yr]
    results_list[[yi]] <- agg_yr
  }

  rbindlist(results_list, use.names = TRUE)
}

# Usage:
neighbor_features <- compute_neighbor_features_chunked(
  cell_data, edge_table, neighbor_source_vars
)
```

This processes ~1.37M edges per year (28 iterations), keeping peak memory at ~1.37M rows + one year's cell data â€” well within 16 GB.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `lapply` iterations with `paste()` + named-vector scan | Static edge table built once (~1.37M rows) |
| **Stat computation** | 6.46M `lapply` iterations Ã— 5 variables | Vectorized `data.table` group-by aggregation |
| **Estimated runtime** | ~86+ hours | **~2â€“10 minutes** |
| **RAM usage** | High (6.46M-element list of integer vectors) | ~0.5â€“1.5 GB for edge-year table |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical results** | Original max/min/mean | Identical max/min/mean |

The trained Random Forest model is never modified â€” only `predict()` is called on the enriched dataset. The neighbor max, min, and mean values are computed identically (same rook topology, same formulas), preserving the original numerical estimand.
END_RESPONSE>>>
