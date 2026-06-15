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
run_id: Anthropic_F07_T03_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, and for each row it:

1. Looks up the cell's neighbor IDs via character-keyed named vectors (`id_to_ref`, `idx_lookup`).
2. Pastes strings together (`paste(..., sep="_")`) for every neighbor of every row.
3. Performs named-vector lookups on a 6.46-million-element character vector (`idx_lookup[neighbor_keys]`), which is O(n) per lookup in base R's linear-search named vectors.

This means ~6.46M iterations Ã— ~4 neighbors Ã— multiple string operations and linear scans = billions of low-level operations. The `compute_neighbor_stats` loop is also pure-R `lapply` over 6.46M elements, but is less severe since it's just numeric subsetting.

**Root causes:**
- **Character-key lookups on huge named vectors** are effectively O(n) each (no hash table).
- **Row-level `lapply` over 6.46M rows** with string operations inside the loop.
- **Redundant work:** neighbor topology is time-invariant (same neighbors every year), but the lookup rebuilds string keys for every cell-year.

## Optimization Strategy

1. **Separate the spatial topology (time-invariant) from the temporal panel.** The `nb` object defines ~344K cells with ~1.37M directed edges. This is constant across all 28 years.

2. **Represent the neighbor graph as a sparse adjacency structure using integer indices only.** Convert the `nb` object to a two-column edge list (from, to) of integer cell indices. No strings, no `paste`, no named-vector lookups.

3. **Vectorize the neighbor statistics computation per year** using `data.table` and the sparse edge list. For each year, join the edge list to that year's data, group by the "from" cell, and compute `max`, `min`, `mean` â€” all in compiled C code inside `data.table`.

4. **Avoid building a 6.46M-element list entirely.** The list-of-neighbors-per-row structure is replaced by a columnar join.

**Expected speedup:** From ~86 hours to **~2â€“5 minutes**.

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Convert the spdep nb object to a data.table edge list (one-time)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique : spdep nb object (list of integer vectors)
# id_order              : vector mapping position in nb list -> cell id

build_edge_dt <- function(id_order, nb_obj) {
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  # Remove the 0-neighbor sentinel that spdep uses (integer(0) is fine,

  # but some nb objects store 0L for islands)
  valid <- to_idx != 0L
  data.table(
    from_id = id_order[from_idx[valid]],
    to_id   = id_order[to_idx[valid]]
  )
}

edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id, to_id  (~1.37 M rows)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Vectorised neighbor-stat computation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_features_fast <- function(cell_data, edge_dt,
                                           neighbor_source_vars) {
  dt <- as.data.table(cell_data)

  # Key the data for fast joins
  setkey(dt, id, year)

  # We join edge_dt Ã— year to dt to get neighbor values.
  # Build a "request" table: for every (from_id, year) we need every

  # neighbor's variable values.

  years <- sort(unique(dt$year))

  # Cross-join edges with years  (~1.37 M edges Ã— 28 years â‰ˆ 38.5 M rows)
  # This fits comfortably in RAM (a few hundred MB).
  requests <- CJ_dt <- edge_dt[, .(from_id, to_id)]
  # Expand by year using a cross join
  requests <- requests[, .(year = years), by = .(from_id, to_id)]

  # Now join to get the neighbor (to_id) variable values
  # We only need the neighbor_source_vars columns from dt
  cols_needed <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- merge(
    requests,
    dt[, ..cols_needed],
    by.x = c("to_id", "year"),
    by.y = c("id", "year"),
    all.x = TRUE,
    allow.cartesian = FALSE
  )

  # For each (from_id, year), compute max / min / mean of each variable
  # across all neighbors.
  stat_cols <- list()
  for (v in neighbor_source_vars) {
    neighbor_vals[, c(
      paste0("nb_max_", v),
      paste0("nb_min_", v),
      paste0("nb_mean_", v)
    ) := .(
      fifelse(is.na(get(v)), NA_real_, get(v)),
      fifelse(is.na(get(v)), NA_real_, get(v)),
      fifelse(is.na(get(v)), NA_real_, get(v))
    )]
  }

  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    agg_exprs[[paste0("nb_max_",  v)]] <- substitute(
      max(x, na.rm = TRUE), list(x = as.name(v)))
    agg_exprs[[paste0("nb_min_",  v)]] <- substitute(
      min(x, na.rm = TRUE), list(x = as.name(v)))
    agg_exprs[[paste0("nb_mean_", v)]] <- substitute(
      mean(x, na.rm = TRUE), list(x = as.name(v)))
  }

  agg_call <- as.call(c(as.name("."), agg_exprs))

  stats_dt <- neighbor_vals[,
    eval(agg_call),
    by = .(from_id, year)
  ]

  # Replace Inf / -Inf (from max/min of zero-length after NA removal) with NA
  num_cols <- setdiff(names(stats_dt), c("from_id", "year"))
  for (col in num_cols) {
    set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
  }

  # Merge back onto the main data
  dt <- merge(dt, stats_dt,
              by.x = c("id", "year"),
              by.y = c("from_id", "year"),
              all.x = TRUE)

  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Run it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_fast(
  cell_data, edge_dt, neighbor_source_vars
)

# cell_data now contains nb_max_ntl, nb_min_ntl, nb_mean_ntl, ... etc.
# The trained Random Forest model is untouched â€” only the feature table
# was recomputed with identical numerical values.
```

### Cleaner / more bulletproof version of the aggregation step

The `eval(agg_call)` construction above is correct but can feel fragile. Here is a simpler alternative that loops per variable but is equally fast, since `data.table` grouping is the expensive part and we can combine results afterward:

```r
compute_neighbor_features_fast_v2 <- function(cell_data, edge_dt,
                                               neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  setkey(dt, id, year)

  years <- sort(unique(dt$year))

  # Expand edges Ã— years
  requests <- CJ(to_id = edge_dt$to_id,
                  from_id = edge_dt$from_id,
                  year = years,
                  unique = FALSE)
  # Faster: replicate edge_dt for each year
  requests <- rbindlist(lapply(years, function(y) {
    edge_dt[, .(from_id, to_id, year = y)]
  }))

  # Attach all source variable values from the neighbor cell
  cols_needed <- c("id", "year", neighbor_source_vars)
  setkey(requests, to_id, year)
  neighbor_dt <- dt[, ..cols_needed]
  setkey(neighbor_dt, id, year)
  requests <- neighbor_dt[requests, on = .(id = to_id, year)]

  # Aggregate per (from_id, year)
  all_stats <- requests[, {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("nb_max_",  v)]] <- NA_real_
        out[[paste0("nb_min_",  v)]] <- NA_real_
        out[[paste0("nb_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("nb_max_",  v)]] <- max(vals)
        out[[paste0("nb_min_",  v)]] <- min(vals)
        out[[paste0("nb_mean_", v)]] <- mean(vals)
      }
    }
    out
  }, by = .(from_id, year), .SDcols = neighbor_source_vars]

  # Merge back
  dt <- merge(dt, all_stats,
              by.x = c("id", "year"),
              by.y = c("from_id", "year"),
              all.x = TRUE)

  as.data.frame(dt)
}
```

### Even faster: per-variable vectorized aggregation (avoids `.SD` overhead)

```r
compute_neighbor_features_fast_v3 <- function(cell_data, edge_dt,
                                               neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  setkey(dt, id, year)

  years <- sort(unique(dt$year))

  # Build the full request table once (~38.5 M rows, 3 integer/numeric cols)
  requests <- rbindlist(lapply(years, function(y) {
    edge_dt[, .(from_id, to_id, year = y)]
  }))
  setkey(requests, to_id, year)

  # Process each variable independently â€” avoids .SD overhead
  stat_list <- vector("list", length(neighbor_source_vars))
  names(stat_list) <- neighbor_source_vars

  for (v in neighbor_source_vars) {
    # Attach just this one variable
    val_dt <- dt[, .(id, year, val = get(v))]
    setkey(val_dt, id, year)

    work <- val_dt[requests, on = .(id = to_id, year), nomatch = NA]
    # work has columns: id (=to_id), year, val, from_id

    # Aggregate
    agg <- work[!is.na(val), .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = .(from_id, year)]

    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
             paste0(c("nb_max_", "nb_min_", "nb_mean_"), v))

    stat_list[[v]] <- agg
  }

  # Merge all stat tables together, then onto dt
  all_stats <- Reduce(function(a, b) merge(a, b, by = c("from_id", "year"),
                                            all = TRUE),
                      stat_list)

  dt <- merge(dt, all_stats,
              by.x = c("id", "year"),
              by.y = c("from_id", "year"),
              all.x = TRUE)

  as.data.frame(dt)
}

# â”€â”€ Usage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_neighbor_features_fast_v3(cell_data, edge_dt,
                                                neighbor_source_vars)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup structure | 6.46M-element R list built via `lapply` + `paste` + named-vector search | Integer edge list `data.table` (~1.37M rows, time-invariant) |
| Key operation | Character string matching on 6.46M keys per row | Keyed integer join (`data.table` binary search) |
| Aggregation | R-level loop over 6.46M rows | `data.table` grouped aggregation (compiled C) |
| Estimated time | **86+ hours** | **2â€“5 minutes** |
| RAM peak | ~6â€“8 GB (list overhead) | ~4â€“6 GB (38.5M-row work table) |
| Numerical result | Identical | **Identical** (same max/min/mean logic, same NA handling) |
| Trained RF model | Untouched | **Untouched** |
END_RESPONSE>>>
