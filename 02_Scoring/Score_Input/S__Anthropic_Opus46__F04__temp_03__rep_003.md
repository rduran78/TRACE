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
run_id: Anthropic_F04_T03_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a named character vector of length 6.46M â€” every lookup is an O(n) hash probe against a very large vector).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop is catastrophically slow. The named-vector `idx_lookup` of length ~6.46M means each key lookup is expensive, and it is performed for every neighbor of every row (total lookups â‰ˆ number of directed neighbor-year pairs â‰ˆ 1.37M neighbors Ã— 28 years â‰ˆ 38.5M lookups against a 6.46M-entry named vector).

### 2. `compute_neighbor_stats` â€” Row-level `lapply` over 6.46 million rows (Ã—5 variables)

Each iteration subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per variable), so ~32.3 million R function calls total. The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also slow.

### Summary of cost

| Component | Iterations | Dominant cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M | String pasting + named vector lookup (~38.5M probes into 6.46M-entry vector) |
| `compute_neighbor_stats` | 6.46M Ã— 5 vars | Interpreted loop + per-row subsetting + `do.call(rbind, ...)` on 6.46M-element list |

Estimated wall time: 86+ hours is consistent with this analysis.

---

## Optimization Strategy

**Replace all row-level R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is a **join** problem. Every cell-year row needs the variable values of its rook neighbors in the same year. This is a classic equi-join that `data.table` handles in seconds.

**Steps:**

1. **Build an edge table** (a two-column data.table of `id â†’ neighbor_id`) from the `spdep::nb` object. This is ~1.37M rows and is built once.
2. **Join** the edge table to the panel data on `(neighbor_id, year)` to retrieve neighbor variable values. This replaces both `build_neighbor_lookup` and the subsetting inside `compute_neighbor_stats`.
3. **Grouped aggregation** (`max`, `min`, `mean`) by `(id, year)` computes all neighbor stats in one vectorized pass per variable (or all variables at once).

This reduces the entire operation from ~86 hours to **minutes** (typically 2â€“5 minutes on a 16 GB laptop).

The trained Random Forest model is untouched. The numerical results (max, min, mean of neighbor values) are identical.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build the directed edge table from the nb object (once)
# ---------------------------------------------------------------
# id_order is the vector of cell IDs in the order matching rook_neighbors_unique.
# rook_neighbors_unique is an spdep::nb object (list of integer index vectors).

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      from_id[pos:(pos + n_i - 1L)] <- id_order[i]
      to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
# edges is ~1.37M rows, two integer columns â€” trivial memory.

# ---------------------------------------------------------------
# STEP 2: Convert panel data to data.table and set keys
# ---------------------------------------------------------------
dt <- as.data.table(cell_data)

# We need a fast lookup table: given (neighbor_id, year) â†’ variable values.
# Create a copy keyed on (id, year) for joining.
setkey(dt, id, year)

# ---------------------------------------------------------------
# STEP 3: For each variable, join + aggregate in one vectorized pass
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(dt, edges, source_vars) {
  # Build the join table: every (id, year) paired with its neighbor_ids.
  # edges has columns: id, neighbor_id
  # dt has columns: id, year, <variables...>
  
  # Expand edges by year: for each row in dt, attach its neighbor_ids.
  # Efficient approach: join dt[, .(id, year)] to edges, then join
  # the result to dt on (neighbor_id, year) to get neighbor values.
  
  # Step A: Get unique (id, year) pairs and cross with edges
  id_year <- dt[, .(id, year)]
  
  # Merge id_year with edges on 'id' to get (id, year, neighbor_id)
  # This is ~1.37M neighbors Ã— 28 years â‰ˆ 38.5M rows â€” fits in 16 GB easily.
  setkey(edges, id)
  setkey(id_year, id)
  expanded <- edges[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded columns: id, neighbor_id, year
  
  # Step B: Join to dt to get neighbor variable values
  # We need dt's variable columns keyed by (id, year), but here the join key
  # is (neighbor_id, year) â†’ dt's (id, year).
  
  # Select only the columns we need from dt for the join
  lookup_cols <- c("id", "year", source_vars)
  dt_lookup <- dt[, ..lookup_cols]
  setnames(dt_lookup, "id", "neighbor_id")
  setkey(dt_lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  joined <- dt_lookup[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # joined columns: neighbor_id, year, <source_vars>, id
  
  # Step C: Grouped aggregation by (id, year)
  # Compute max, min, mean for each source variable
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
  stats <- joined[,
    setNames(lapply(source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(NA_real_, NA_real_, NA_real_)
      } else {
        list(max(vals), min(vals), mean(vals))
      }
    }), source_vars),
    by = .(id, year)
  ]
  
  # The above is slightly awkward; cleaner approach below:
  # Compute all stats in one grouped operation.
  
  stats <- joined[, {
    out <- vector("list", length(source_vars) * 3L)
    k <- 1L
    for (v in source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k]]     <- NA_real_
        out[[k + 1]] <- NA_real_
        out[[k + 2]] <- NA_real_
      } else {
        out[[k]]     <- max(vals)
        out[[k + 1]] <- min(vals)
        out[[k + 2]] <- mean(vals)
      }
      k <- k + 3L
    }
    names(out) <- agg_names
    out
  }, by = .(id, year)]
  
  return(stats)
}

neighbor_stats <- compute_all_neighbor_features(dt, edges, neighbor_source_vars)

# ---------------------------------------------------------------
# STEP 4: Merge back into the main data.table
# ---------------------------------------------------------------
setkey(neighbor_stats, id, year)
setkey(dt, id, year)
dt <- neighbor_stats[dt, on = c("id", "year")]

# Handle cells with no neighbors (rows not in neighbor_stats already get NA
# from the right join above, which is the correct behavior matching the original).

# Convert back to data.frame if the downstream RF predict() expects one:
cell_data <- as.data.frame(dt)

# ---------------------------------------------------------------
# STEP 5: Run the (already trained) Random Forest prediction
# ---------------------------------------------------------------
# The trained model object is unchanged. Example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Even Cleaner Aggregation (Alternative Step C)

If the `get()` inside `by` grouping feels slow on 38.5M rows, you can compute each variable's stats separately with fully vectorized `data.table` syntax, which avoids any interpreted inner loop:

```r
# Alternative: fully vectorized, one variable at a time
add_neighbor_features_fast <- function(dt, edges, source_vars) {
  
  id_year <- dt[, .(id, year)]
  setkey(edges, id)
  setkey(id_year, id)
  expanded <- edges[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  
  for (v in source_vars) {
    # Build a small lookup: (neighbor_id, year) â†’ value
    lk <- dt[, .(neighbor_id = id, year, val = get(v))]
    setkey(lk, neighbor_id, year)
    setkey(expanded, neighbor_id, year)
    
    tmp <- lk[expanded, on = c("neighbor_id", "year"), nomatch = NA]
    
    agg <- tmp[!is.na(val), .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ), by = .(id, year)]
    
    new_names <- paste0("neighbor_", c("max_", "min_", "mean_"), v)
    setnames(agg, c("nmax", "nmin", "nmean"), new_names)
    
    setkey(agg, id, year)
    setkey(dt, id, year)
    dt <- agg[dt, on = c("id", "year")]
  }
  
  return(dt)
}

dt <- as.data.table(cell_data)
setkey(dt, id, year)
dt <- add_neighbor_features_fast(dt, edges, neighbor_source_vars)
cell_data <- as.data.frame(dt)
```

---

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string ops) | ~2 sec (vectorized edge table) |
| Expand edges Ã— years | N/A (implicit in loop) | ~5 sec (keyed join, ~38.5M rows) |
| Join neighbor values | ~hours (named vector probes) | ~10 sec per variable (keyed equi-join) |
| Grouped aggregation | ~hours (6.46M Ã— 5 `lapply`) | ~20 sec per variable (vectorized `data.table` groupby) |
| **Total neighbor features** | **86+ hours** | **~2â€“5 minutes** |
| RF prediction | unchanged | unchanged |

**Memory footprint:** The expanded join table is ~38.5M rows Ã— 3â€“4 columns of integers/doubles â‰ˆ ~1â€“1.5 GB, well within 16 GB RAM.

**Numerical equivalence:** The `max`, `min`, and `mean` computations over the same neighbor sets with the same NA-removal logic produce identical results. The trained Random Forest model is never modified.
END_RESPONSE>>>
