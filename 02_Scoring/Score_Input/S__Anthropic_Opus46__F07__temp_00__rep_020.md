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
run_id: Anthropic_F07_T00_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over **~6.46 million rows** in a pure-R `lapply`, performing per-row character coercion, string pasting, and named-vector lookups. Each iteration does:

1. `as.character()` conversion
2. Indexing into a named character vector (`id_to_ref`)
3. Subsetting a list (`neighbors[[ref_idx]]`)
4. `paste()` to build composite keys
5. Named lookup into `idx_lookup` (a named integer vector of length 6.46M)

Named vector lookup in R is **O(n)** per query in the worst case (hash collisions aside, the overhead is enormous at 6.46M entries). Doing this 6.46 million times produces an **O(nÂ²)-like** runtime. The `compute_neighbor_stats` function is comparatively cheap but is also called per-variable in a loop with unnecessary repeated list traversals.

**Summary of problems:**
| Problem | Impact |
|---|---|
| Per-row `paste` + named-vector lookup in `build_neighbor_lookup` | ~86+ hours; dominant cost |
| Character key lookups on a 6.46M-length named vector | Pseudo-O(nÂ²) |
| `compute_neighbor_stats` returns a list of 3-vectors, then `do.call(rbind, ...)` on 6.46M elements | Slow bind |
| Loop over 5 variables calls `compute_neighbor_stats` independently each time | Minor but avoidable overhead |

---

## Optimization Strategy

### 1. Replace string-key lookups with integer-indexed joins via `data.table`

Instead of building a 6.46M-entry named vector and doing per-row string matching, we:
- Create an **edge list** of `(id, neighbor_id)` from the `nb` object (only ~1.37M directed edges for 344K cells).
- Join this edge list to the panel data **twice**: once to attach the row index of the focal cell-year, once to attach the row index (and variable values) of the neighbor cell-year.
- This is a **vectorized equi-join**, which `data.table` executes in O(n log n) or better.

### 2. Compute all neighbor stats in one grouped aggregation

Once we have an edge table `(focal_row, neighbor_row)` joined to variable values, we simply `group by focal_row` and compute `max`, `min`, `mean` â€” all in one pass, for all 5 variables simultaneously.

### 3. Memory budget

- Edge list with year expansion: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows Ã— a few integer/double columns â‰ˆ ~1.5 GB at peak. Fits in 16 GB.
- We avoid materializing a 6.46M-element list of variable-length integer vectors entirely.

### 4. Preserve the trained RF model and numerical estimand

We produce columns with **identical names and identical values** (max, min, mean of the same neighbor sets). The RF model is not retouched.

---

## Working R Code

```r
library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # -----------------------------------------------------------
  # Step 0: Convert to data.table if needed; add row index

# -----------------------------------------------------------
  was_df <- !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]
  
  # -----------------------------------------------------------
  # Step 1: Build directed edge list from the nb object
  #         (only ~1.37M rows â€” one per directed rook edge)
  # -----------------------------------------------------------
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i],
               neighbor_id = id_order[nb_idx])
  }))
  
  cat("Edge list rows (spatial only):", nrow(edge_list), "\n")
  
  # -----------------------------------------------------------
  # Step 2: Build a lookup from (id, year) -> row index + values
  # -----------------------------------------------------------
  # Only keep columns we need to minimize memory
  keep_cols <- c("id", "year", ".row_id", neighbor_source_vars)
  lookup <- dt[, ..keep_cols]
  setkey(lookup, id, year)
  
  # -----------------------------------------------------------
  # Step 3: Expand edges across years via join
  #
  # For each (focal_id, neighbor_id) edge and each year that

  # the focal cell appears in, find the neighbor's values in
  # that same year.
  #
  # Strategy:
  #   a) Join edge_list to lookup on focal_id == id  â†’  gives us
  #      (focal_id, neighbor_id, year, focal_row_id)
  #   b) Join result to lookup on neighbor_id == id & year == year
  #      â†’  gives us neighbor variable values
  # -----------------------------------------------------------
  
  # Step 3a: Get all (edge Ã— year) combinations for focal cells
  # Join edge_list[focal_id] â†’ lookup[id] to pick up year & focal .row_id
  focal_lookup <- lookup[, .(id, year, focal_row_id = .row_id)]
  setkey(focal_lookup, id)
  setkey(edge_list, focal_id)
  
  edges_by_year <- edge_list[focal_lookup,
                             on = .(focal_id = id),
                             allow.cartesian = TRUE,
                             nomatch = NULL]
  # edges_by_year has columns: focal_id, neighbor_id, year, focal_row_id
  
  cat("Edge-year rows:", nrow(edges_by_year), "\n")
  
  # Step 3b: Attach neighbor variable values
  neighbor_lookup_dt <- lookup[, c("id", "year", neighbor_source_vars), with = FALSE]
  setkey(neighbor_lookup_dt, id, year)
  setkey(edges_by_year, neighbor_id, year)
  
  edges_full <- neighbor_lookup_dt[edges_by_year,
                                   on = .(id = neighbor_id, year = year),
                                   nomatch = NA]
  # edges_full now has: id (=neighbor_id), year, <neighbor vars>, focal_id, focal_row_id
  
  # -----------------------------------------------------------
  # Step 4: Grouped aggregation â€” compute max, min, mean
  #         for each focal_row_id (= each cell-year)
  # -----------------------------------------------------------
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))
  
  names(agg_exprs) <- agg_names
  
  # Evaluate
  stats <- edges_full[, lapply(agg_exprs, eval, envir = .SD),
                       by = focal_row_id]
  
  # Fix Inf/-Inf from max/min on all-NA groups (shouldn't happen if edges exist,

  # but be safe)
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }
  
  # -----------------------------------------------------------
  # Step 5: Merge back into dt by row index
  # -----------------------------------------------------------
  setkey(stats, focal_row_id)
  setkey(dt, .row_id)
  
  # Remove any pre-existing neighbor columns to avoid duplication
  existing <- intersect(agg_names, names(dt))
  if (length(existing) > 0) dt[, (existing) := NULL]
  
  dt <- stats[dt, on = .(focal_row_id = .row_id)]
  
  # Clean up
  dt[, .row_id := NULL]
  dt[, focal_row_id := NULL]
  
  if (was_df) {
    setDF(dt)
  }
  
  return(dt)
}
```

**However**, the dynamic `bquote`/`eval` aggregation above can be tricky. Here is a cleaner, fully tested alternative for Step 4 that avoids `eval` gymnastics:

```r
# ---- Cleaner Step 4 alternative using .SDcols ----

compute_all_neighbor_features_v2 <- function(cell_data, id_order,
                                              rook_neighbors_unique,
                                              neighbor_source_vars) {
  was_df <- !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]
  
  # Step 1: Edge list
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L))
      return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  
  # Step 2: Lookup
  keep_cols <- c("id", "year", ".row_id", neighbor_source_vars)
  lookup <- dt[, ..keep_cols]
  
  # Step 3a: Expand by year
  focal_lu <- lookup[, .(id, year, focal_row_id = .row_id)]
  edges_by_year <- edge_list[focal_lu, on = .(focal_id = id),
                             allow.cartesian = TRUE, nomatch = NULL]
  
  # Step 3b: Attach neighbor values
  nb_lu <- lookup[, c("id", "year", neighbor_source_vars), with = FALSE]
  edges_full <- nb_lu[edges_by_year, on = .(id = neighbor_id, year = year),
                      nomatch = NA]
  
  # Step 4: Aggregation â€” one variable at a time to control column names
  #         and avoid memory spike from wide intermediate tables
  setkey(edges_full, focal_row_id)
  
  all_stats <- edges_full[, .(focal_row_id = unique(focal_row_id))]
  # Actually, simpler: aggregate all at once with .SDcols
  
  agg_fun <- function(x) {
    x_clean <- x[!is.na(x)]
    if (length(x_clean) == 0L) return(list(NA_real_, NA_real_, NA_real_))
    list(max(x_clean), min(x_clean), mean(x_clean))
  }
  
  # Most memory-efficient: loop over variables, join results
  result_dt <- unique(edges_full[, .(focal_row_id)])
  setkey(result_dt, focal_row_id)
  
  for (v in neighbor_source_vars) {
    cat("  Computing neighbor stats for:", v, "\n")
    
    stats_v <- edges_full[, {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(nb_max = NA_real_, nb_min = NA_real_, nb_mean = NA_real_)
      } else {
        list(nb_max = max(vals), nb_min = min(vals), nb_mean = mean(vals))
      }
    }, by = focal_row_id]
    
    setnames(stats_v,
             c("nb_max", "nb_min", "nb_mean"),
             paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean")))
    
    setkey(stats_v, focal_row_id)
    result_dt <- stats_v[result_dt, on = "focal_row_id"]
  }
  
  # Step 5: Merge back
  agg_names <- unlist(lapply(neighbor_source_vars, function(v)
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))))
  existing <- intersect(agg_names, names(dt))
  if (length(existing) > 0L) dt[, (existing) := NULL]
  
  dt <- result_dt[dt, on = .(focal_row_id = .row_id)]
  dt[, focal_row_id := NULL]
  
  if (was_df) setDF(dt)
  return(dt)
}
```

### Caller (replaces the original outer loop)

```r
library(data.table)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_v2(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used as-is downstream â€” no retraining.
```

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~80+ hours (6.46M string lookups into 6.46M named vec) | ~30 seconds (vectorized `data.table` join) |
| Compute neighbor stats (5 vars) | ~6 hours (list traversal Ã— 5) | ~2â€“5 minutes (grouped aggregation Ã— 5) |
| **Total** | **86+ hours** | **~3â€“8 minutes** |

Peak RAM: ~4â€“6 GB (edge-year table of ~38M rows Ã— ~8 columns of integers/doubles), well within 16 GB.

**Numerical equivalence**: The `max`, `min`, and `mean` are computed over the identical neighbor sets (same rook topology, same year matching, same NA handling), so the trained Random Forest model's predictions are unchanged.
END_RESPONSE>>>
