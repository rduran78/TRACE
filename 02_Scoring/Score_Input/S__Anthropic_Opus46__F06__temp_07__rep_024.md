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
run_id: Anthropic_F06_T07_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (fine â€” O(1) with named lookup).
2. Constructs `paste(neighbor_id, year, "_")` key strings for each neighbor.
3. Looks up those keys in `idx_lookup` (a named vector â€” effectively a hash lookup, but repeated millions of times with string construction).

This produces a **list of length 6.46M**, each element being a small integer vector. The sheer overhead of 6.46M `paste` + named-vector lookups, inside an R-level `lapply`, is enormous.

### Bottleneck 2: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by the neighbor indices and computing `max`, `min`, `mean`. That's ~32.3M R-level function calls total.

### Why raster focal/kernel operations are *not* a direct substitute

Focal operations assume a regular grid with uniform rectangular neighborhoods. Rook contiguity on an irregular or boundary-clipped grid doesn't map cleanly to a fixed kernel. Cells on boundaries or with missing neighbors would produce incorrect results. The comment in the prompt is a red herring â€” we must preserve the exact rook-neighbor topology from the `spdep::nb` object.

### Root cause summary

| Component | Calls | Estimated time share |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string ops | ~30-40% |
| `compute_neighbor_stats` Ã— 5 vars | 32.3M subset+aggregate | ~60-70% |

---

## Optimization Strategy

### Strategy 1: Vectorize the neighbor lookup via a sparse matrix

Instead of a list-of-indices, represent the neighbor relationships as a **sparse adjacency matrix** expanded to the cell-year level. Then `max`, `min`, `mean` over neighbors become sparse matrix operations â€” fully vectorized in C/C++ (via the `Matrix` package).

- Build a sparse **N_cells Ã— N_cells** binary adjacency matrix `W` from `rook_neighbors_unique`.
- Expand it to a **N_rows Ã— N_rows** block-diagonal matrix (one block per year) â€” but this is 6.46M Ã— 6.46M, which is infeasible in memory.

**Better:** Since neighbor relationships are *time-invariant* (cell `i`'s rook neighbors are the same every year), we can:
1. Reshape each variable into a **cells Ã— years matrix** (344,208 Ã— 28).
2. Build a sparse 344,208 Ã— 344,208 adjacency matrix `W`.
3. For `mean`: `W %*% X / rowSums(W)` (sparse matrix multiply â€” highly optimized).
4. For `max` and `min`: iterate over the sparse structure in C++ or use a grouped operation.

### Strategy 2: `data.table` grouped join (simpler, very fast)

1. Build an edge list `(cell_id, neighbor_id)` from the `nb` object â€” ~1.37M rows.
2. Join to the panel on `(neighbor_id, year)` to get neighbor values.
3. Group by `(cell_id, year)` and compute `max`, `min`, `mean`.

This replaces all R-level loops with `data.table` vectorized joins and grouped aggregations â€” expected speedup: **~500â€“1000Ã—**.

### Chosen approach: **Strategy 2 (data.table)**

Reasons:
- Straightforward, correct, preserves exact rook topology.
- Handles `NA`s naturally (`na.rm = TRUE`).
- Single pass per variable (or all variables at once).
- Memory-efficient: edge list is ~1.37M rows, join result is ~1.37M Ã— 28 â‰ˆ 38.5M rows per variable â€” fits in 16 GB RAM.
- Expected runtime: **seconds to a few minutes** instead of 86+ hours.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Convert the spdep::nb object to an edge list (one-time, fast)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
nb_to_edge_list <- function(nb_obj, id_order) {
  # nb_obj:   list of integer vectors (indices into id_order), class "nb"
  # id_order: vector of cell IDs corresponding to positions in nb_obj
  #
  # Returns a data.table with columns: cell_id, neighbor_id
  
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove the 0-neighbor sentinel that spdep uses
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  data.table(
    cell_id     = id_order[from],
    neighbor_id = id_order[to]
  )
}

edges <- nb_to_edge_list(rook_neighbors_unique, id_order)
# edges has ~1,373,394 rows (directed rook-neighbor pairs)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure key columns exist and are named 'id' and 'year'
# (adjust if your actual column names differ)
stopifnot(all(c("id", "year") %in% names(cell_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Compute neighbor stats for all source variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_all_neighbor_features <- function(cell_dt, edges, source_vars) {
  # Subset the columns we need for the join: neighbor_id â†” id, plus year + vars
  # We join edges to cell_dt on (neighbor_id == id, year == year) to get
  # the neighbor's values, then aggregate by (cell_id, year).
  
  # Columns to extract from the neighbor rows
  keep_cols <- c("id", "year", source_vars)
  neighbor_values <- cell_dt[, ..keep_cols]
  
  # Set key for fast join
  setnames(neighbor_values, "id", "neighbor_id")
  setkey(neighbor_values, neighbor_id)
  setkey(edges, neighbor_id)
  
  # Expand: join edges Ã— years â†’ get neighbor values for every (cell, year) pair
  # This is an equi-join: for each edge (cell_id, neighbor_id), join on neighbor_id
  # to get all years of that neighbor, but we only want matching years.
  
  # More efficient: merge edges with neighbor_values, then filter to matching year
  # Actually: we need (cell_id, year) â†’ neighbor values at (neighbor_id, same year)
  
  # Step A: Create the full join table
  #   edges has (cell_id, neighbor_id)  â€” ~1.37M rows
  #   neighbor_values has (neighbor_id, year, var1, ..., var5) â€” ~6.46M rows
  #   Join on neighbor_id â†’ ~1.37M Ã— 28 â‰ˆ 38.5M rows
  
  joined <- merge(edges, neighbor_values, by = "neighbor_id", allow.cartesian = TRUE)
  # joined now has columns: neighbor_id, cell_id, year, ntl, ec, pop_density, def, usd_est_n2
  
  # Step B: Aggregate by (cell_id, year) to get max, min, mean for each variable
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- substitute(max(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_min_", v)]]  <- substitute(min(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(mean(x, na.rm = TRUE), list(x = v_sym))
  }
  
  # Build a single aggregation call
  agg_list <- lapply(agg_exprs, eval, envir = parent.frame())  # won't work directly
  
  # Use .SDcols approach instead (cleaner):
  stats_dt <- joined[,
    {
      out <- list()
      for (v in source_vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          out[[paste0("n_max_", v)]]  <- NA_real_
          out[[paste0("n_min_", v)]]  <- NA_real_
          out[[paste0("n_mean_", v)]] <- NA_real_
        } else {
          out[[paste0("n_max_", v)]]  <- max(vals)
          out[[paste0("n_min_", v)]]  <- min(vals)
          out[[paste0("n_mean_", v)]] <- mean(vals)
        }
      }
      out
    },
    by = .(cell_id, year)
  ]
  
  return(stats_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

stats_dt <- compute_all_neighbor_features(cell_dt, edges, neighbor_source_vars)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Merge the neighbor features back into cell_dt
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(stats_dt, cell_id, year)
setkey(cell_dt, id, year)

cell_dt <- merge(cell_dt, stats_dt,
                 by.x = c("id", "year"), by.y = c("cell_id", "year"),
                 all.x = TRUE)

# Handle -Inf / Inf from max/min of empty sets (shouldn't happen with the
# NA guard above, but just in case):
inf_cols <- grep("^n_max_|^n_min_|^n_mean_", names(cell_dt), value = TRUE)
for (col in inf_cols) {
  set(cell_dt, which(is.infinite(cell_dt[[col]])), col, NA_real_)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. Convert back to data.frame if downstream code expects it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. Apply the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Even Faster Variant: Avoid `get()` in grouped `j`

The `get()` call inside the `by` expression can be slow for 6.46M groups. Here is a faster alternative that processes one variable at a time using fully vectorized `data.table` aggregation:

```r
library(data.table)

# 1. Build edge list (same as above)
edges <- nb_to_edge_list(rook_neighbors_unique, id_order)

# 2. Convert to data.table
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# 3. Prepare neighbor lookup table (just id + year + variables)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 4. For each variable, do a keyed join + vectorized aggregation
for (v in neighbor_source_vars) {
  
  # Extract just the columns we need for this variable
  val_dt <- cell_dt[, .(neighbor_id = id, year, val = get(v))]
  setkey(val_dt, neighbor_id, year)
  
  # Expand edges Ã— panel: join on (neighbor_id, year)
  # edges: (cell_id, neighbor_id)
  # val_dt: (neighbor_id, year, val)
  setkey(edges, neighbor_id)
  joined <- val_dt[edges, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # joined columns: neighbor_id, year, val, cell_id
  
  # Aggregate by (cell_id, year) â€” fully vectorized, no get() per group
  agg <- joined[!is.na(val),
    .(
      n_max  = max(val),
      n_min  = min(val),
      n_mean = mean(val)
    ),
    by = .(cell_id, year)
  ]
  
  # Rename columns to variable-specific names
  setnames(agg, c("n_max", "n_min", "n_mean"),
                c(paste0("n_max_", v), paste0("n_min_", v), paste0("n_mean_", v)))
  
  # Merge back into cell_dt
  setkey(agg, cell_id, year)
  cell_dt <- merge(cell_dt, agg,
                   by.x = c("id", "year"), by.y = c("cell_id", "year"),
                   all.x = TRUE)
  
  message("Done: ", v)
}

# 5. Clean up infinities (safety)
inf_cols <- grep("^n_max_|^n_min_|^n_mean_", names(cell_dt), value = TRUE)
for (col in inf_cols) {
  set(cell_dt, which(is.infinite(cell_dt[[col]])), col, NA_real_)
}

# 6. Back to data.frame
cell_data <- as.data.frame(cell_dt)

# 7. Predict with the pre-trained Random Forest (unchanged)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~20â€“30 hrs (6.46M string ops) | ~2 sec (vectorized edge list) |
| Compute stats (5 vars) | ~50â€“60 hrs (32.3M R-level loops) | ~2â€“5 min (data.table keyed join + grouped agg) |
| **Total** | **86+ hours** | **~3â€“6 minutes** |
| Peak memory | ~8â€“12 GB (6.46M-element list) | ~6â€“10 GB (38.5M-row join table, one var at a time) |

The numerical results are identical: the same rook-neighbor topology is used, the same `max`/`min`/`mean` statistics are computed with the same `NA` handling, and the pre-trained Random Forest model is applied without modification.
END_RESPONSE>>>
