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
run_id: Anthropic_F11_T07_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence points clearly to a different bottleneck: **the spatial neighbor feature engineering pipeline**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named character vector lookup in R is O(n) in the worst case and extremely slow at scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this function alone generates and looks up **~25.8 million** string keys against a named vector of 6.46M entries. This is the dominant cost.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46M rows with `lapply`, subsetting numeric vectors, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also expensive.

3. **Random Forest inference** (`predict()`) on a pre-trained model with ~110 predictors and 6.46M rows is a single vectorized C-level call (in `ranger` or `randomForest`). It's comparatively fast â€” typically minutes, not hours.

**The 86+ hour runtime is dominated by the row-level `lapply` loops with string manipulation and named-vector lookups over millions of rows.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup` entirely** with a vectorized `data.table` equi-join. Instead of building string keys and doing named-vector lookups row-by-row, we:
   - Expand the neighbor list into a two-column edge table (`id`, `neighbor_id`).
   - Join against the data on (`neighbor_id`, `year`) to get row indices of neighbors directly.
   - Group by the focal row to get a list of neighbor row-indices.

2. **Replace `compute_neighbor_stats`** with a single vectorized `data.table` grouped aggregation over the edge table. Instead of `lapply` over 6.46M entries, we join the edge table to the variable column and compute `max/min/mean` per group in one pass â€” fully vectorized in C.

3. **Process all 5 variables in one pass** over the edge table rather than 5 separate `lapply` loops.

This reduces estimated runtime from **86+ hours to ~5â€“15 minutes** on the same hardware.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build the edge table (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, rook_neighbors_unique) {
  # Expand the nb object into a data.table of directed edges: focal_id -> neighbor_id
  # id_order[i] is the cell id for the i-th entry in the nb list.
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute and attach all neighbor features at once
#    (replaces compute_neighbor_stats + outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # Convert to data.table if not already; keep original row order
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # --- Step 1: Build edge table (focal_id -> neighbor_id) ---
  edges <- build_edge_table(id_order, rook_neighbors_unique)
  # edges has columns: focal_id, neighbor_id

  # --- Step 2: Cross with years to get (focal_id, year, neighbor_id, year) ---
  #   Instead of crossing, join edges to data twice:
  #   - First join: get the focal row index
  #   - Second join: get the neighbor row's variable values
  #
  #   But we can be smarter: join edges to dt on neighbor side to get
  #   neighbor values, then aggregate by (focal_id, year).

  # Key the data for fast joins
  setkey(dt, id, year)

  # We need focal_id + year to identify each focal row.
  # For each (focal_id, year), the neighbors are (neighbor_id, same year).

  # Create a join table: every (focal_id, year) paired with its neighbor_ids
  # by joining edges to the unique (id, year) combos in dt.

  focal_keys <- dt[, .(focal_id = id, year, .row_id)]

  # Join focal_keys with edges on focal_id
  # Result: for each focal row, all its neighbor_ids (to be looked up at same year)
  setkey(edges, focal_id)
  setkey(focal_keys, focal_id)

  # This is the big expansion: ~6.46M rows Ã— ~4 neighbors = ~25.8M rows
  expanded <- edges[focal_keys,
                    .(focal_row_id = .row_id, year = i.year, neighbor_id),
                    on = .(focal_id),
                    allow.cartesian = TRUE,
                    nomatch = NA]

  # Drop rows where neighbor_id is NA (cells with no neighbors)
  expanded <- expanded[!is.na(neighbor_id)]

  # --- Step 3: Look up neighbor values by joining on (neighbor_id, year) ---
  # Subset dt to only the columns we need for the join
  cols_needed <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- dt[, ..cols_needed]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  # Equi-join: attach neighbor variable values
  merged <- neighbor_vals[expanded, on = .(neighbor_id, year), nomatch = NA]

  # --- Step 4: Aggregate per focal row: max, min, mean for each variable ---
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", v, c("_max", "_min", "_mean"))
  }))

  # Build the aggregation call dynamically
  agg_list <- setNames(agg_exprs, agg_names)

  # Evaluate aggregation grouped by focal_row_id
  stats <- merged[,
    lapply(agg_list, eval, envir = .SD),
    by = .(focal_row_id)
  ]

  # The above dynamic approach can be tricky; here is a cleaner equivalent:
  # We build a simpler aggregation using .SDcols
  stats <- merged[,
    {
      out <- list()
      for (v in neighbor_source_vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          out[[paste0("neighbor_", v, "_max")]]  <- NA_real_
          out[[paste0("neighbor_", v, "_min")]]  <- NA_real_
          out[[paste0("neighbor_", v, "_mean")]] <- NA_real_
        } else {
          out[[paste0("neighbor_", v, "_max")]]  <- max(vals)
          out[[paste0("neighbor_", v, "_min")]]  <- min(vals)
          out[[paste0("neighbor_", v, "_mean")]] <- mean(vals)
        }
      }
      out
    },
    by = .(focal_row_id)
  ]

  # --- Step 5: Join aggregated stats back to dt by row id ---
  setkey(stats, focal_row_id)
  setkey(dt, .row_id)

  dt <- stats[dt, on = .(focal_row_id = .row_id)]

  # Rows with no neighbors will have NA (correct behavior, matches original)
  # Replace -Inf/Inf from max/min of empty sets with NA
  for (v in neighbor_source_vars) {
    for (suffix in c("_max", "_min", "_mean")) {
      col <- paste0("neighbor_", v, suffix)
      if (col %in% names(dt)) {
        set(dt, which(is.infinite(dt[[col]])), col, NA_real_)
      }
    }
  }

  # Clean up helper columns
  dt[, focal_row_id := NULL]

  # Return as data.frame if the original was a data.frame
  if (!is.data.table(cell_data)) {
    setDF(dt)
  }

  dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Usage â€” drop-in replacement for the original outer loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Random Forest prediction â€” UNCHANGED (not the bottleneck)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The pre-trained model is loaded and used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste` + named-vector lookup | Single `data.table` equi-join on integer keys |
| **Neighbor stats** | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind, ...)` | One grouped `data.table` aggregation over ~25.8M edge rows |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **RF model** | Preserved (unchanged) | Preserved (unchanged) |
| **Numerical results** | Original `max/min/mean` | Identical `max/min/mean` â€” same estimand |

**The bottleneck is the neighbor feature engineering, not Random Forest inference.** The fix is to replace row-level R loops and string-key lookups with vectorized `data.table` joins and grouped aggregations.
END_RESPONSE>>>
