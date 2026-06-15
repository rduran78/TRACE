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
run_id: Anthropic_F11_T00_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential `lapply`.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million** R-level function calls total.

3. **`do.call(rbind, result)`** on a list of 6.46M small vectors is notoriously slow in R â€” it creates a massive argument list and binds row-by-row.

4. By contrast, Random Forest **prediction** on a pre-trained model is a single vectorized call (`predict(model, newdata)`) that runs in compiled C/C++ code. Even with 6.46M rows and 110 predictors, this typically completes in seconds to a few minutes. Loading a serialized model (`readRDS`) is also fast. Writing predictions is a single `fwrite` or `write.csv` call.

**Conclusion:** The 86+ hour runtime is dominated by the row-level R `lapply` loops in neighbor lookup construction and neighbor statistics computation, not by Random Forest inference.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` merge/join approach. Instead of looping over every row and doing per-row string operations, we expand the neighbor list into an edge table and join it against the data to get row indices â€” all in vectorized, compiled C code via `data.table`.

2. **Replace `compute_neighbor_stats()`** with a grouped `data.table` aggregation. Once we have an edge table mapping each row to its neighbor rows, we can extract neighbor values, group by the focal row, and compute `max`, `min`, `mean` in a single vectorized pass.

3. **Eliminate `do.call(rbind, ...)`** entirely â€” `data.table` aggregation returns a proper table directly.

4. **Process all 5 variables** in a single pass over the edge table rather than 5 separate `lapply` loops.

Expected speedup: from 86+ hours to **minutes** (roughly 1,000â€“10,000Ã— faster).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the neighbor edge table (replaces build_neighbor_lookup)
# ============================================================

build_neighbor_edges <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # Create mapping from position in id_order to cell id
  # rook_neighbors_unique[[i]] gives neighbor positions for id_order[i]
  
  n_ids <- length(id_order)
  
  # Expand the nb object into a focal_id -> neighbor_id edge list
  # Each element of rook_neighbors_unique is an integer vector of indices into id_order
  lengths_vec <- lengths(rook_neighbors_unique)
  focal_pos <- rep(seq_len(n_ids), times = lengths_vec)
  neighbor_pos <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- neighbor_pos > 0L
  focal_pos <- focal_pos[valid]
  neighbor_pos <- neighbor_pos[valid]
  
  edges <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  
  return(edges)
}

# ============================================================
# STEP 2: Compute all neighbor stats vectorized (replaces compute_neighbor_stats)
# ============================================================

compute_all_neighbor_features <- function(cell_data_dt, edges, neighbor_source_vars) {
  
  # Ensure cell_data_dt has a row index for fast joining
  cell_data_dt[, .row_idx := .I]
  
  # Get unique years
  years <- unique(cell_data_dt$year)
  
  # Build a keyed lookup: (id, year) -> row_idx plus the source variable values
  cols_needed <- c("id", "year", ".row_idx", neighbor_source_vars)
  lookup <- cell_data_dt[, ..cols_needed]
  setkey(lookup, id, year)
  
  # For each focal row, we need to find its neighbors in the same year.
  # Strategy: join cell_data (focal) with edges on focal_id = id,
  # then join the result with cell_data (neighbor) on neighbor_id = id AND same year.
  
  # Focal table: each row's id, year, and row index
  focal <- cell_data_dt[, .(focal_row = .row_idx, focal_id = id, year = year)]
  
  # Join focal with edges to get (focal_row, focal_id, year, neighbor_id)
  setkey(edges, focal_id)
  setkey(focal, focal_id)
  
  # This is the big expansion: each focal row x its neighbors
  expanded <- edges[focal, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # Columns: focal_id, neighbor_id, focal_row, year
  
  # Now join with lookup to get neighbor variable values
  setkey(expanded, neighbor_id, year)
  setkey(lookup, id, year)
  
  expanded <- lookup[expanded, on = c(id = "neighbor_id", "year"), nomatch = NA]
  # Now expanded has: id (=neighbor_id), year, .row_idx (neighbor's row), 
  # neighbor source var values, focal_row, focal_id
  
  # Compute grouped stats for each variable
  # Group by focal_row, compute max/min/mean of each neighbor source var
  
  agg_exprs <- list()
  for (var_name in neighbor_source_vars) {
    vn <- var_name
    max_name  <- paste0(vn, "_neighbor_max")
    min_name  <- paste0(vn, "_neighbor_min")
    mean_name <- paste0(vn, "_neighbor_mean")
    
    agg_exprs[[max_name]]  <- call("max",  as.name(vn), na.rm = TRUE)
    agg_exprs[[min_name]]  <- call("min",  as.name(vn), na.rm = TRUE)
    agg_exprs[[mean_name]] <- call("mean", as.name(vn), na.rm = TRUE)
  }
  
  # Build the aggregation call
  agg_result <- expanded[, 
    lapply(.SD, function(x) {
      x <- x[!is.na(x)]
      if (length(x) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(x), min(x), mean(x))
    }),
    by = focal_row,
    .SDcols = neighbor_source_vars
  ]
  
  # Actually, the above would return 3 rows per group. Better approach:
  # Compute each stat separately and merge, or use a cleaner aggregation.
  
  # Clean approach: one aggregation with explicit stat columns
  stat_list <- vector("list", length(neighbor_source_vars) * 3)
  names_list <- character(length(neighbor_source_vars) * 3)
  k <- 1
  for (vn in neighbor_source_vars) {
    for (stat in c("max", "min", "mean")) {
      names_list[k] <- paste0(vn, "_neighbor_", stat)
      k <- k + 1
    }
  }
  
  # More efficient: compute all at once with a single grouped operation
  agg_result <- expanded[, {
    out <- vector("list", length(neighbor_source_vars) * 3)
    k <- 1L
    for (vn in neighbor_source_vars) {
      vals <- get(vn)
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
    names(out) <- names_list
    out
  }, by = focal_row]
  
  # Now merge back into cell_data_dt by row index
  setkey(agg_result, focal_row)
  
  # For rows with no neighbors (islands), they won't appear in agg_result.
  # We need to handle them: left join from cell_data_dt
  new_cols <- setdiff(names(agg_result), "focal_row")
  cell_data_dt[agg_result, (new_cols) := mget(new_cols), on = c(.row_idx = "focal_row")]
  
  # Rows not matched get NA automatically (already the default for unmatched joins)
  
  cell_data_dt[, .row_idx := NULL]
  
  return(cell_data_dt)
}

# ============================================================
# STEP 3: Full optimized pipeline
# ============================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, 
                                   rf_model, neighbor_source_vars) {
  
  # Convert to data.table for performance
  cell_data_dt <- as.data.table(cell_data)
  
  # --- Neighbor feature engineering (THE ACTUAL BOTTLENECK, now optimized) ---
  message("Building neighbor edge table...")
  edges <- build_neighbor_edges(cell_data_dt, id_order, rook_neighbors_unique)
  
  message("Computing neighbor statistics for all variables...")
  cell_data_dt <- compute_all_neighbor_features(
    cell_data_dt, edges, neighbor_source_vars
  )
  
  message("Neighbor features complete.")
  
  # --- Random Forest inference (NOT the bottleneck) ---
  message("Running Random Forest predictions...")
  predictions <- predict(rf_model, newdata = cell_data_dt)
  cell_data_dt[, prediction := predictions]
  
  message("Writing predictions...")
  fwrite(cell_data_dt, "predictions_output.csv")
  
  message("Pipeline complete.")
  return(cell_data_dt)
}

# ============================================================
# USAGE
# ============================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# rf_model <- readRDS("trained_rf_model.rds")  # Preserved, not retrained
#
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model,
#   neighbor_source_vars   = neighbor_source_vars
# )
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with per-row `paste` + named-vector hash lookups | Vectorized `data.table` edge-table construction via `rep`/`unlist` |
| **Neighbor stats** | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind, ...)` | Single grouped `data.table` aggregation over pre-joined edge table |
| **Total R-level loop iterations** | ~38.8M+ | ~0 (all vectorized/compiled) |
| **Estimated runtime** | 86+ hours | Minutes (typically 5â€“20 min depending on RAM/disk) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical estimand** | Preserved âœ“ | Preserved âœ“ (identical max/min/mean logic, same NA handling) |

The colleague should focus optimization efforts on the spatial feature engineering, not on Random Forest inference.
END_RESPONSE>>>
