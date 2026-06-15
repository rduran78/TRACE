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
run_id: Anthropic_F12_T03_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" inside `compute_neighbor_stats()`; it's a straightforward `lapply` returning pre-allocated length-3 vectors. That part is O(n) and relatively efficient.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive character key construction and named-vector lookup**: For each of the ~6.46 million rows, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to build character keys, then does named-vector indexing into `idx_lookup` (itself a named vector of length ~6.46 million). Named-vector lookup in R is **O(n)** per query (linear scan or hash with overhead), and this is done for every neighbor of every row. With ~1.37 million directed neighbor relationships spread across 28 years, this produces roughly **38.4 million key lookups** into a 6.46-million-entry named vector â€” all inside a sequential `lapply` with per-element `paste` and `as.character` overhead.

2. **Redundant recomputation across years**: The spatial neighbor structure is **time-invariant** â€” cell A's rook neighbors are the same in every year. Yet `build_neighbor_lookup` recomputes the neighbor-row mapping for every cell-year row independently, doing 28Ã— the necessary spatial work.

3. **Per-element R-level overhead**: The `lapply` over 6.46 million rows with multiple character operations, subsetting, and `is.na` filtering per iteration incurs enormous interpreter overhead.

The `compute_neighbor_stats` function, by contrast, does only cheap integer-vector subsetting (`vals[idx]`) and simple numeric operations â€” it is fast once the lookup exists.

## Optimization Strategy

1. **Build the lookup using integer indexing via merge/join, not character key hashing.** Use `data.table` to create a fast equi-join between (neighbor_id, year) and (id, year) to resolve row indices in vectorized, C-level code.

2. **Exploit time-invariance of the spatial structure.** Build the spatial neighbor pairs once (344,208 cells Ã— their neighbors â‰ˆ 1.37M pairs), then cross-join with years in a single vectorized operation.

3. **Vectorize `compute_neighbor_stats`** using `data.table` grouped aggregation instead of row-wise `lapply`.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build neighbor lookup via vectorized data.table join
#    (replaces build_neighbor_lookup)
# ---------------------------------------------------------------
build_neighbor_lookup_fast <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a row index
  # Ensure row_idx exists
  data_dt[, row_idx := .I]

  # --- Step A: Build spatial neighbor edge list (time-invariant) ---
  # neighbors is an nb object: list of integer index vectors into id_order
  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nbrs <- neighbors[[i]]
    if (length(nbrs) == 0L || (length(nbrs) == 1L && nbrs[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nbrs])
  }))

  # --- Step B: Cross with years via join ---
  # Create keyed version of data for joining
  data_key <- data_dt[, .(id, year, row_idx)]
  setkey(data_key, id, year)

  # Get focal row indices
  focal_key <- copy(data_key)
  setnames(focal_key, c("id", "year", "row_idx"), c("focal_id", "year", "focal_row"))

  # Get neighbor row indices
  nbr_key <- copy(data_key)
  setnames(nbr_key, c("id", "year", "row_idx"), c("neighbor_id", "year", "nbr_row"))

  # Join: for each edge Ã— year, get focal_row and nbr_row
  setkey(edge_list, focal_id)
  setkey(focal_key, focal_id, year)

  # Expand edges across all years of the focal cell
  edges_with_focal <- merge(edge_list, focal_key, by = "focal_id", allow.cartesian = TRUE)

  # Now resolve neighbor rows
  setkey(edges_with_focal, neighbor_id, year)
  setkey(nbr_key, neighbor_id, year)

  edges_full <- merge(edges_with_focal, nbr_key, by = c("neighbor_id", "year"), nomatch = 0L)

  # Return the full edge table: focal_row -> nbr_row
  edges_full[, .(focal_row, nbr_row)]
}

# ---------------------------------------------------------------
# 2. Vectorized neighbor stats via data.table grouping
#    (replaces compute_neighbor_stats + do.call(rbind,...))
# ---------------------------------------------------------------
compute_neighbor_stats_fast <- function(data_dt, edges_dt, var_name) {
  # edges_dt has columns: focal_row, nbr_row
  # Pull neighbor values via integer indexing (vectorized)
  vals <- data_dt[[var_name]]
  work <- edges_dt[, nbr_val := vals[nbr_row]]

  # Remove NAs in neighbor values
  work <- work[!is.na(nbr_val)]

  # Grouped aggregation â€” all in C-level data.table code
  stats <- work[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = focal_row]

  # Build output columns aligned to all rows
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[stats$focal_row]  <- stats$nb_max
  out_min[stats$focal_row]  <- stats$nb_min
  out_mean[stats$focal_row] <- stats$nb_mean

  list(nb_max = out_max, nb_min = out_min, nb_mean = out_mean)
}

# ---------------------------------------------------------------
# 3. Main pipeline (replaces outer loop)
# ---------------------------------------------------------------
run_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table if needed (non-destructive copy)
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  message("Building vectorized neighbor edge table...")
  t0 <- Sys.time()
  edges_dt <- build_neighbor_lookup_fast(dt, id_order, rook_neighbors_unique)
  message("  Edge table built: ", nrow(edges_dt), " directed cell-year edges in ",
          round(difftime(Sys.time(), t0, units = "mins"), 1), " min")

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)
    t1 <- Sys.time()
    stats <- compute_neighbor_stats_fast(dt, copy(edges_dt), var_name)

    # Assign columns with same naming convention as original code
    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    set(dt, j = col_max,  value = stats$nb_max)
    set(dt, j = col_min,  value = stats$nb_min)
    set(dt, j = col_mean, value = stats$nb_mean)

    message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), " sec")
  }

  # Remove helper column and return as data.frame to preserve downstream compatibility
  dt[, row_idx := NULL]
  as.data.frame(dt)
}

# ---------------------------------------------------------------
# Usage (drop-in replacement):
# ---------------------------------------------------------------
# cell_data <- run_neighbor_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Then predict with the EXISTING trained model â€” no retraining:
# # predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M `lapply` iterations with `paste`/named-vector lookup | Eliminated |
| **Key construction** | Character `paste` + named-vector search (R-level, per row) | Integer equi-join via `data.table` merge (C-level, vectorized) |
| **Year redundancy** | Recomputes spatial structure 28Ã— | Builds spatial edges once, cross-joins with years |
| **Stats computation** | `lapply` over 6.46M rows + `do.call(rbind,...)` | `data.table` grouped `max/min/mean` (single pass, C-level) |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes on the same laptop |
| **Numerical output** | Identical `max`, `min`, `mean` per cell-year | Identical (preserved) |
| **Trained RF model** | Untouched | Untouched |

The colleague was looking at the wrong function. The real cost was the **6.46-million-iteration, character-key-based, year-redundant spatial lookup construction** in `build_neighbor_lookup()`, not the relatively lightweight `do.call(rbind, ...)` assembly in `compute_neighbor_stats()`.
END_RESPONSE>>>
