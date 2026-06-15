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
run_id: Anthropic_F12_T10_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on a list of 6.46 million small 3-element vectors is not free, it is a single vectorized C-level operation that completes in seconds. There is no "repeated list binding" inside `compute_neighbor_stats()` â€” it uses `lapply` to build a list in one pass, then binds once. This is already the standard efficient R idiom.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive per-row string operations**: For every one of ~6.46 million rows, it calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then performs named-vector lookups via `idx_lookup[neighbor_keys]`. Constructing and hashing ~6.46M Ã— ~4 neighbors â‰ˆ 25+ million string keys inside an `lapply` is extremely expensive.

2. **Repeated `as.character()` coercion**: `as.character(data$id[i])` is called 6.46 million times inside the loop, one element at a time.

3. **Named vector lookup at scale**: `idx_lookup` is a named vector with ~6.46 million entries. Named vector lookup in R is O(n) partial matching by default, and even with hashing, doing millions of individual lookups into a vector of that size is slow. An environment or `data.table` join would be far faster.

4. **Redundant computation across years**: The neighbor *structure* is purely spatial â€” cell A's neighbors are the same cells regardless of year. Yet the function recomputes neighbor indices for every cell-year row, doing 28Ã— more work than necessary. With 344,208 cells and 28 years, it computes 6.46M lookups when only 344,208 unique spatial lookups are needed.

**In summary**: `build_neighbor_lookup()` is doing ~6.46 million iterations of string construction, string-based named-vector lookups, and NA filtering. This, not the `rbind`, is what drives the 86+ hour runtime.

---

## Optimization Strategy

1. **Compute spatial neighbor index mapping only once per cell (344K), not per cell-year (6.46M).** Since the rook neighborhood is time-invariant, build a cell-to-cell mapping, then expand to rows using fast integer indexing per year.

2. **Replace all string-key named-vector lookups with `data.table` hash joins or environment-based lookups.** These are O(1) amortized.

3. **Vectorize `compute_neighbor_stats()`** by operating on the full numeric vector with group-level operations or, at minimum, replacing `lapply` + `do.call(rbind, ...)` with direct matrix pre-allocation.

4. **Preserve the trained Random Forest model** â€” we only change feature-engineering code, producing the same numerical columns.

5. **Preserve the original numerical estimand** â€” max, min, mean of each neighbor variable are computed identically.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Optimized build_neighbor_lookup
#
# Key insight: neighbor relationships are spatial, not temporal.
# Build a cell-level mapping once (344K cells), then expand to
# row-level by year using fast integer indexing.
# ==============================================================

build_neighbor_lookup_fast <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id' and 'year' (and others)
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep::nb object (list of integer neighbor indices)

  # --- Spatial mapping (done once, 344K cells) ---
  # Map each cell ID to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Get unique cell IDs present in the data
  unique_ids <- unique(data_dt$id)

  # For each unique cell, find the row indices of its neighbors

  # per year using data.table keyed joins (O(1) hash lookups).

  # Key the data by id and year for fast lookups
  setkey(data_dt, id, year)

  # Add a row-index column
  data_dt[, .row_idx := .I]

  # Build a (cell -> neighbor_cells) list at the spatial level
  # This is only 344K entries, not 6.46M
  cell_to_neighbor_cells <- lapply(as.character(unique_ids), function(cid) {
    ref <- id_to_ref[cid]
    if (is.na(ref)) return(integer(0))
    nb_indices <- neighbors[[ref]]
    if (length(nb_indices) == 0) return(integer(0))
    id_order[nb_indices]
  })
  names(cell_to_neighbor_cells) <- as.character(unique_ids)

  # --- Expand to row-level using vectorized data.table join ---
  # For each row i, its neighbor rows are the rows with
  # id âˆˆ cell_to_neighbor_cells[[data$id[i]]] AND year == data$year[i]

  # Build an edge table: (focal_id, neighbor_id)
  edge_list <- rbindlist(lapply(seq_along(unique_ids), function(k) {
    cid <- unique_ids[k]
    nb_cells <- cell_to_neighbor_cells[[as.character(cid)]]
    if (length(nb_cells) == 0) return(NULL)
    data.table(focal_id = cid, neighbor_id = nb_cells)
  }))

  if (nrow(edge_list) == 0) {
    # No neighbors at all â€” return empty lookup
    return(vector("list", nrow(data_dt)))
  }

  # Cross with years: for each (focal_id, neighbor_id) pair,
  # we need all years present for the focal cell.
  # But since the panel is balanced (all cells Ã— all years),
  # every year applies to every pair.
  years <- sort(unique(data_dt$year))

  # Build the full (focal_id, year, neighbor_id) table via cross join
  # This is ~1.37M pairs Ã— 28 years â‰ˆ 38.5M rows, but we can
  # do this as a keyed join instead of materializing everything.

  # More memory-efficient approach: join edge_list against data_dt
  # to resolve neighbor_id + year -> row_idx, then group by focal row.

  # First, get (focal_id, year, focal_row_idx)
  focal_rows <- data_dt[, .(focal_id = id, year, focal_row_idx = .row_idx)]

  # Join focal rows with edge list to get (focal_row_idx, year, neighbor_id)
  setkey(edge_list, focal_id)
  setkey(focal_rows, focal_id)

  # This join expands each focal row by its number of neighbors
  expanded <- edge_list[focal_rows, on = "focal_id", allow.cartesian = TRUE,
                        nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, year, focal_row_idx

  # Now resolve neighbor_id + year -> neighbor_row_idx
  neighbor_index <- data_dt[, .(neighbor_id = id, year, neighbor_row_idx = .row_idx)]
  setkey(neighbor_index, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  matched <- neighbor_index[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Keep only matched rows
  matched <- matched[!is.na(neighbor_row_idx)]

  # Group by focal_row_idx to build the lookup list
  setkey(matched, focal_row_idx)
  lookup_dt <- matched[, .(neighbor_rows = list(neighbor_row_idx)),
                       by = focal_row_idx]

  # Initialize full lookup with empty integer vectors
  n <- nrow(data_dt)
  neighbor_lookup <- vector("list", n)
  for (j in seq_len(n)) neighbor_lookup[[j]] <- integer(0)

  # Fill in from the grouped result
  for (k in seq_len(nrow(lookup_dt))) {
    neighbor_lookup[[lookup_dt$focal_row_idx[k]]] <- lookup_dt$neighbor_rows[[k]]
  }

  # Clean up temporary column
  data_dt[, .row_idx := NULL]

  neighbor_lookup
}


# ==============================================================
# STEP 2: Optimized compute_neighbor_stats
#
# Replace lapply + do.call(rbind) with fully vectorized
# data.table grouped aggregation using the edge information.
# ==============================================================

compute_neighbor_stats_fast <- function(data_dt, edge_matched, var_name) {
  # edge_matched: data.table with focal_row_idx, neighbor_row_idx
  # Compute stats in a fully vectorized manner

  n <- nrow(data_dt)

  # Get neighbor values via integer indexing (vectorized)
  work <- edge_matched[, .(focal_row_idx, val = data_dt[[var_name]][neighbor_row_idx])]
  work <- work[!is.na(val)]

  # Grouped aggregation â€” single pass, fully vectorized
  stats <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row_idx]

  # Build output matrix (NA for rows without neighbors)
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(out) <- paste0("neighbor_", c("max_", "min_", "mean_"), var_name)
  out[stats$focal_row_idx, 1] <- stats$nb_max
  out[stats$focal_row_idx, 2] <- stats$nb_min
  out[stats$focal_row_idx, 3] <- stats$nb_mean

  out
}


# ==============================================================
# STEP 3: Full optimized pipeline
# ==============================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  data_dt <- as.data.table(cell_data)
  data_dt[, .row_idx := .I]

  # ---- Build spatial edge list once (344K cells) ----
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  unique_ids <- unique(data_dt$id)

  edge_list <- rbindlist(lapply(unique_ids, function(cid) {
    ref <- id_to_ref[as.character(cid)]
    if (is.na(ref)) return(NULL)
    nb_idx <- rook_neighbors_unique[[ref]]
    if (length(nb_idx) == 0) return(NULL)
    data.table(focal_id = cid, neighbor_id = id_order[nb_idx])
  }))

  if (is.null(edge_list) || nrow(edge_list) == 0) {
    warning("No neighbor relationships found.")
    return(cell_data)
  }

  # ---- Resolve to row indices via keyed joins ----
  focal_index <- data_dt[, .(focal_id = id, year, focal_row_idx = .row_idx)]
  neighbor_index <- data_dt[, .(neighbor_id = id, year, neighbor_row_idx = .row_idx)]

  setkey(edge_list, focal_id)
  setkey(focal_index, focal_id)
  expanded <- edge_list[focal_index, on = "focal_id",
                        allow.cartesian = TRUE, nomatch = NULL]

  setkey(expanded, neighbor_id, year)
  setkey(neighbor_index, neighbor_id, year)
  edge_matched <- neighbor_index[expanded, on = c("neighbor_id", "year"),
                                  nomatch = NA]
  edge_matched <- edge_matched[!is.na(neighbor_row_idx)]

  # Keep only what we need
  edge_matched <- edge_matched[, .(focal_row_idx, neighbor_row_idx)]
  setkey(edge_matched, focal_row_idx)

  # ---- Compute neighbor features for each variable ----
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "\n")
    stats_mat <- compute_neighbor_stats_fast(data_dt, edge_matched, var_name)

    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    data_dt[, (col_max)  := stats_mat[, 1]]
    data_dt[, (col_min)  := stats_mat[, 2]]
    data_dt[, (col_mean) := stats_mat[, 3]]
  }

  data_dt[, .row_idx := NULL]

  # Return as data.frame to preserve downstream compatibility
  # (including the trained Random Forest model's expectations)
  as.data.frame(data_dt)
}

# ==============================================================
# USAGE (drop-in replacement for the original pipeline)
# ==============================================================
# cell_data <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# The trained Random Forest model is untouched â€” only the
# feature-engineering stage is optimized. The output columns
# (neighbor_max_*, neighbor_min_*, neighbor_mean_*) contain
# identical numerical values to the original implementation.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup()`: 6.46M iterations of string paste + named-vector lookup | Eliminated |
| **Spatial redundancy** | Recomputes identical neighbor structure 28Ã— (once per year) | Computes once for 344K cells, expands via join |
| **Key lookup mechanism** | Named character vector (slow at scale) | `data.table` keyed hash join (O(1) amortized) |
| **Stat computation** | `lapply` over 6.46M rows â†’ `do.call(rbind, ...)` | Fully vectorized `data.table` grouped aggregation |
| **Estimated runtime** | 86+ hours | Minutes (estimated 5â€“15 min on the same laptop) |
| **Trained RF model** | âœ… Preserved (untouched) | âœ… Preserved |
| **Numerical output** | max/min/mean per neighbor set | Identical max/min/mean per neighbor set |

**Verdict**: The colleague misidentified the symptom (`do.call(rbind, ...)`) as the disease. The actual bottleneck is the per-row string construction and named-vector lookup in `build_neighbor_lookup()`, compounded by the failure to exploit the time-invariant spatial structure. The fix is to compute the spatial neighbor mapping once and resolve row indices via `data.table` hash joins.
END_RESPONSE>>>
