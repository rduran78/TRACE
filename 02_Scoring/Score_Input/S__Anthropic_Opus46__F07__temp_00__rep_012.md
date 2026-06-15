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
run_id: Anthropic_F07_T00_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) â€” for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries.
2. **`lapply` over 6.46M rows** â€” each iteration does string pasting, named-vector subsetting, and NA filtering. The per-element overhead of R's interpreted loop is enormous at this scale.
3. **Memory**: The resulting `neighbor_lookup` list of 6.46M integer vectors, plus the intermediate character vectors, can easily consume several GB.

Then `compute_neighbor_stats` does a second `lapply` over 6.46M elements â€” less expensive per iteration, but still slow in interpreted R.

**Root cause summary:**
- ~6.46M iterations of character-key construction and named-vector lookup â†’ O(nÂ²)-like behavior.
- Everything is done in interpreted R with no vectorization or hashing.
- The 86+ hour estimate is credible given the scale.

---

## Optimization Strategy

### 1. Replace character-key lookup with integer-key lookup via `data.table`

Use `data.table` to create a fast integer join between `(id, year)` and row indices. This replaces all `paste`/`setNames`/named-vector operations with O(1) hash-based lookups.

### 2. Vectorize the neighbor expansion

Instead of looping row-by-row, **expand all neighbor relationships into a single edge table** (a two-column data.frame of `(source_row, neighbor_row)`), then use grouped vectorized operations (`data.table` aggregation) to compute max, min, and mean in one pass per variable.

This turns the entire pipeline into:
- One `data.table` merge to map `(neighbor_cell_id, year)` â†’ row index.
- One grouped aggregation per variable.

### 3. Memory estimate

The directed rook-neighbor edge list has ~1.37M spatial edges Ã— 28 years â‰ˆ **~38.5M rows** (two integer columns â‰ˆ 308 MB). This fits comfortably in 16 GB.

### 4. Preserve the trained RF model and numerical estimand

The code only changes **how** neighbor features are computed, not **what** is computed. The max/min/mean values are identical, so the RF model's input features are unchanged. No retraining is needed.

---

## Working R Code

```r
library(data.table)

#' Build a vectorized edge table mapping each (cell, year) row to its
#' neighbor (cell, year) rows.  Replaces build_neighbor_lookup entirely.
#'
#' @param cell_data   data.frame/data.table with columns `id` and `year`
#'                    (and all predictor columns).
#' @param id_order    integer vector: the cell IDs in the order used by
#'                    the spdep::nb object (i.e., id_order[k] is the
#'                    cell-ID of the k-th element of rook_neighbors_unique).
#' @param neighbors   spdep::nb list (rook_neighbors_unique).  neighbors[[k]]
#'                    is an integer vector of indices into id_order.
#' @return A data.table with columns  src_row  and  nbr_row  (integer row
#'         indices into cell_data).

build_edge_table <- function(cell_data, id_order, neighbors) {

  ## ---- 1.  Build spatial edge list (cell-ID level) ----------------------
  n_cells <- length(id_order)
  # Pre-allocate: count total directed edges
  n_edges_spatial <- sum(lengths(neighbors))

  src_id <- integer(n_edges_spatial)
  nbr_id <- integer(n_edges_spatial)
  pos <- 1L
  for (k in seq_len(n_cells)) {
    nb <- neighbors[[k]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) next
    len <- length(nb)
    idx <- pos:(pos + len - 1L)
    src_id[idx] <- id_order[k]
    nbr_id[idx] <- id_order[nb]
    pos <- pos + len
  }
  # Trim if any nb objects had 0-neighbor entries
  if (pos - 1L < n_edges_spatial) {
    src_id <- src_id[seq_len(pos - 1L)]
    nbr_id <- nbr_id[seq_len(pos - 1L)]
  }
  spatial_edges <- data.table(src_id = src_id, nbr_id = nbr_id)

  ## ---- 2.  Map (id, year) â†’ row index -----------------------------------
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # We only need id, year, row_idx for the join
  key_dt <- dt[, .(id, year, row_idx)]

  ## ---- 3.  Cross-join spatial edges with years ---------------------------
  years <- sort(unique(dt$year))
  edge_year <- spatial_edges[, CJ(year = years), by = .(src_id, nbr_id)]
  #   columns: src_id, nbr_id, year

  ## ---- 4.  Join to get row indices for source and neighbor ---------------
  setnames(key_dt, c("id", "year", "row_idx"),
                    c("src_id", "year", "src_row"))
  edge_year <- key_dt[edge_year, on = .(src_id, year), nomatch = 0L]

  setnames(key_dt, c("src_id", "year", "src_row"),
                    c("nbr_id", "year", "nbr_row"))
  edge_year <- key_dt[edge_year, on = .(nbr_id, year), nomatch = 0L]

  # Clean up: keep only the mapping columns

  edge_year <- edge_year[, .(src_row, nbr_row)]
  setkey(edge_year, src_row)

  return(edge_year)
}


#' Compute neighbor max, min, mean for one variable using the edge table,
#' and add the three new columns to cell_data (in place if data.table).
#'
#' @param cell_dt     data.table version of cell_data (with row order preserved).
#' @param edge_table  data.table from build_edge_table (src_row, nbr_row).
#' @param var_name    character: name of the source variable.
#' @return  cell_dt, invisibly (modified in place).

compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_table) {

  # Pull neighbor values via the edge table
  vals <- cell_dt[[var_name]]
  et   <- copy(edge_table)
  et[, nbr_val := vals[nbr_row]]

  # Drop NAs in neighbor values before aggregation

  et <- et[!is.na(nbr_val)]

  # Grouped aggregation â€” one pass
  agg <- et[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), keyby = src_row]

  # Initialize new columns with NA
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  set(cell_dt, j = col_max,  value = NA_real_)
  set(cell_dt, j = col_min,  value = NA_real_)
  set(cell_dt, j = col_mean, value = NA_real_)

  # Fill in computed values at the correct rows
  set(cell_dt, i = agg$src_row, j = col_max,  value = agg$nb_max)
  set(cell_dt, i = agg$src_row, j = col_min,  value = agg$nb_min)
  set(cell_dt, i = agg$src_row, j = col_mean, value = agg$nb_mean)

  invisible(cell_dt)
}


## =========================================================================
##  MAIN PIPELINE  (drop-in replacement for the original outer loop)
## =========================================================================

# Convert once to data.table (preserves row order)
cell_dt <- as.data.table(cell_data)

# Build the edge table â€” runs once (~30 s â€“ 2 min on a laptop)
message("Building edge table â€¦")
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("  neighbor stats for: %s", var_name))
  compute_and_add_neighbor_features_fast(cell_dt, var_name, edge_table)
}

# Convert back to data.frame if downstream code expects one
cell_data <- as.data.frame(cell_dt)

message("Done â€” neighbor features added.
")
```

---

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` (6.46M list elements via character keys) | ~60â€“80 hours | **Replaced** by `build_edge_table`: ~1â€“2 min |
| `compute_neighbor_stats` (6.46M `lapply` iterations Ã— 5 vars) | ~6â€“10 hours | **Replaced** by `data.table` grouped aggregation: ~1â€“3 min per variable |
| **Total** | **86+ hours** | **~10â€“20 minutes** |

## Why the Estimand Is Preserved

The three statistics computed per `(cell, year)` row are:

```
max(neighbor_vals[!is.na(neighbor_vals)])
min(neighbor_vals[!is.na(neighbor_vals)])
mean(neighbor_vals[!is.na(neighbor_vals)])
```

The optimized code computes **exactly the same quantities** â€” the `data.table` aggregation after `et <- et[!is.na(nbr_val)]` is algebraically identical. Rows with zero non-NA neighbors receive `NA` in all three columns, matching the original `c(NA, NA, NA)` return. The trained Random Forest model receives identical input features and requires no retraining.
END_RESPONSE>>>
