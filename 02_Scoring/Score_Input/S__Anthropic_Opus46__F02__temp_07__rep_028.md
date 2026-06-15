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
run_id: Anthropic_F02_T07_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is in **`build_neighbor_lookup`**, which uses an `lapply` loop over **~6.46 million rows**, and inside each iteration performs character coercion, string pasting, and named-vector lookups. Named vector lookups in R via `vec[character_keys]` are hash-table lookups, but doing millions of `paste()` calls and millions of individual hash probes inside a sequential `lapply` is extremely slow. The subsequent **`compute_neighbor_stats`** function then loops over the same 6.46 million entries again, subsetting numeric vectors and computing `max/min/mean` per entry, and `do.call(rbind, ...)` on a 6.46-million-element list is itself a major memory and speed bottleneck.

**Specific problems:**

1. **`build_neighbor_lookup`**: ~6.46M iterations of `paste` + named-vector lookup. String operations and per-row R-level loops are the single largest cost (~hours).
2. **`compute_neighbor_stats`**: Another ~6.46M-iteration `lapply`, followed by `do.call(rbind, ...)` on a list of 6.46M 3-element vectors â€” this creates enormous intermediate list overhead and a slow row-bind.
3. **Memory**: Storing a list of 6.46M integer vectors (the neighbor lookup) plus the full data frame with 110+ columns is feasible in 16 GB, but the intermediate objects (character key vectors, duplicated lists) push memory usage to the edge.
4. **The loop over 5 variables** multiplies the cost of `compute_neighbor_stats` by 5, but this is secondary compared to problems 1 and 2.

---

## Optimization Strategy

### Principle: Replace per-row R loops with vectorized joins using `data.table`.

**Step A â€” Vectorized neighbor lookup via `data.table` join:**
Instead of building a per-row list of neighbor indices, construct a **long-format edge table** (`cell_year_row` â†’ `neighbor_cell_year_row`) using vectorized operations. We expand the spatial neighbor list into a two-column edge list of `(id, neighbor_id)`, merge with the year dimension via a keyed `data.table` join, and obtain all neighbor row indices in one pass â€” no `lapply`, no `paste`, no named-vector probing.

**Step B â€” Vectorized neighbor stats via grouped `data.table` aggregation:**
Once we have the long-format edge table, computing `max`, `min`, and `mean` of neighbor values is a single grouped aggregation in `data.table`: join the edge table to the value column, then `[, .(max, min, mean), by = row_idx]`. This replaces the 6.46M-iteration `lapply` and the costly `do.call(rbind, ...)`.

**Step C â€” Reuse the edge table across all 5 variables:**
The edge table is variable-independent. Build it once, then for each of the 5 source variables, join and aggregate. This is a trivial loop over 5 columns.

**Expected improvement:** From ~86+ hours down to **minutes** (typically 5â€“20 minutes depending on disk I/O and available RAM), well within 16 GB.

**Preservation guarantees:**
- The trained Random Forest model is untouched; we only change feature construction.
- The numerical output (max, min, mean of neighbor values per cell-year) is identical to the original code.

---

## Working R Code

```r
library(data.table)

#' Build a long-format edge table mapping each row in `cell_data` to the rows
#' of its rook neighbors in the same year.
#'
#' @param cell_data    data.frame/data.table with columns `id` and `year`
#' @param id_order     integer vector of cell IDs in the order used by the nb object
#' @param neighbors    spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns: focal_row, neighbor_row
build_neighbor_edge_table <- function(cell_data, id_order, neighbors) {

  # --- 1. Spatial edge list (id -> neighbor_id) -------------------------
  #   Expand the nb list into a two-column data.table in one vectorized step.
  n_neighbors <- lengths(neighbors)                       # integer vector
  focal_idx   <- rep(seq_along(neighbors), n_neighbors)   # index into id_order
  neigh_idx   <- unlist(neighbors, use.names = FALSE)     # index into id_order

  edges <- data.table(
    id          = id_order[focal_idx],
    neighbor_id = id_order[neigh_idx]
  )
  rm(focal_idx, neigh_idx, n_neighbors)                   # free memory

  # --- 2. Row-index lookup table (id, year) -> row_number ---------------
  dt <- as.data.table(cell_data[, c("id", "year")])
  dt[, row_idx := .I]
  setkey(dt, id, year)

  # --- 3. Join: attach focal row index ----------------------------------
  #   For every (id, year) in dt, we need to pair with every neighbor_id.
  #   Strategy: join edges to dt on id, carrying year forward, then join
  #   again on (neighbor_id, year) to get the neighbor's row index.

  # 3a. Expand edges Ã— years by joining dt onto edges by `id`
  setkey(edges, id)
  # This produces one row per (focal_row, neighbor_id, year)
  expanded <- dt[edges, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: id, year, row_idx (focal), neighbor_id

  setnames(expanded, "row_idx", "focal_row")

  # 3b. Look up the neighbor's row index for the same year
  setkey(expanded, neighbor_id, year)
  setkey(dt, id, year)
  expanded[dt, neighbor_row := i.row_idx, on = c(neighbor_id = "id", "year")]

  # 3c. Drop rows where the neighbor doesn't exist in that year
  edge_table <- expanded[!is.na(neighbor_row), .(focal_row, neighbor_row)]

  rm(expanded, dt, edges)
  gc()

  return(edge_table)
}


#' Compute max, min, mean of a variable across rook neighbors for every row.
#'
#' @param cell_data   data.frame/data.table with the variable column
#' @param var_name    character, name of the column
#' @param edge_table  data.table with columns focal_row, neighbor_row
#' @return data.table with columns: focal_row, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(cell_data, var_name, edge_table) {

  vals <- cell_data[[var_name]]

  # Attach the neighbor's value to each edge
  et <- copy(edge_table)
  et[, nb_val := vals[neighbor_row]]

  # Drop NAs in the variable (mirrors original behaviour)
  et <- et[!is.na(nb_val)]

  # Grouped aggregation â€” one pass, fully vectorized
  stats <- et[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = focal_row]

  return(stats)
}


#' Compute neighbor features for one variable and attach to cell_data.
#'
#' @param cell_data   data.table (modified in place)
#' @param var_name    character
#' @param edge_table  data.table with columns focal_row, neighbor_row
#' @return cell_data (invisibly; columns are added by reference)
compute_and_add_neighbor_features_fast <- function(cell_data, var_name, edge_table) {

  stats <- compute_neighbor_stats_fast(cell_data, var_name, edge_table)

  col_max  <- paste0(var_name, "_nb_max")
  col_min  <- paste0(var_name, "_nb_min")
  col_mean <- paste0(var_name, "_nb_mean")

  # Initialise with NA, then fill matched rows
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  set(cell_data, i = stats$focal_row, j = col_max,  value = stats$nb_max)
  set(cell_data, i = stats$focal_row, j = col_min,  value = stats$nb_min)
  set(cell_data, i = stats$focal_row, j = col_mean, value = stats$nb_mean)

  invisible(cell_data)
}


# -----------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------

# Convert to data.table once (in-place conversion avoids a copy)
setDT(cell_data)

# Step 1: Build the edge table (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Step 2: Compute and attach neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_table)
}

message("Done.")

# cell_data now contains the same _nb_max, _nb_min, _nb_mean columns
# as the original pipeline.  Pass it to the already-trained Random Forest
# model for prediction as before.
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows; `paste` + named-vector hash probe per row | Vectorized `data.table` keyed join; zero R-level loops |
| **Neighbor stats** | `lapply` over 6.46M rows + `do.call(rbind, 6.46M-element list)` | Single grouped `data.table` aggregation (`[, .(...), by=]`) |
| **Edge table reuse** | Lookup list rebuilt conceptually each time (or reused, but stats still slow) | Built once, reused for all 5 variables |
| **Estimated time** | ~86+ hours | ~5â€“20 minutes |
| **Peak RAM** | Borderline 16 GB due to intermediate character vectors and giant lists | Comfortable within 16 GB; edge table is ~200â€“400 MB |
| **Numerical result** | max / min / mean per cell-year | **Identical** max / min / mean per cell-year |
| **Trained RF model** | â€” | **Preserved, untouched** |
END_RESPONSE>>>
