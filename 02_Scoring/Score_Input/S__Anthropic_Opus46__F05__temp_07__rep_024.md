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
run_id: Anthropic_F05_T07_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: Per-Row String Operations at Scale

Inside `build_neighbor_lookup`, for each of the ~6.46 million rows, the code:

1. **Creates `idx_lookup`** â€” this line `setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))` is defined *outside* the `lapply` but still constructs a named character vector of 6.46M string keys once. That part is O(N) and acceptable in isolation.

2. **Per-row `paste()` + named-vector lookup** â€” for each row `i`, it builds `neighbor_keys` via `paste()` and then does character-based lookup into `idx_lookup`. With ~4 neighbors per cell on average, this is ~6.46M Ã— 4 = ~25.8M `paste()` calls and named-vector hash lookups. Named-vector lookup in R is hash-based but has significant per-call overhead due to string hashing and R's general interpreted overhead.

### The Broader Algorithmic Issue

The entire `build_neighbor_lookup` function is architecturally suboptimal because it solves a **join problem** with **row-by-row string manipulation**. The neighbor lookup is fundamentally: *"for each (cell, year), find the row indices of (neighbor_cell, same_year)"*. This is a **merge/join** operation that can be vectorized completely.

**Estimated cost of current approach:**
- `build_neighbor_lookup`: ~6.46M iterations of `lapply`, each doing `paste()` + character indexing â†’ **hours**
- `compute_neighbor_stats`: runs 5 times over the lookup, each iterating 6.46M entries â†’ significant but secondary
- Total: **86+ hours** as reported

### The Fix: Vectorized Join via `data.table`

We can replace the entire per-row string-key construction with:
1. An **integer-keyed join** â€” map `(id, year)` â†’ row index using `data.table` keyed joins.
2. **Expand the neighbor list once** into a flat edge table `(source_row, neighbor_id)`, then join to get `(source_row, neighbor_row)`.
3. **Compute all neighbor stats vectorially** using `data.table` grouped aggregation.

This eliminates all per-row `paste()` calls and replaces the `lapply` with vectorized operations.

---

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Map (id,year)â†’row | 6.46M-entry named char vector | `data.table` integer-keyed join |
| Expand neighbors per row | `lapply` over 6.46M rows, `paste` per neighbor | Vectorized expansion of `nb` object into flat `data.table` |
| Look up neighbor rows | Character hash per neighbor key | Keyed `data.table` equi-join on `(id, year)` integer columns |
| Compute stats | `lapply` over 6.46M rows per variable | `data.table` grouped `max/min/mean` per variable |

**Expected runtime:** Minutes instead of days. The join is O(E) where E â‰ˆ total directed neighbor-year pairs (~1.37M neighbors Ã— 28 years â‰ˆ 38.5M edges), and grouped aggregation is similarly O(E).

---

## Working R Code

```r
library(data.table)

#' Build a flat edge table: for every row in cell_data, list the row indices
#' of its rook neighbors in the same year.
#'
#' @param cell_data       data.frame/data.table with columns `id` and `year`
#' @param id_order        integer vector of cell IDs in the order matching the nb object
#' @param rook_neighbors  spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns: src_row, nbr_row
build_neighbor_edge_table <- function(cell_data, id_order, rook_neighbors) {

  dt <- as.data.table(cell_data)
  dt[, src_row := .I]

  # --- Step 1: Expand the spatial nb object into a flat (src_id, nbr_id) table ---
  # Each element of rook_neighbors is an integer vector of indices into id_order.
  n_cells <- length(id_order)
  stopifnot(length(rook_neighbors) == n_cells)

  # Pre-compute lengths for allocation
  lens <- lengths(rook_neighbors)
  total_edges <- sum(lens)

  src_id_vec <- rep(id_order, times = lens)
  nbr_id_vec <- id_order[unlist(rook_neighbors, use.names = FALSE)]

  edges_spatial <- data.table(
    src_id = src_id_vec,
    nbr_id = nbr_id_vec
  )

  # --- Step 2: Join to get (src_row, nbr_row) for every (cell, year) ---
  # Build a lookup from (id, year) -> row index
  row_lookup <- dt[, .(id, year, src_row)]
  setkey(row_lookup, id, year)

  # For every row in dt, get its spatial neighbors via join on src_id
  # First, create (src_row, src_id, year, nbr_id) by joining dt rows to spatial edges
  src_info <- dt[, .(src_row, src_id = id, year)]
  setkey(edges_spatial, src_id)
  setkey(src_info, src_id)

  # This is a many-to-many join: each row has multiple neighbors,
  # and each src_id appears in multiple years.
  # Use edges_spatial as the left table keyed on src_id, join with src_info.
  # More efficient: join src_info to edges_spatial on src_id
  expanded <- edges_spatial[src_info,
    .(src_row, nbr_id, year),
    on = .(src_id),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Now look up the row index of each (nbr_id, year)
  # Create a keyed lookup: (id, year) -> row index
  nbr_lookup <- dt[, .(nbr_row = src_row, id, year)]
  setkey(nbr_lookup, id, year)

  # Join to resolve nbr_id + year -> nbr_row
  expanded[, id := nbr_id]
  result <- nbr_lookup[expanded, .(src_row, nbr_row), on = .(id, year), nomatch = NA]

  # Drop edges where the neighbor cell-year doesn't exist in the data

  result <- result[!is.na(nbr_row)]

  return(result)
}


#' Compute neighbor max, min, mean for a variable using the edge table.
#'
#' @param cell_data  data.frame/data.table with the source variable
#' @param var_name   character: column name to aggregate
#' @param edge_dt    data.table with columns src_row, nbr_row
#' @return data.table with nrow(cell_data) rows and columns:
#'         nb_max_{var}, nb_min_{var}, nb_mean_{var}
compute_neighbor_stats_fast <- function(cell_data, var_name, edge_dt) {

  dt <- as.data.table(cell_data)
  n <- nrow(dt)

  # Attach the neighbor's value to each edge
  vals <- dt[[var_name]]
  edge_work <- edge_dt[, .(src_row, nbr_val = vals[nbr_row])]

  # Drop NAs in the variable
  edge_work <- edge_work[!is.na(nbr_val)]

  # Grouped aggregation
  agg <- edge_work[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), keyby = src_row]

  # Allocate full-length result with NAs for rows with no valid neighbors
  out <- data.table(
    src_row = seq_len(n),
    nb_max  = NA_real_,
    nb_min  = NA_real_,
    nb_mean = NA_real_
  )
  out[agg, on = .(src_row), `:=`(
    nb_max  = i.nb_max,
    nb_min  = i.nb_min,
    nb_mean = i.nb_mean
  )]

  # Rename columns to match original naming convention
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(out, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  out[, src_row := NULL]
  return(out)
}


#' Main driver: replaces the entire outer loop.
#'
#' @param cell_data              data.frame with columns id, year, and all source vars
#' @param id_order               integer vector of cell IDs matching the nb object order
#' @param rook_neighbors_unique  spdep nb object
#' @param neighbor_source_vars   character vector of variable names
#' @return cell_data with neighbor feature columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                "def", "usd_est_n2")) {

  message("Building neighbor edge table...")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge table: %s edges built in %.1f seconds.",
                  format(nrow(edge_dt), big.mark = ","),
                  (proc.time() - t0)[3]))

  cell_data <- as.data.table(cell_data)

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s'...", var_name))
    t1 <- proc.time()
    stats_dt <- compute_neighbor_stats_fast(cell_data, var_name, edge_dt)
    cell_data <- cbind(cell_data, stats_dt)
    message(sprintf("  Done in %.1f seconds.", (proc.time() - t1)[3]))
  }

  return(cell_data)
}


# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================
# cell_data <- add_all_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # The trained Random Forest model is untouched â€” just use predict() as before:
# # preds <- predict(rf_model, newdata = cell_data)
```

---

## Complexity and Memory Analysis

| Metric | Original | Optimized |
|--------|----------|-----------|
| **Neighbor expansion** | O(N) `lapply`, each with `paste` + char hash | O(E) vectorized `data.table` join |
| **Stats computation** | O(N) `lapply` Ã— 5 vars | O(E) grouped aggregation Ã— 5 vars |
| **Total string ops** | ~25.8M `paste()` + 25.8M char lookups | **Zero** |
| **Edge table memory** | N/A (implicit) | ~38.5M rows Ã— 2 int cols â‰ˆ **0.6 GB** |
| **Expected runtime** | 86+ hours | **~5â€“15 minutes** |

### Why This Preserves Correctness

- **Same estimand**: For each `(cell, year)` row, we compute `max`, `min`, `mean` over the exact same set of neighbor values â€” those neighbors identified by `rook_neighbors_unique` that exist in the same year. The `nomatch = NA` + `!is.na(nbr_row)` filtering exactly replicates the original `result[!is.na(result)]` logic.
- **Same column names**: Output columns follow the same `nb_max_{var}`, `nb_min_{var}`, `nb_mean_{var}` naming pattern (adjust the prefix if your `compute_and_add_neighbor_features` used a different convention).
- **Trained RF model untouched**: No model retraining is needed â€” the feature columns are numerically identical, so `predict(rf_model, newdata = cell_data)` produces the same predictions.

### Key Insight

The fundamental algorithmic shift is from **row-iteration with string-key lookups** (O(N) interpreted R loop with per-element string overhead) to a **single vectorized relational join** (O(E) in compiled C code inside `data.table`). The string-key pattern was indeed a symptom of solving a join problem imperatively rather than declaratively.
END_RESPONSE>>>
