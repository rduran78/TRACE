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
run_id: Anthropic_F06_T00_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row neighbor indices

- For each of the **~6.46 million rows**, the function does a character-key lookup (`paste` + named-vector indexing) to resolve neighbor row indices.
- `paste()` over 6.46M rows, repeated inside an `lapply` that itself runs 6.46M times, generates enormous numbers of temporary character strings.
- Named-vector lookup via `idx_lookup[neighbor_keys]` is hash-based but is called millions of times with small vectors, so the overhead per call dominates.
- The resulting object is a **list of 6.46 million integer vectors** â€” large in memory and slow to construct.

### Bottleneck B: `compute_neighbor_stats` â€” per-row `lapply` over 6.46M elements

- For each variable, an `lapply` iterates over 6.46M list elements, extracting values, removing NAs, and computing `max`, `min`, `mean`.
- This is repeated **5 times** (once per neighbor source variable), so ~32.3 million R-level function calls.
- Each call has R interpreter overhead that dwarfs the actual arithmetic.

### Why raster focal/kernel operations are not directly applicable

Raster focal operations (e.g., `terra::focal`) assume a regular rectangular grid with a fixed kernel. While the 344K cells likely form a grid, the panel structure (cell Ã— year) and the need for exact rook-neighbor relationships from a precomputed `spdep::nb` object mean that a focal approach would require reshaping into a 3D raster stack and verifying that the rook neighbors match exactly. Any mismatch would **change the numerical estimand**. The matrix/data.table approach below preserves the exact neighbor structure and is faster and safer.

---

## 2. Optimization Strategy

### Strategy: Sparse-matrix multiplication replaces both bottlenecks

1. **Build a sparse adjacency matrix `W`** (6.46M Ã— 6.46M) from the `nb` object and the panel structure â€” but stored as a sparse matrix with only ~1.37M Ã— 28 â‰ˆ 38.4M non-zero entries. This is done **once**.

2. **Compute neighbor stats via vectorized sparse operations:**
   - `neighbor_mean` = (W %*% x) / (W %*% 1_valid) â€” weighted by non-NA count.
   - `neighbor_max` and `neighbor_min` require a trick: iterate over the sparse structure column-wise, or use `data.table` grouped operations on the COO (coordinate) representation of W.

3. **Use `data.table` for the grouped max/min/mean** on the edge list (COO form of W), which is ~38.4M rows â€” trivially fast with `data.table`.

This reduces the runtime from **86+ hours to ~5â€“15 minutes** on a 16 GB laptop.

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with a row-order column
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]  # preserve original row order

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a cell-year edge list from the nb object (done ONCE)
#
#   For every directed rook-neighbor pair (i -> j) in the spatial nb
#   object, and for every year, create an edge:
#     (row of cell i in year t)  -->  (row of cell j in year t)
#
#   This replaces build_neighbor_lookup entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_list <- function(cell_dt, id_order, nb_obj) {
  # --- spatial edge list (directed, from nb object) ---
  from_ref <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_ref   <- unlist(nb_obj)

  spatial_edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # --- map (id, year) -> row_idx ---
  key_dt <- cell_dt[, .(id, year, row_idx)]
  setkey(key_dt, id, year)

  # --- cross-join spatial edges with all years ---
  years <- sort(unique(cell_dt$year))
  panel_edges <- CJ(edge_idx = seq_len(nrow(spatial_edges)), year = years)
  panel_edges[, `:=`(
    from_id = spatial_edges$from_id[edge_idx],
    to_id   = spatial_edges$to_id[edge_idx]
  )]
  panel_edges[, edge_idx := NULL]

  # --- resolve row indices for "from" (the focal cell) ---
  panel_edges <- merge(
    panel_edges,
    key_dt,
    by.x = c("from_id", "year"),
    by.y = c("id", "year"),
    all.x = TRUE
  )
  setnames(panel_edges, "row_idx", "from_row")

  # --- resolve row indices for "to" (the neighbor cell) ---
  panel_edges <- merge(
    panel_edges,
    key_dt,
    by.x = c("to_id", "year"),
    by.y = c("id", "year"),
    all.x = TRUE
  )
  setnames(panel_edges, "row_idx", "to_row")

  # --- drop any edges where either cell-year is missing ---
  panel_edges <- panel_edges[!is.na(from_row) & !is.na(to_row)]

  panel_edges[, .(from_row, to_row)]
}

cat("Building panel edge list...\n")
edges <- build_edge_list(cell_dt, id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s directed cell-year edges\n",
            formatC(nrow(edges), format = "d", big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor stats for each variable via data.table
#         grouped aggregation on the edge list.
#
#   For each focal row (from_row), we look up the neighbor values
#   (to_row) and compute max, min, mean â€” exactly matching the
#   original compute_neighbor_stats logic.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_and_add_all_neighbor_features <- function(cell_dt, edges, var_names) {
  for (var_name in var_names) {
    cat(sprintf("  Processing neighbor stats for: %s\n", var_name))

    # Attach the neighbor's value to each edge
    edges[, nbr_val := cell_dt[[var_name]][to_row]]

    # Grouped aggregation: max, min, mean of non-NA neighbor values
    stats <- edges[!is.na(nbr_val),
                   .(nb_max  = max(nbr_val),
                     nb_min  = min(nbr_val),
                     nb_mean = mean(nbr_val)),
                   by = from_row]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]

    # Fill in computed values
    cell_dt[stats$from_row, (max_col)  := stats$nb_max]
    cell_dt[stats$from_row, (min_col)  := stats$nb_min]
    cell_dt[stats$from_row, (mean_col) := stats$nb_mean]

    # Clean up temporary column
    edges[, nbr_val := NULL]
  }

  cell_dt
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
cell_dt <- compute_and_add_all_neighbor_features(cell_dt, edges, neighbor_source_vars)
cat("Done.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Restore original row order and convert back if needed
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setorder(cell_dt, row_idx)
cell_dt[, row_idx := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element R list built via `lapply` + `paste` + named-vector lookup | Single `data.table` merge builds a flat edge list (~38M rows) |
| **Stats computation** | `lapply` over 6.46M elements Ã— 5 vars = 32.3M R function calls | 5 Ã— one `data.table` grouped aggregation over ~38M rows |
| **Memory** | 6.46M-element list of integer vectors (~1â€“2 GB) + overhead | One 2-column integer edge table (~300 MB) reused for all vars |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** on a 16 GB laptop |
| **Numerical result** | max/min/mean of non-NA rook-neighbor values | **Identical** â€” same directed edges, same aggregation functions, same NA handling |
| **Trained RF model** | â€” | **Untouched** â€” used only for prediction after features are computed |

The key insight is that the neighbor structure is **static across variables**: build the edge list once, then reuse it. `data.table`'s grouped aggregation (`by = from_row`) replaces millions of R-level `lapply` iterations with a single optimized C-level pass per variable.
END_RESPONSE>>>
