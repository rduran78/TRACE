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
run_id: Anthropic_F04_T07_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs character paste keys (`"id_year"`) for every neighbor.
4. Looks up those keys in a named character vector (`idx_lookup`), which is an **O(n) named-vector search repeated millions of times**.

**Key problems:**

- **Named vector lookup is O(n) per probe in base R** when the vector is large (6.46M entries). This is called millions of times Ã— multiple neighbors per call.
- **Character string pasting and matching** inside a hot loop over 6.46M rows is extremely expensive.
- **`compute_neighbor_stats` uses a base-R `lapply` + `do.call(rbind, ...)`** over 6.46M list elements, which is slow due to repeated memory allocation.
- The entire design is **row-wise and scalar**; it never exploits vectorization or fast hash-based joins.

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins** â€” O(1) amortized lookups.
2. **Pre-build an edge list (long table) of `(row_index, neighbor_row_index)` once** using vectorized `data.table` joins, eliminating the per-row `lapply` entirely.
3. **Compute neighbor stats vectorially** using `data.table` grouped aggregation on the edge list â€” one pass per variable, fully vectorized.
4. **Preserve exact numerical output**: still computes max, min, mean of non-NA neighbor values, yielding identical columns.

Expected speedup: from ~86+ hours to **minutes** (typically 5â€“15 min on a 16 GB laptop).

## Optimized R Code

```r
library(data.table)

#' Build a vectorized edge list mapping each cell-year row to its neighbor rows.
#' Returns a data.table with columns: row_i, neighbor_row_i
build_neighbor_edgelist <- function(data, id_order, neighbors) {
  # data must have columns: id, year (and be ordered by original row number)
  dt <- as.data.table(data[, c("id", "year")])
  dt[, row_i := .I]

  # --- Step 1: Build cell-level edge list from the nb object ---
  # id_order[k] is the cell id for the k-th entry in the nb object
  # neighbors[[k]] gives integer indices into id_order for k's neighbors
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove zero-neighbor entries (spdep uses integer(0) for islands)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  cell_edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx, valid)

  # --- Step 2: Cross join cell edges with years to get row-level edges ---
  years <- sort(unique(dt$year))

  # Create a keyed lookup: (id, year) -> row_i
  setkey(dt, id, year)

  # Expand cell edges by year
  # Use CJ-like expansion but more memory-friendly: replicate edges for each year
  cell_edges_expanded <- cell_edges[, .(from_id, to_id, year = rep(list(years), .N))]
  # More efficient: use cross join
  cell_edges_expanded <- CJ_dt(cell_edges, years)

  # Actually, the cleanest vectorized approach:
  n_edges <- nrow(cell_edges)
  n_years <- length(years)

  edge_year <- data.table(
    from_id = rep(cell_edges$from_id, each = n_years),
    to_id   = rep(cell_edges$to_id,   each = n_years),
    year    = rep(years, times = n_edges)
  )
  rm(cell_edges)

  # --- Step 3: Join to get row indices for both from and to ---
  # Join for 'from' side
  edge_year[dt, on = .(from_id = id, year = year), row_i := i.row_i]
  # Join for 'to' (neighbor) side
  edge_year[dt, on = .(to_id = id, year = year), neighbor_row_i := i.row_i]

  # Drop edges where either side has no matching row (shouldn't happen, but safe)
  edge_year <- edge_year[!is.na(row_i) & !is.na(neighbor_row_i),
                         .(row_i, neighbor_row_i)]

  return(edge_year)
}

# Helper: if CJ_dt was referenced above, we don't actually need it;
# the explicit rep() approach is used instead.

#' Compute neighbor max, min, mean for one variable using the edge list.
#' Returns a data.table with columns: row_i, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(data_dt, edgelist, var_name) {
  vals <- data_dt[[var_name]]

  # Attach neighbor values to edge list
  el <- edgelist[, .(row_i, neighbor_row_i)]
  el[, nb_val := vals[neighbor_row_i]]

  # Remove NA neighbor values
  el <- el[!is.na(nb_val)]

  # Grouped aggregation â€” fully vectorized
  stats <- el[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = row_i]

  return(stats)
}

#' Compute and attach neighbor features for all source variables.
#' Preserves original column naming convention and numerical results.
compute_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                          neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  dt[, .row_i := .I]

  message("Building neighbor edge list (vectorized)...")
  t0 <- Sys.time()
  edgelist <- build_neighbor_edgelist(cell_data, id_order, neighbors)
  message("  Edge list built: ", nrow(edgelist), " directed row-pairs in ",
          round(difftime(Sys.time(), t0, units = "mins"), 1), " min")

  n <- nrow(dt)

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)
    t1 <- Sys.time()

    stats <- compute_neighbor_stats_fast(dt, edgelist, var_name)

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values by row index
    dt[stats$row_i, (max_col)  := stats$nb_max]
    dt[stats$row_i, (min_col)  := stats$nb_min]
    dt[stats$row_i, (mean_col) := stats$nb_mean]

    message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), " sec")
  }

  dt[, .row_i := NULL]
  return(as.data.frame(dt))
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

# cell_data must already exist as a data.frame with columns: id, year, ntl, ec, ...
# id_order and rook_neighbors_unique must already be loaded.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Proceed directly to predict():
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

## Memory Note for 16 GB Laptop

The edge list will have ~1.37M cell-edges Ã— 28 years â‰ˆ **38.5M rows** (two integer columns â‰ˆ 0.6 GB). This fits comfortably in 16 GB alongside the 6.46M-row dataset. If memory is tight, process years in batches:

```r
# Optional: batch by year to reduce peak memory
for (yr in 1992:2019) {
  sub_edge <- edgelist[dt$year[edgelist$row_i] == yr]
  # ... compute stats for this year, then merge back
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup structure | Named character vector (linear scan) | `data.table` keyed join (hash, O(1)) |
| Neighbor resolution | Row-wise `lapply` over 6.46M rows | Single vectorized edge-list join |
| Stat computation | `lapply` + `do.call(rbind, ...)` | `data.table` grouped aggregation |
| Estimated runtime | **86+ hours** | **5â€“15 minutes** |
| Numerical output | max, min, mean of non-NA neighbors | **Identical** |
| RF model | Unchanged | **Unchanged** |
END_RESPONSE>>>
