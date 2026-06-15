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
run_id: Anthropic_F04_T10_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`, not `compute_neighbor_stats`. Here's why:

**`build_neighbor_lookup`** iterates over ~6.46 million rows with `lapply`, and for each row it:

1. Performs a **named-vector character lookup** (`id_to_ref[as.character(...)]`) â€” this is O(n) hash lookup but with repeated `as.character` coercion across millions of rows.
2. Constructs **character key strings** via `paste(..., sep="_")` for every neighbor of every row â€” millions of `paste` calls, each producing a small character vector.
3. Looks up those keys in `idx_lookup` â€” another named-vector character lookup, repeated per-neighbor, per-row.

With ~6.46M rows and an average of ~4 rook neighbors per cell, this means roughly **25.8 million individual string constructions and hash lookups**, all inside a single-threaded R `lapply`. String operations in R are notoriously slow because they are not vectorized at the C level in the same way numeric operations are. The `lapply` over 6.46M elements also has per-iteration R interpreter overhead.

**`compute_neighbor_stats`** is lighter, but still iterates 6.46M times per variable (5 variables Ã— 6.46M = 32.3M iterations), calling `max`, `min`, `mean` on small vectors each time.

**Combined**: The nested R-level loops, repeated string allocation, and per-element function-call overhead explain the 86+ hour estimate.

---

## Optimization Strategy

### Principle: Replace row-level R loops and string key construction with vectorized integer-index operations using `data.table`.

1. **Eliminate `build_neighbor_lookup` entirely as a per-row `lapply`.** Instead, build a flat `data.table` of `(row_index, neighbor_row_index)` pairs using vectorized joins â€” no `paste` keys, no character lookups.

2. **Replace `compute_neighbor_stats` with a single grouped `data.table` aggregation** per variable â€” `max`, `min`, `mean` computed in C-level `data.table` internals, not R-level `lapply`.

3. **Memory-safe**: The edge table will have ~25.8M rows Ã— 2 integer columns â‰ˆ 200 MB, well within 16 GB RAM.

4. **Preserves numerical output exactly**: same `max`, `min`, `mean` over the same neighbor sets, same variable names appended. The trained Random Forest model is untouched.

**Expected speedup**: From 86+ hours to roughly **5â€“15 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

# â”€â”€ Step 1: Build a flat edge table (vectorized, no lapply) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_table <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # Map each cell id to its position in id_order
  # id_order is the vector of unique cell IDs corresponding to the nb object
  n_cells <- length(id_order)

  # Expand the nb list into a flat edge list (cell_position -> neighbor_position)
  # This is over spatial cells only (~344K), not cell-years, so it's fast.
  from_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique)

  # Remove zero-neighbor entries (spdep uses integer(0) or 0 for no-neighbors)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  # Map spatial positions to cell IDs
  from_id <- id_order[from_idx]
  to_id   <- id_order[to_idx]

  # Build a spatial edge table: from_id -> to_id
  spatial_edges <- data.table(from_id = from_id, to_id = to_id)

  # Now join with the cell-year data to expand to cell-year rows.
  # We need: for each (from_id, year) row, find all (to_id, year) rows.

  # Add row indices to the main data
  cell_data_dt[, row_idx := .I]

  # Create a keyed lookup: (id, year) -> row_idx
  # We'll join the spatial edges with the data twice: once for "from", once for "to"

  # Key columns for joining
  from_lookup <- cell_data_dt[, .(from_id = id, year, from_row = row_idx)]
  to_lookup   <- cell_data_dt[, .(to_id = id, year, to_row = row_idx)]

  # Join: spatial_edges Ã— years

  # For each (from_id, to_id) pair, match all years present in from_id's data
  # First join spatial_edges to from_lookup on from_id to get (from_id, to_id, year, from_row)
  setkey(spatial_edges, from_id)
  setkey(from_lookup, from_id)
  edges_with_year <- spatial_edges[from_lookup,
                                    .(to_id, year, from_row),
                                    on = "from_id",
                                    allow.cartesian = TRUE,
                                    nomatch = 0L]

  # Now join to to_lookup on (to_id, year) to get to_row
  setkey(edges_with_year, to_id, year)
  setkey(to_lookup, to_id, year)
  edge_table <- edges_with_year[to_lookup,
                                 .(from_row, to_row),
                                 on = c("to_id", "year"),
                                 nomatch = 0L]

  return(edge_table)
}


# â”€â”€ Step 2: Compute neighbor stats via grouped data.table aggregation â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_dt <- function(cell_data_dt, edge_table, var_name) {
  # Extract the variable values indexed by to_row (the neighbor rows)
  edge_table[, val := cell_data_dt[[var_name]][to_row]]

  # Remove NAs in neighbor values
  valid_edges <- edge_table[!is.na(val)]

  # Grouped aggregation: for each from_row, compute max, min, mean
  agg <- valid_edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = from_row]

  # Initialize result columns with NA
  n <- nrow(cell_data_dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)

  # Fill in computed values
  col_max[agg$from_row]  <- agg$nb_max
  col_min[agg$from_row]  <- agg$nb_min
  col_mean[agg$from_row] <- agg$nb_mean

  # Clean up temporary column
  edge_table[, val := NULL]

  # Build output column names consistent with original pipeline
  # (adjust naming convention to match your compute_and_add_neighbor_features output)
  max_name  <- paste0(var_name, "_neighbor_max")
  min_name  <- paste0(var_name, "_neighbor_min")
  mean_name <- paste0(var_name, "_neighbor_mean")

  list(
    names  = c(max_name, min_name, mean_name),
    values = list(col_max, col_min, col_mean)
  )
}


# â”€â”€ Step 3: Full pipeline replacement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  cell_data_dt <- as.data.table(cell_data)

  message("Building vectorized edge table...")
  t0 <- Sys.time()
  edge_table <- build_neighbor_edge_table(cell_data_dt, id_order, rook_neighbors_unique)
  message("  Edge table: ", nrow(edge_table), " edges built in ",
          round(difftime(Sys.time(), t0, units = "secs"), 1), "s")

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)
    t1 <- Sys.time()
    stats <- compute_neighbor_stats_dt(cell_data_dt, edge_table, var_name)
    for (j in seq_along(stats$names)) {
      set(cell_data_dt, j = stats$names[j], value = stats$values[[j]])
    }
    message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), "s")
  }

  # Remove helper column
  cell_data_dt[, row_idx := NULL]

  # Return as data.frame if the original was a data.frame (for RF compatibility)
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(cell_data_dt))
  }
  return(cell_data_dt)
}


# â”€â”€ Usage (drop-in replacement for the original outer loop) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with `paste`/string hash | Single vectorized `data.table` equi-join (~25.8M row edge table) |
| **Stat computation** | 6.46M `lapply` iterations Ã— 5 vars (R-level `max/min/mean`) | 5 grouped `data.table` aggregations (C-level) |
| **String operations** | ~25.8M `paste()` calls + character key lookups | Zero string operations in hot path |
| **Estimated time** | 86+ hours | ~5â€“15 minutes |
| **Peak memory** | Millions of small R list elements (high GC pressure) | ~200 MB flat integer edge table + ~150 MB aggregation workspace |
| **Numerical output** | `max`, `min`, `mean` of non-NA neighbor values | Identical â€” same neighbor sets, same aggregation functions |
| **RF model** | Preserved | Preserved (untouched) |
END_RESPONSE>>>
