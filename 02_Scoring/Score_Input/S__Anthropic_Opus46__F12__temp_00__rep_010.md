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
run_id: Anthropic_F12_T00_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` â€” Colleague's Claim

`do.call(rbind, result)` on ~6.46 million 3-element vectors is indeed inefficient (it creates a matrix row-by-row), and could be replaced. But this operation runs only **5 times** (once per variable). Binding 6.46M small vectors into a matrix is on the order of seconds to low minutes per call â€” noticeable but not the 86-hour bottleneck.

### `build_neighbor_lookup()` â€” The True Bottleneck

This is the real culprit. It runs a `lapply` over **every row** (~6.46 million rows), and inside each iteration it:

1. **Calls `as.character()` and performs a named-vector lookup** (`id_to_ref[as.character(data$id[i])]`) â€” character coercion + hash lookup per row.
2. **Subsets `id_order` by the neighbor list** to get neighbor cell IDs.
3. **Calls `paste()` to construct composite keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” string allocation per row.
4. **Looks up those keys in `idx_lookup`**, a named character vector of length 6.46M â€” this is a **linear-scan or hash lookup on a massive named vector, repeated millions of times**.
5. **Filters NAs** from the result.

The critical insight: `idx_lookup` is a named vector with **6.46 million entries**. Named vector lookup in R uses hashing, but constructing millions of paste keys and performing millions of hash lookups is extremely expensive. With ~4 neighbors per cell on average (rook contiguity), that's ~25.8 million string constructions and hash lookups â€” **per call**. And the entire function is called once, but the `lapply` body runs 6.46M times with string operations each time.

**The bottleneck is `build_neighbor_lookup()`**: specifically, the per-row `paste()` key construction and named-vector lookup against a 6.46M-entry lookup table, executed 6.46 million times.

### Why 86+ hours?

- 6.46M iterations Ã— (character coercion + paste + hash lookup on 6.46M-key vector + NA filtering) â‰ˆ catastrophic.
- `compute_neighbor_stats` by contrast is just numeric indexing into a pre-built integer-index list â€” fast.

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()` entirely** â€” eliminate the per-row `lapply`. Instead, use a `data.table` join to map (id, year) â†’ row index, then expand the neighbor list to an edge list and join in bulk.

2. **Replace `do.call(rbind, ...)` in `compute_neighbor_stats()`** with a pre-allocated matrix and direct vectorized aggregation via `data.table` grouping â€” this addresses the colleague's concern too, though it's secondary.

3. **Preserve the trained Random Forest model** â€” we only change feature engineering / data prep, not the model.

4. **Preserve the original numerical estimand** â€” the computed neighbor max, min, mean values will be identical.

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED build_neighbor_lookup
# ==============================================================
# Strategy: convert the nb object to a flat edge list, then use
# data.table keyed joins to resolve (neighbor_id, year) -> row_index
# in one vectorized pass. No per-row paste/lookup.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # --- Convert data to data.table if not already ---
  dt <- as.data.table(data)
  dt[, row_idx := .I]  # original row index

  # --- Build edge list from nb object ---
  # neighbors[[i]] gives the neighbor indices (into id_order) for id_order[i]
  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  # Map ref indices to actual cell IDs
  edge_dt <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # --- Expand edges across all years present in data ---
  # Get unique years
  years <- sort(unique(dt$year))

  # For each row in dt, we need its id and year to find its neighbors.
  # Approach: join dt (as "from" rows) to edge_dt to get neighbor IDs,
  # then join again to dt to get neighbor row indices.

  # Step 1: Create a keyed version of dt for lookups
  # Key: (id, year) -> row_idx
  dt_key <- dt[, .(id, year, row_idx)]
  setkey(dt_key, id, year)

  # Step 2: For each row in dt, get its neighbor cell IDs via edge_dt

  # First, join dt rows to edge_dt on id == from_id
  setkey(edge_dt, from_id)

  # dt_key has (id, year, row_idx) â€” row_idx is the "from" row
  # We want: for each (from_id=id, year), find all to_id, then find row_idx of (to_id, year)

  # Expand: each row in dt gets its neighbor IDs
  from_rows <- dt_key[, .(from_row_idx = row_idx, id, year)]
  setkey(from_rows, id)

  # Join to get neighbor IDs for each row
  expanded <- edge_dt[from_rows, on = .(from_id = id), allow.cartesian = TRUE,
                      nomatch = NULL]
  # expanded now has: from_id, to_id, from_row_idx, year

  # Step 3: Resolve (to_id, year) -> neighbor row index
  expanded[, neighbor_row_idx := dt_key[.(expanded$to_id, expanded$year), row_idx]]

  # Drop NAs (neighbor cell-year combinations not in data)
  expanded <- expanded[!is.na(neighbor_row_idx)]

  # Step 4: Build the lookup list, indexed by from_row_idx
  n_rows <- nrow(dt)

  # Split neighbor_row_idx by from_row_idx
  # Use data.table grouping for speed
  lookup_dt <- expanded[, .(neighbors = list(as.integer(neighbor_row_idx))),
                         by = from_row_idx]

  # Initialize full list with empty integer vectors
  neighbor_lookup <- vector("list", n_rows)
  for (i in seq_len(n_rows)) {
    neighbor_lookup[[i]] <- integer(0)
  }

  # Fill in the non-empty entries
  neighbor_lookup[lookup_dt$from_row_idx] <- lookup_dt$neighbors


  return(neighbor_lookup)
}


# ==============================================================
# OPTIMIZED compute_neighbor_stats
# ==============================================================
# Strategy: use the expanded edge list directly with data.table
# grouped aggregation â€” no per-element lapply, no do.call(rbind).

compute_neighbor_stats_fast <- function(data, neighbor_lookup_edges, var_name) {
  # neighbor_lookup_edges: data.table with (from_row_idx, neighbor_row_idx)
  # This avoids rebuilding from the list form.

  n_rows <- nrow(data)
  vals <- data[[var_name]]

  # Attach neighbor values
  edges <- copy(neighbor_lookup_edges)
  edges[, neighbor_val := vals[neighbor_row_idx]]

  # Drop NAs in the variable
  edges_clean <- edges[!is.na(neighbor_val)]

  # Aggregate by from_row_idx
  agg <- edges_clean[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = from_row_idx]

  # Build output columns, defaulting to NA
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)

  out_max[agg$from_row_idx]  <- agg$nb_max
  out_min[agg$from_row_idx]  <- agg$nb_min
  out_mean[agg$from_row_idx] <- agg$nb_mean

  cbind(out_max, out_min, out_mean)
}


# ==============================================================
# FULL OPTIMIZED PIPELINE
# ==============================================================
# This replaces the original outer loop. The trained RF model is
# untouched â€” we only rebuild the neighbor features identically.

run_optimized_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  cat("Building vectorized neighbor edge list...\n")

  # --- Build flat edge list from nb object ---
  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

  edge_dt <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # --- Key the data for fast joins ---
  dt_key <- dt[, .(id, year, row_idx)]
  setkey(dt_key, id, year)

  # --- Expand edges: join each (from_id) to all its years ---
  from_info <- dt_key[, .(from_row_idx = row_idx, from_id = id, year)]

  cat("Joining edges to row indices (vectorized)...\n")

  # Join: for each row, get its neighbor cell IDs
  setkey(edge_dt, from_id)
  setkey(from_info, from_id)

  expanded <- edge_dt[from_info, on = .(from_id), allow.cartesian = TRUE,
                      nomatch = NULL]
  # Columns: from_id, to_id, from_row_idx, year

  # Resolve neighbor row indices via keyed join
  expanded[dt_key, on = .(to_id = id, year = year),
           neighbor_row_idx := i.row_idx]

  # Drop unresolved

  expanded <- expanded[!is.na(neighbor_row_idx)]

  # Keep only what we need
  lookup_edges <- expanded[, .(from_row_idx, neighbor_row_idx)]

  cat(sprintf("Edge list built: %s directed neighbor-year edges.\n",
              format(nrow(lookup_edges), big.mark = ",")))

  # --- Compute neighbor stats for each variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))

    stats_mat <- compute_neighbor_stats_fast(dt, lookup_edges, var_name)

    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := stats_mat[, 1]]
    dt[, (min_col)  := stats_mat[, 2]]
    dt[, (mean_col) := stats_mat[, 3]]
  }

  # Remove helper column and return as data.frame to preserve compatibility
  dt[, row_idx := NULL]

  cat("Done. Neighbor features added.\n")
  return(as.data.frame(dt))
}

# ==============================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================
# cell_data <- run_optimized_neighbor_pipeline(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # The trained Random Forest model is unchanged.
# # Predictions proceed as before:
# # predictions <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| `do.call(rbind, result)` | "Main bottleneck" | **Secondary** â€” runs 5Ã— on 6.46M rows; slow but minutes, not hours |
| `compute_neighbor_stats` list binding | "Repeated list binding" | No repeated binding â€” single `lapply` + one `rbind`; mischaracterized |
| **`build_neighbor_lookup`** | Not mentioned | **TRUE BOTTLENECK** â€” 6.46M iterations of `paste()` + named-vector hash lookup against a 6.46M-key vector |

**Root cause**: `build_neighbor_lookup` performs **O(n Ã— k)** string constructions and hash lookups where n = 6.46M rows and k â‰ˆ 4 neighbors, totaling ~25.8M `paste` + hash operations against a massive named vector â€” all inside a sequential `lapply`. This dominates the 86+ hour runtime.

**The fix**: Replace the per-row `lapply` with a single vectorized `data.table` join that expands the spatial neighbor graph across all years in one pass, reducing the operation from hours to minutes. The `compute_neighbor_stats` function is similarly vectorized as a grouped aggregation. Expected runtime: **minutes instead of 86+ hours**. The trained Random Forest model and all numerical outputs are preserved exactly.
END_RESPONSE>>>
