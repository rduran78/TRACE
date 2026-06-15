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
run_id: Anthropic_F04_T07_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is definitively in `build_neighbor_lookup`, not `compute_neighbor_stats` or the Random Forest. Here's why:

**`build_neighbor_lookup`** iterates a `lapply` over **~6.46 million rows**. For each row it:
1. Looks up the cell's reference index via a named character vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs **string keys** by pasting neighbor IDs with the current year (`paste(..., sep="_")`).
4. Matches those keys against a **named character vector of 6.46 million entries** (`idx_lookup`).

The string construction and named-vector lookup (which is O(n) hash-table probing per call, repeated millions of times) produces approximately **6.46M Ã— ~4 neighbors Ã— (one `paste` + one named-vector match)** operations. This is catastrophically slow in interpreted R. The resulting `neighbor_lookup` list of 6.46M integer vectors then gets traversed five more times in `compute_neighbor_stats`.

**`compute_neighbor_stats`** is comparatively lighter â€” it's just numeric subsetting and `max/min/mean` â€” but it still loops 6.46M times in R with `lapply` and `do.call(rbind, ...)` on a 6.46M-element list, which is also needlessly slow.

**Key inefficiencies:**
1. **String-key construction and lookup**: `paste()` on millions of rows and named-vector matching is extremely slow.
2. **Row-level R loop**: Pure-R `lapply` over 6.46M rows with non-vectorized bodies.
3. **Redundant work across years**: The spatial neighbor topology is **identical for every year** â€” a cell's rook neighbors don't change from 1992 to 2019. Yet the code rebuilds string keys and re-resolves them for every cell-year row.
4. **`do.call(rbind, ...)`** on a 6.46M-element list of 3-element vectors is a known R anti-pattern.

---

## Optimization Strategy

### Core insight: **Separate space from time.**

The neighbor graph is purely spatial (344,208 cells). The panel has 28 years. Instead of building a 6.46M-row lookup, build a **344K-cell spatial lookup once**, then exploit the regular panel structure (every cell appears in every year, or we handle gaps) to compute neighbor statistics via **vectorized matrix/data.table operations**.

### Specific steps:

1. **Build a spatial-only neighbor lookup** â€” a list of length 344,208 mapping each cell index to its neighbor cell indices. This is essentially just `rook_neighbors_unique` re-indexed. Done once, trivially fast.

2. **Reshape each variable into a cell Ã— year matrix** (344,208 rows Ã— 28 columns). This allows vectorized column-wise (i.e., within-year) operations.

3. **For each variable and each year-column**, gather neighbor values using the spatial index list and compute max/min/mean in a **vectorized** fashion using `data.table` or a pre-allocated matrix approach. The key trick: build a long edge-list `(cell_i, neighbor_j)` once (~1.37M edges), then use `data.table` grouped aggregation per year â€” this turns the entire computation into a fast grouped `mean/max/min` over a ~1.37M-row table, repeated 28 times. Total: ~38.4M grouped operations, handled natively by `data.table` in seconds.

4. **Merge results back** into the original `cell_data` data.frame, preserving column names and numerical values exactly.

**Expected speedup**: From 86+ hours to **minutes** (roughly 2â€“10 minutes depending on I/O).

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  # ---------------------------------------------------------------
  # STEP 0: Convert to data.table for speed; record original order

  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .rowid_orig := .I]  # preserve original row order

  # ---------------------------------------------------------------
  # STEP 1: Build spatial-only edge list (cell index -> neighbor index)
  #
  # id_order is the vector of cell IDs in the same order as

  # rook_neighbors_unique (an nb object): rook_neighbors_unique[[k]]
  # gives the integer indices (into id_order) of neighbors of
  # id_order[k].
  #
  # We build a two-column data.table: (cell_id, neighbor_id)
  # ---------------------------------------------------------------
  n_cells <- length(id_order)

  # Pre-compute lengths for pre-allocation
  n_lengths <- vapply(rook_neighbors_unique, length, integer(1))
  total_edges <- sum(n_lengths)

  # Pre-allocate vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (k in seq_len(n_cells)) {
    nb_idx <- rook_neighbors_unique[[k]]
    n_nb   <- n_lengths[k]
    if (n_nb > 0L) {
      rng <- pos:(pos + n_nb - 1L)
      from_id[rng] <- id_order[k]
      to_id[rng]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }

  edges <- data.table(cell_id = from_id, neighbor_id = to_id)

  # ---------------------------------------------------------------
  # STEP 2: For each source variable, compute neighbor max/min/mean
  #         using vectorized data.table joins and grouped aggregation
  # ---------------------------------------------------------------

  # Create a minimal keyed lookup: (id, year) -> variable values
  # We will join edges against this to get neighbor values.

  # Ensure 'id' and 'year' columns exist
  stopifnot(all(c("id", "year") %in% names(dt)))

  # Key the main table for fast joins
  setkeyv(dt, c("id", "year"))

  # Get unique years
  years <- sort(unique(dt$year))

  for (var_name in neighbor_source_vars) {
    cat("Processing neighbor features for:", var_name, "\n")

    # Extract only the columns we need: id, year, variable
    val_dt <- dt[, .(id, year, val = get(var_name))]
    setkeyv(val_dt, c("id", "year"))

    # For each cell-year, we need the values of its spatial neighbors in the same year.
    # Strategy: join edges with val_dt to get neighbor values, then aggregate.

    # edges has (cell_id, neighbor_id)
    # We want: for each (cell_id, year), get val of each neighbor_id in that year.

    # Build: edges Ã— years  -> (cell_id, neighbor_id, year)
    # Then join to val_dt on (neighbor_id, year) to get neighbor val.
    # Then group by (cell_id, year) to get max, min, mean.

    # But edges Ã— years = 1.37M Ã— 28 = ~38.4M rows â€” very manageable.

    # Efficient approach: loop over years to keep memory bounded,
    # or do it all at once. 38.4M rows Ã— few columns fits in RAM easily.

    # All-at-once approach:
    year_dt <- data.table(year = years)
    edge_year <- edges[, CJ_val := TRUE][
      , .(cell_id, neighbor_id)
    ]

    # Cross join edges with years
    edge_year <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
    edge_year[, cell_id     := edges$cell_id[edge_idx]]
    edge_year[, neighbor_id := edges$neighbor_id[edge_idx]]
    edge_year[, edge_idx := NULL]

    # Join to get neighbor values
    setnames(val_dt, "id", "neighbor_id_join")
    setnames(val_dt, "neighbor_id_join", "id")  # revert â€” let's be clean:

    # Restart val_dt cleanly
    val_dt <- dt[, .(neighbor_id = id, year, val = get(var_name))]
    setkeyv(val_dt, c("neighbor_id", "year"))

    # Keyed join
    setkeyv(edge_year, c("neighbor_id", "year"))
    edge_year <- val_dt[edge_year, on = .(neighbor_id, year)]

    # Now edge_year has columns: neighbor_id, year, val, cell_id
    # Aggregate by (cell_id, year)
    agg <- edge_year[
      !is.na(val),
      .(nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)),
      by = .(cell_id, year)
    ]

    # Determine output column names (must match original pipeline)
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
    setnames(agg, "cell_id", "id")

    # Merge back into dt
    setkeyv(agg, c("id", "year"))
    setkeyv(dt,  c("id", "year"))

    # Remove old columns if they exist (idempotent re-runs)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(dt)) dt[, (col) := NULL]
    }

    dt <- agg[dt, on = .(id, year)]

    cat("  Done:", var_name, "\n")
  }

  # ---------------------------------------------------------------
  # STEP 3: Restore original row order and return as data.frame
  # ---------------------------------------------------------------
  setorder(dt, .rowid_orig)
  dt[, .rowid_orig := NULL]

  return(as.data.frame(dt))
}
```

### Cleaner, more memory-efficient version (year-chunked):

```r
library(data.table)

build_spatial_edge_list <- function(id_order, rook_neighbors_unique) {
  n_cells <- length(id_order)
  n_lengths <- vapply(rook_neighbors_unique, length, integer(1))
  total_edges <- sum(n_lengths)

  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  pos <- 1L

  for (k in seq_len(n_cells)) {
    nb_idx <- rook_neighbors_unique[[k]]
    n_nb   <- n_lengths[k]
    if (n_nb > 0L) {
      rng <- pos:(pos + n_nb - 1L)
      from_id[rng] <- id_order[k]
      to_id[rng]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }

  data.table(from_cell = from_id, to_cell = to_id)
}


compute_all_neighbor_features <- function(cell_data, id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)
  dt[, .orig_order := .I]

  # Build spatial edge list once (~1.37M rows)
  edges <- build_spatial_edge_list(id_order, rook_neighbors_unique)

  years <- sort(unique(dt$year))

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  [%s] Computing neighbor features for: %s\n",
                format(Sys.time(), "%H:%M:%S"), var_name))

    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    # Pre-allocate result columns with NA
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Process year by year to keep peak memory low
    for (yr in years) {
      # Extract values for this year: (id -> val)
      yr_vals <- dt[year == yr, .(id, val = get(var_name))]
      setkeyv(yr_vals, "id")

      # Join neighbor values: for each edge, look up the neighbor's value
      # edges: (from_cell, to_cell)
      # We want: val of to_cell in this year
      edge_vals <- yr_vals[
        edges,
        .(from_cell, val = x.val),
        on = .(id = to_cell),
        nomatch = NA
      ]

      # Aggregate by from_cell
      agg <- edge_vals[
        !is.na(val),
        .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
        by = from_cell
      ]

      # Update dt in place for this year
      if (nrow(agg) > 0L) {
        # Build index for matching
        idx_dt <- dt[year == yr, .(id, .orig_order)]
        idx_dt <- agg[idx_dt, on = .(from_cell = id)]

        # Write results back using .orig_order as row index
        matched <- !is.na(idx_dt$nb_max)
        rows <- idx_dt$.orig_order[matched]
        set(dt, i = rows, j = max_col,  value = idx_dt$nb_max[matched])
        set(dt, i = rows, j = min_col,  value = idx_dt$nb_min[matched])
        set(dt, i = rows, j = mean_col, value = idx_dt$nb_mean[matched])
      }
    }
  }

  setorder(dt, .orig_order)
  dt[, .orig_order := NULL]
  as.data.frame(dt)
}


# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
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
| **Lookup construction** | 6.46M string `paste` + named-vector match in R loop | One spatial edge list (1.37M rows), built once via pre-allocated integer vectors |
| **Neighbor stats** | `lapply` over 6.46M rows, `do.call(rbind, 6.46M-element list)` | `data.table` grouped aggregation per year (~1.37M rows Ã— 28 years), vectorized C-level |
| **String operations** | ~25.8M `paste()` calls + named-vector lookups | **Zero** â€” integer joins only |
| **Memory pattern** | 6.46M-element list of integer vectors + 6.46M-element list of 3-vectors | ~1.37M-row edge table + year-chunked temporaries (~344K rows) |
| **Estimated runtime** | 86+ hours | **~2â€“10 minutes** |
| **Numerical result** | max, min, mean of neighbor values | Identical max, min, mean of neighbor values |
| **RF model** | Preserved | Preserved (untouched) |
END_RESPONSE>>>
