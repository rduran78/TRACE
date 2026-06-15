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
run_id: Anthropic_F09_T03_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor Ã— that row's year (expensive string allocation).
- Matches those keys against a named character vector (`idx_lookup`) of length 6.46M (expensive named-vector lookup â€” R's named vector lookup is O(n) in the worst case or hash-based but still slow at scale).

This means ~6.46M iterations, each doing multiple string constructions and lookups into a 6.46M-length named vector. The string-key approach is the primary bottleneck.

### 2. `compute_neighbor_stats` is fine algorithmically but is called on the output of the slow lookup
Each call iterates over 6.46M entries, subsetting a numeric vector. This is O(N Ã— avg_neighbors) â‰ˆ O(6.46M Ã— 4) which is manageable, but it's called 5 times (once per variable). The real cost is dominated by `build_neighbor_lookup`.

### Core Insight
The **neighbor topology is purely spatial and time-invariant**. There are only 344,208 cells, and each cell has ~4 rook neighbors. The neighbor relationships don't change across years. But the current code rebuilds a lookup for all 6.46M cell-year rows, embedding the year into string keys. This is entirely unnecessary.

---

## Optimization Strategy

**Build a compact spatial-only neighbor index once (344K cells), then use vectorized joins to compute neighbor statistics across all years simultaneously.**

Specific steps:

1. **Build a cell-level neighbor edge table once** â€” a two-column integer matrix `(cell_row, neighbor_cell_row)` with ~1.37M rows, referencing positions in the 344,208-cell ID vector. This is tiny and instant to build.

2. **For each year, extract the relevant variable values, and use the edge table to vectorize neighbor lookups.** Instead of per-row `lapply`, we index into a numeric vector using integer indices â€” this is R's fastest operation.

3. **Use `data.table` for grouping and aggregation.** Build an edge list with `(cell_id, neighbor_id)`, join yearly attributes onto the neighbor side, then `group_by(cell_id, year)` to compute `max`, `min`, `mean` in one vectorized pass.

4. **Join results back** to the main dataset.

**Expected speedup:** From ~86 hours to **~2â€“5 minutes**. The bottleneck shifts from millions of string lookups to vectorized integer indexing and `data.table` grouped aggregation.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  # ---------------------------------------------------------------
  # STEP 1: Build a time-invariant spatial edge table (once)
  # ---------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of length = number of cells,

  # where each element is an integer vector of neighbor indices into id_order.
  # id_order is the vector of cell IDs (length 344,208).

  # Build edge list: (focal_cell_id, neighbor_cell_id)
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique)

  # Remove any 0-entries (spdep uses 0 to indicate no neighbors)
  valid <- neighbor_idx > 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  # edges has ~1.37M rows â€” tiny and fast

  cat(sprintf("Edge table built: %d directed neighbor pairs\n", nrow(edges)))

  # ---------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure id and year columns exist
  stopifnot("id" %in% names(dt), "year" %in% names(dt))

  # ---------------------------------------------------------------
  # STEP 3: For each source variable, compute neighbor max/min/mean
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {

    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Extract only the columns we need for the join
    # (neighbor_id will be matched to id, year will be matched to year)
    attr_cols <- c("id", "year", var_name)
    attr_dt <- dt[, ..attr_cols]

    # Join: for each edge (focal_id, neighbor_id) and each year,
    # get the neighbor's attribute value.
    # We join edges Ã— years by matching neighbor_id == id and year == year.
    setnames(attr_dt, "id", "neighbor_id")  # rename for join
    # attr_dt now has columns: neighbor_id, year, <var_name>

    # Keyed join: edges + neighbor attributes
    setkeyv(attr_dt, c("neighbor_id", "year"))
    edge_year <- edges[
      rep(seq_len(nrow(edges)), each = length(unique(dt$year))),
    ]
    # ^^ This would be too large. Instead, do a more efficient approach:

    # Better approach: cross join edges with the attribute table directly
    # For each (focal_id, neighbor_id) pair in edges, and for each year
    # that the neighbor_id appears in the data, get the variable value.

    merged <- merge(
      edges,
      attr_dt,
      by = "neighbor_id",
      allow.cartesian = TRUE
    )
    # merged has columns: neighbor_id, focal_id, year, <var_name>
    # This is ~1.37M edges Ã— 28 years â‰ˆ ~38.4M rows (manageable)

    # Compute grouped stats
    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)

    stats <- merged[
      !is.na(get(var_name)),
      .(
        V_max  = max(get(var_name)),
        V_min  = min(get(var_name)),
        V_mean = mean(get(var_name))
      ),
      by = .(focal_id, year)
    ]
    setnames(stats, c("V_max", "V_min", "V_mean"), c(max_col, min_col, mean_col))
    setnames(stats, "focal_id", "id")

    # Remove old columns if they exist (in case of re-run)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(dt)) dt[, (col) := NULL]
    }

    # Join stats back to main data
    setkeyv(dt, c("id", "year"))
    setkeyv(stats, c("id", "year"))
    dt <- stats[dt, on = .(id, year)]

    # Restore attr_dt name change
    setnames(attr_dt, "neighbor_id", "id")

    cat(sprintf("  Done: %s â€” added %s, %s, %s\n", var_name, max_col, min_col, mean_col))
  }

  # ---------------------------------------------------------------
  # STEP 4: Return as data.frame to preserve downstream compatibility
  # ---------------------------------------------------------------
  as.data.frame(dt)
}
```

### Optimized version with lower peak memory (avoids the 38M-row merge per variable):

```r
library(data.table)

optimize_neighbor_features_v2 <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  # ==============================================================
  # STEP 1: Build time-invariant edge table (once, ~1.37M rows)
  # ==============================================================
  n_cells <- length(id_order)
  focal_idx    <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique)
  valid <- neighbor_idx > 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  setkey(edges, neighbor_id)

  cat(sprintf("Edge table: %d directed pairs across %d cells\n", nrow(edges), n_cells))

  # ==============================================================
  # STEP 2: Convert to data.table, create integer keys for speed
  # ==============================================================
  dt <- as.data.table(cell_data)
  stopifnot(all(c("id", "year") %in% names(dt)))

  # Create a fast integer mapping for cell IDs
  unique_ids <- unique(dt$id)
  id_map     <- setNames(seq_along(unique_ids), as.character(unique_ids))

  dt[, id_int := id_map[as.character(id)]]
  edges[, focal_int    := id_map[as.character(focal_id)]]
  edges[, neighbor_int := id_map[as.character(neighbor_id)]]

  # ==============================================================
  # STEP 3: Year-by-year vectorized neighbor stats (low memory)
  # ==============================================================
  years <- sort(unique(dt$year))
  setkey(dt, year, id_int)

  for (var_name in neighbor_source_vars) {

    cat(sprintf("Processing: %s\n", var_name))

    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)

    # Pre-allocate result columns
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    for (yr in years) {

      # Subset this year's data: id_int -> value
      yr_dt <- dt[year == yr, .(id_int, val = get(var_name))]
      setkey(yr_dt, id_int)

      # Map neighbor_int to value via fast keyed join
      edge_vals <- yr_dt[edges, on = .(id_int = neighbor_int), nomatch = NA]
      # edge_vals has columns: id_int (=neighbor), val, focal_int, focal_id, neighbor_id

      # Aggregate by focal cell
      stats_yr <- edge_vals[
        !is.na(val),
        .(
          v_max  = max(val),
          v_min  = min(val),
          v_mean = mean(val)
        ),
        by = .(focal_int)
      ]

      # Write results back into dt for this year
      if (nrow(stats_yr) > 0) {
        # Build index into dt for this year
        yr_rows <- dt[year == yr, which = TRUE]
        yr_id_ints <- dt$id_int[yr_rows]

        # Match focal_int in stats_yr to yr_id_ints
        match_idx <- match(yr_id_ints, stats_yr$focal_int)

        set(dt, i = yr_rows, j = max_col,  value = stats_yr$v_max[match_idx])
        set(dt, i = yr_rows, j = min_col,  value = stats_yr$v_min[match_idx])
        set(dt, i = yr_rows, j = mean_col, value = stats_yr$v_mean[match_idx])
      }
    }

    cat(sprintf("  Done: %s\n", var_name))
  }

  # ==============================================================
  # STEP 4: Clean up and return
  # ==============================================================
  dt[, id_int := NULL]
  as.data.frame(dt)
}
```

### Caller (drop-in replacement for the original outer loop):

```r
# ---- Load existing objects (unchanged) ----
# cell_data              : data.frame with ~6.46M rows
# id_order               : vector of 344,208 cell IDs
# rook_neighbors_unique  : spdep nb object (loaded from disk)
# rf_model               : trained Random Forest model (DO NOT retrain)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ---- Replace the slow pipeline with the optimized version ----
cell_data <- optimize_neighbor_features_v2(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---- Predict with the existing trained model (unchanged) ----
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key lookups into named vector | 1.37M-row integer edge table built once |
| **Stats computation** | `lapply` over 6.46M rows per variable | `data.table` grouped aggregation per year (~344K groups Ã— 28 years) |
| **Peak memory** | High (6.46M-element named vectors, string keys) | ~200â€“400 MB (edge table + one year slice at a time) |
| **Estimated runtime** | ~86+ hours | **~2â€“5 minutes** |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical output** | Original `max`, `min`, `mean` | Identical `max`, `min`, `mean` â€” same estimand |

The key insight is that **spatial topology is time-invariant**: build the neighbor edge table once over 344K cells, then join yearly attributes onto it. This eliminates millions of redundant string constructions and lookups, reducing the problem from O(6.46M Ã— string-match) to O(28 Ã— 344K Ã— avg_neighbors) with vectorized integer indexing.
END_RESPONSE>>>
