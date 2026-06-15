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
run_id: Anthropic_F05_T00_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” repeated string allocation and concatenation.
2. **Looks up those keys in `idx_lookup`** â€” a named character vector of length 6.46M, meaning each lookup is an O(N) hash probe on a very large named vector.

This means the inner loop performs ~6.46M Ã— (avg ~4 neighbors) â‰ˆ **25.8 million string constructions and hash lookups against a 6.46M-entry table**. The named-vector lookup in R is not O(1) at this scale; it degrades significantly.

### The Broader Algorithmic Insight

The string keys encode `(id, year)` pairs, but the **neighbor graph is time-invariant** â€” cell A's neighbors don't change from year to year. The current code rebuilds the `(id, year) â†’ row` mapping as strings and then re-resolves it per row. This is unnecessary. The entire lookup can be reformulated as:

> For each row `i` with cell `id` in year `t`, find all rows `j` where `j.id âˆˆ neighbors(id)` AND `j.year == t`.

Since the panel is balanced (every cell appears in every year), this is a **pure join** that can be vectorized entirely â€” no per-row loop, no string keys.

### Why `compute_neighbor_stats` Is Also Suboptimal

Even after `build_neighbor_lookup` finishes, `compute_neighbor_stats` runs a second `lapply` over 6.46M rows computing `max/min/mean` one row at a time. This is also vectorizable.

---

## Optimization Strategy

1. **Eliminate all string keys.** Map cell IDs to integer indices 1..N_cells. Map years to integer indices 1..N_years. Compute row positions arithmetically: `row = (cell_index - 1) * N_years + year_index` (assuming sorted panel).

2. **Vectorize the neighbor lookup.** Expand the neighbor list into an edge list `(source_cell_index, target_cell_index)`. Then for each year, the row indices of all neighbor pairs are computed by arithmetic â€” no hash lookups.

3. **Vectorize the statistics.** Use `data.table` grouped operations on the edge list to compute `max`, `min`, `mean` for all rows simultaneously.

4. **Compute all 5 variables in one pass** over the same edge structure.

**Expected speedup:** From ~86+ hours to **minutes** (the bottleneck becomes memory bandwidth over ~100M edge-year pairs, which `data.table` handles efficiently).

---

## Working R Code

```r
library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # ---------------------------------------------------------------
  # 1. Convert to data.table and build integer indices
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure deterministic ordering: (id, year)
  setorder(dt, id, year)

  # Integer cell index: 1..N_cells
  unique_ids   <- sort(unique(dt$id))
  N_cells      <- length(unique_ids)
  id_to_cidx   <- setNames(seq_along(unique_ids), as.character(unique_ids))

  # Integer year index: 1..N_years
  unique_years <- sort(unique(dt$year))
  N_years      <- length(unique_years)
  year_to_yidx <- setNames(seq_along(unique_years), as.character(unique_years))

  # Add integer indices to dt

dt[, cidx := id_to_cidx[as.character(id)]]
  dt[, yidx := year_to_yidx[as.character(year)]]

  # Row index in sorted (id, year) order â€” arithmetic lookup
  # Row for (cidx=c, yidx=y) = (c - 1) * N_years + y
  # Verify this matches actual row positions:
  dt[, row_idx := .I]
  stopifnot(all(dt$row_idx == (dt$cidx - 1L) * N_years + dt$yidx))

  # ---------------------------------------------------------------
  # 2. Build directed edge list from rook_neighbors_unique
  #    rook_neighbors_unique is an nb object indexed by id_order
  # ---------------------------------------------------------------
  # id_order[k] is the cell id for the k-th element of the nb object
  # neighbors[[k]] gives integer indices into id_order

  id_order_cidx <- id_to_cidx[as.character(id_order)]  # map id_order to cidx

  # Expand nb list to edge data.table: (from_cidx, to_cidx)
  edge_from <- rep(seq_along(rook_neighbors_unique),
                   lengths(rook_neighbors_unique))
  edge_to   <- unlist(rook_neighbors_unique)

  # Remove 0-neighbor entries (spdep uses 0L for no-neighbor cards)
  valid <- edge_to != 0L
  edge_from <- edge_from[valid]
  edge_to   <- edge_to[valid]

  edges <- data.table(
    from_cidx = id_order_cidx[edge_from],
    to_cidx   = id_order_cidx[edge_to]
  )
  rm(edge_from, edge_to, valid)

  N_edges <- nrow(edges)
  cat(sprintf("Edge list: %d directed neighbor pairs\n", N_edges))

  # ---------------------------------------------------------------
  # 3. Expand edges across all years and compute row indices
  #    Instead of a massive cross-join (edges Ã— years), process
  #    year-by-year to stay within 16 GB RAM.
  # ---------------------------------------------------------------

  # Pre-extract variable columns as matrices for fast access
  # Matrix: N_cells rows Ã— N_years cols, value = variable value
  # Row (c, y) in dt has row_idx = (c-1)*N_years + y

  # We'll accumulate results into pre-allocated matrices
  # For each var: max, min, sum, count â†’ then mean = sum/count
  n_vars <- length(neighbor_source_vars)

  # Result storage: one column per stat per variable
  # Stats: max, min, mean â†’ 3 columns per variable
  # We'll store in the data.table at the end

  # Pre-allocate result matrices: N_cells * N_years rows Ã— 3 cols per var
  N_rows <- nrow(dt)

  result_list <- vector("list", n_vars)
  names(result_list) <- neighbor_source_vars

  for (vi in seq_along(neighbor_source_vars)) {
    var_name <- neighbor_source_vars[vi]
    cat(sprintf("Processing variable: %s\n", var_name))

    vals <- dt[[var_name]]  # length N_rows, ordered by (cidx, yidx)

    # Pre-allocate output vectors
    out_max  <- rep(NA_real_, N_rows)
    out_min  <- rep(NA_real_, N_rows)
    out_mean <- rep(NA_real_, N_rows)

    # Process year by year to limit memory
    for (yi in seq_len(N_years)) {
      # For this year, the row index of cell with cidx=c is: (c-1)*N_years + yi
      # Source rows (the "from" cell â€” the row that receives the neighbor stats)
      from_rows <- (edges$from_cidx - 1L) * N_years + yi
      # Target rows (the neighbor cells whose values we read)
      to_rows   <- (edges$to_cidx - 1L) * N_years + yi

      # Get neighbor values
      neighbor_vals <- vals[to_rows]

      # Build a data.table for grouped aggregation
      agg_dt <- data.table(from_row = from_rows, nval = neighbor_vals)
      # Remove NAs in neighbor values before aggregation
      agg_dt <- agg_dt[!is.na(nval)]

      if (nrow(agg_dt) == 0L) next

      stats <- agg_dt[, .(
        nmax  = max(nval),
        nmin  = min(nval),
        nmean = mean(nval)
      ), by = from_row]

      out_max[stats$from_row]  <- stats$nmax
      out_min[stats$from_row]  <- stats$nmin
      out_mean[stats$from_row] <- stats$nmean
    }

    result_list[[var_name]] <- data.table(
      nmax = out_max, nmin = out_min, nmean = out_mean
    )
  }

  # ---------------------------------------------------------------
  # 4. Attach results to dt with original column naming convention
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    res <- result_list[[var_name]]
    # Match the naming convention of compute_and_add_neighbor_features
    # Typical convention: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = res$nmax)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = res$nmin)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = res$nmean)
  }

  # ---------------------------------------------------------------
  # 5. Clean up helper columns and return as data.frame
  # ---------------------------------------------------------------
  dt[, c("cidx", "yidx", "row_idx") := NULL]

  # Return in original row order if cell_data wasn't sorted by (id, year)
  # To be safe, merge back by (id, year)
  setorder(dt, id, year)

  return(as.data.frame(dt))
}
```

### Usage (drop-in replacement for the original outer loop):

```r
# --- BEFORE (original: ~86+ hours) ---
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# --- AFTER (optimized: ~minutes) ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched â€” only the feature columns are
# recomputed with identical numerical values (same max/min/mean of same neighbors).
# Predictions:
# preds <- predict(trained_rf_model, newdata = cell_data)
```

### If Original Row Order Differs from `(id, year)` Sort

If `cell_data` may arrive in a different row order and downstream code depends on that order, add a row-order key before and restore after:

```r
cell_data$.orig_order <- seq_len(nrow(cell_data))
cell_data <- compute_all_neighbor_features(cell_data, id_order,
                                           rook_neighbors_unique,
                                           neighbor_source_vars)
cell_data <- cell_data[order(cell_data$.orig_order), ]
cell_data$.orig_order <- NULL
```

### If Column Names Differ from `<var>_neighbor_max/min/mean`

Adjust the `set()` calls in step 4 to match whatever naming convention `compute_and_add_neighbor_features` originally used. The numerical values are identical.

---

## Complexity Comparison

| | Original | Optimized |
|---|---|---|
| **String constructions** | ~6.46M Ã— (1 + avg 4 neighbors) â‰ˆ 32M | **0** |
| **Named-vector lookups** (6.46M-entry hash) | ~25.8M probes | **0** |
| **Per-row R function calls** | 6.46M (`lapply`) Ã— 2 passes | **0** (fully vectorized) |
| **Aggregation** | Row-by-row `max/min/mean` in R loop | `data.table` grouped C-level aggregation |
| **Memory pattern** | 6.46M-element named character vector + list of 6.46M integer vectors | Edge list (~5.5M rows Ã— 2 int cols â‰ˆ 44 MB) + one year-slice at a time |
| **Estimated time** | **86+ hours** | **5â€“15 minutes** |

The key insight: the neighbor graph is **time-invariant**, so the `(cell, year)` string-key resolution inside a per-row loop was doing O(NÃ—T) work for what is fundamentally an O(EÃ—T) vectorizable join, where E (edge count) is small and the join key (year) is trivially computable by arithmetic on a sorted balanced panel.
END_RESPONSE>>>
