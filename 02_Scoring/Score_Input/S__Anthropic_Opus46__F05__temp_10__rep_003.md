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
run_id: Anthropic_F05_T10_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: Row-Level `paste`/`match` Inside `lapply` Over 6.46M Rows

1. **`idx_lookup` construction** (`paste` over 6.46M rows, then `setNames`) happens once â€” that's fine.
2. **But inside the `lapply` over every row `i`** (6.46M iterations), the code:
   - Calls `paste()` to build string keys for every neighbor of every row.
   - Performs a named-vector lookup (`idx_lookup[neighbor_keys]`) which is an O(n) hash probe *per key*.
   - This means ~1.37M neighbor relationships Ã— 28 years Ã— string allocation + hash lookup per cell-year = billions of string operations.

3. **The neighbor topology is year-invariant.** The rook neighbors don't change across years. The entire `build_neighbor_lookup` function re-discovers, via string manipulation, information that could be expressed as a simple integer offset: "for row `i` in year `t`, its neighbors are at rows `j1, j2, j3, â€¦` in the same year `t`." This is a **pure indexing problem** that never needs strings.

4. **`compute_neighbor_stats` then loops again** over 6.46M entries in `lapply`, calling R-level subsetting and `max`/`min`/`mean` per row. This is done 5 times (once per variable), but the neighbor structure is identical each time.

### Summary: Two Systemic Inefficiencies

| Layer | Problem | Impact |
|-------|---------|--------|
| **Neighbor lookup construction** | String-keyed join over 6.46M rows via `lapply` with `paste` per row | ~hours of string allocation/GC |
| **Neighbor stats computation** | R-level `lapply` over 6.46M rows, repeated 5Ã— for 5 variables | ~hours of interpreted-loop overhead |

Both can be eliminated with a **vectorized, integer-index, matrix-based** reformulation.

---

## Optimization Strategy

### Key Insight: Year-Invariant Topology â†’ Integer Arithmetic

If the data is sorted by `(id, year)` â€” or even just by `(year, id)` â€” and every cell appears in every year, then the neighbor indices for year `t` are a fixed integer offset from year `t'`. We can:

1. **Build the neighbor lookup once as an integer edge list** (from-row â†’ to-row) for the entire panel using vectorized operations â€” no strings, no `lapply`.
2. **Compute all neighbor statistics in one vectorized pass per variable** using the edge list and `data.table` grouped aggregation, or sparse-matrix multiplication for means/sums and row-wise operations for min/max.

### Approach: Edge-List + `data.table` Grouped Aggregation

- Construct an edge list: `data.frame(from_row = ..., to_row = ...)` mapping each cell-year row to its neighbor cell-year rows.
- For each variable, extract neighbor values via vector indexing `vals[edge$to_row]`, then aggregate by `from_row` using `data.table`.
- This replaces billions of string ops and millions of R-level `lapply` iterations with pure vectorized integer indexing + `data.table` grouped ops.

**Expected speedup:** From ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 0: Ensure data is a data.table, sorted consistently
# =============================================================================
# cell_data must have columns: id, year, and all predictor variables.
# rook_neighbors_unique is an nb object (list of integer neighbor indices)
#   aligned to id_order (a vector of cell IDs).

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars) {
  
  # Convert to data.table if needed (by reference if already one)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # ------------------------------------------------------------------
  # STEP 1: Build a spatial edge list (cell-level, year-invariant)
  #
  # rook_neighbors_unique[[k]] gives the indices (into id_order) of
  # the neighbors of id_order[k].
  # We build: from_cell_id -> to_cell_id
  # ------------------------------------------------------------------
  n_cells <- length(id_order)
  
  # Number of neighbors per cell
  n_neighbors <- vapply(rook_neighbors_unique, length, integer(1))
  
  # "from" index into id_order (repeated for each neighbor)
  from_idx <- rep(seq_len(n_cells), times = n_neighbors)
  
  # "to" index into id_order
  to_idx <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Convert to actual cell IDs
  from_cell <- id_order[from_idx]
  to_cell   <- id_order[to_idx]
  
  # Spatial edge list (year-invariant)
  spatial_edges <- data.table(from_cell = from_cell, to_cell = to_cell)
  
  cat(sprintf("Spatial edge list: %d directed edges\n", nrow(spatial_edges)))
  
  # ------------------------------------------------------------------
  # STEP 2: Map cell IDs to row numbers in cell_data, per year
  #
  # Instead of string keys, we use integer join.
  # Create a row-index column.
  # ------------------------------------------------------------------
  cell_data[, .row_idx := .I]
  
  # Keyed lookup table: (id, year) -> row index
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  
  # ------------------------------------------------------------------
  # STEP 3: Expand spatial edges across all years â†’ full panel edge list
  #
  # Every spatial edge (from_cell, to_cell) is valid for every year.
  # We cross-join with years.
  # ------------------------------------------------------------------
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  cat(sprintf("Expanding %d spatial edges across %d years...\n",
              nrow(spatial_edges), n_years))
  
  # Replicate edge list for each year (vectorized)
  full_edges <- spatial_edges[rep(seq_len(.N), times = n_years)]
  full_edges[, year := rep(years, each = nrow(spatial_edges))]
  
  cat(sprintf("Full panel edge list: %d directed cell-year edges\n", nrow(full_edges)))
  
  # ------------------------------------------------------------------
  # STEP 4: Map (from_cell, year) and (to_cell, year) to row indices
  #
  # Use data.table keyed joins â€” O(n log n), no string operations.
  # ------------------------------------------------------------------
  setkey(row_lookup, id, year)
  
  # Map "from" side
  full_edges[, from_row := row_lookup[.(full_edges$from_cell, full_edges$year), .row_idx]]
  
  # Map "to" side
  full_edges[, to_row := row_lookup[.(full_edges$to_cell, full_edges$year), .row_idx]]
  
  # Drop edges where either side is missing (cell not present in that year)
  full_edges <- full_edges[!is.na(from_row) & !is.na(to_row)]
  
  cat(sprintf("Valid panel edges after join: %d\n", nrow(full_edges)))
  
  # We only need integer row indices from here
  edge_from <- full_edges$from_row
  edge_to   <- full_edges$to_row
  
  # Free memory
  rm(full_edges, spatial_edges, row_lookup, from_idx, to_idx, from_cell, to_cell)
  gc()
  
  # ------------------------------------------------------------------
  # STEP 5: For each variable, compute neighbor max, min, mean
  #         using vectorized indexing + data.table grouped aggregation
  # ------------------------------------------------------------------
  
  # Pre-build a data.table with just the from_row for grouping
  edge_dt <- data.table(from_row = edge_from, to_row = edge_to)
  setkey(edge_dt, from_row)
  
  n_rows <- nrow(cell_data)
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))
    
    # Extract neighbor values via integer indexing (vectorized)
    vals <- cell_data[[var_name]]
    edge_dt[, nval := vals[to_row]]
    
    # Grouped aggregation â€” extremely fast in data.table
    agg <- edge_dt[!is.na(nval),
                   .(nb_max  = max(nval),
                     nb_min  = min(nval),
                     nb_mean = mean(nval)),
                   by = from_row]
    
    # Initialize result columns with NA
    max_col  <- paste0("max_nb_", var_name)
    min_col  <- paste0("min_nb_", var_name)
    mean_col <- paste0("mean_nb_", var_name)
    
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]
    
    # Assign by integer index (vectorized)
    cell_data[agg$from_row, (max_col)  := agg$nb_max]
    cell_data[agg$from_row, (min_col)  := agg$nb_min]
    cell_data[agg$from_row, (mean_col) := agg$nb_mean]
    
    cat(sprintf("  Done: %s â€” %d rows with valid neighbor stats\n",
                var_name, nrow(agg)))
  }
  
  # Clean up helper column
  edge_dt[, nval := NULL]
  cell_data[, .row_idx := NULL]
  
  cat("All neighbor features computed.\n")
  return(cell_data)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names (max_nb_*, min_nb_*, mean_nb_*) and numerical values
# are identical to the original implementation.
```

---

## Memory Budget Check

| Object | Approximate Size |
|--------|-----------------|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `edge_from` + `edge_to` (1.37M Ã— 28 = ~38.5M integers Ã— 2) | ~0.6 GB |
| `edge_dt` with `nval` column | ~0.9 GB |
| Aggregation temporaries | ~0.3 GB |
| **Total peak** | **~7.5 GB** âœ“ fits 16 GB |

If memory is tight (the full `full_edges` data.table before subsetting to integer vectors is the peak), you can build it in year-chunks:

```r
# OPTIONAL: Year-chunked variant if memory is very tight
# Replace STEP 3-4 with:

edge_from_list <- vector("list", n_years)
edge_to_list   <- vector("list", n_years)

setkey(row_lookup, id, year)

for (yi in seq_along(years)) {
  yr <- years[yi]
  yr_edges <- copy(spatial_edges)  # shallow is fine, but copy to be safe
  yr_edges[, year := yr]
  
  yr_edges[, from_row := row_lookup[.(yr_edges$from_cell, yr), .row_idx]]
  yr_edges[, to_row   := row_lookup[.(yr_edges$to_cell,   yr), .row_idx]]
  yr_edges <- yr_edges[!is.na(from_row) & !is.na(to_row)]
  
  edge_from_list[[yi]] <- yr_edges$from_row
  edge_to_list[[yi]]   <- yr_edges$to_row
}

edge_from <- unlist(edge_from_list, use.names = FALSE)
edge_to   <- unlist(edge_to_list,   use.names = FALSE)
rm(edge_from_list, edge_to_list)
```

---

## Why This Preserves Correctness

| Property | Original | Optimized |
|----------|----------|-----------|
| Neighbor definition | `rook_neighbors_unique` via `spdep::nb` | Same `nb` object, same edges |
| Year matching | `paste(id, year)` string equality | `data.table` integer join on `(id, year)` â€” identical semantics |
| Aggregation | `max()`, `min()`, `mean()` per row | Identical functions via `data.table` grouped-by `from_row` |
| Missing handling | `!is.na()` filter, return `NA` if no valid neighbors | Same: `edge_dt[!is.na(nval), ...]`, rows without matches remain `NA` |
| Column names | `max_nb_*`, `min_nb_*`, `mean_nb_*` | Identical |
| Random Forest | Not touched | Not touched |

**The numerical estimand is exactly preserved.** The only change is algorithmic: replacing O(N) string construction + hash lookups inside a 6.46M-iteration `lapply` with vectorized integer indexing and `data.table` grouped aggregation.

**Expected runtime: 3â€“10 minutes** (vs. 86+ hours), a ~500â€“1700Ã— speedup.
END_RESPONSE>>>
