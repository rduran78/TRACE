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
run_id: Anthropic_F12_T10_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a linear-time concatenation operation that runs in seconds to low minutes for this data size. The `lapply` inside `compute_neighbor_stats()` does no "repeated list binding" â€” it returns a fixed-length vector `c(NA, NA, NA)` or `c(max, min, mean)` per element, so there is no growing-list pathology.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** Inside the `lapply`, for *each* of the ~6.46M rows, the function:
   - Calls `as.character(data$id[i])` and looks up `id_to_ref[...]` (named character vector lookup â€” O(n) hash probe repeated 6.46M times).
   - Extracts `neighbor_cell_ids` via subsetting `id_order[neighbors[[ref_idx]]]`.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” allocating new strings for every neighbor of every row.
   - Looks up `idx_lookup[neighbor_keys]` â€” probing a named vector of 6.46M entries with string keys, repeated for every neighbor of every row.

2. **Total string operations scale as rows Ã— avg_neighbors.** With ~6.46M rows and an average of ~4 rook neighbors per cell, that's ~25.8 million `paste()` calls and ~25.8 million named-vector string lookups *inside the inner function*, on top of the 6.46M outer iterations. Named vector lookup by string key in R uses hashing, but the repeated `paste` allocation and hash probing at this scale dominates runtime massively â€” likely accounting for **>90% of the 86-hour estimate**.

3. `compute_neighbor_stats()` is comparatively fast: it does only integer indexing into a numeric vector (vectorized, cache-friendly) and computes three summary statistics per element. For 6.46M elements with ~4 neighbors each, this should complete in under a minute.

**Conclusion:** The bottleneck is the O(rows Ã— neighbors) string construction and string-keyed hash lookup in `build_neighbor_lookup()`. The fix is to eliminate all string operations and replace them with integer arithmetic for row indexing.

---

## Optimization Strategy

1. **Replace string-key lookup with integer arithmetic.** Since the data has a regular panel structure (344,208 cells Ã— 28 years), we can map any `(cell, year)` pair to a row index using integer math if we sort the data appropriately, or use an integer-keyed hash (via `match()` on integer encoding or a pre-built integer lookup table).

2. **Vectorize `build_neighbor_lookup()`** by eliminating the per-row `lapply` entirely. Instead, precompute neighbor row indices for all rows at once using vectorized operations.

3. **Vectorize `compute_neighbor_stats()`** using the pre-built neighbor structure with matrix operations instead of per-row `lapply`.

4. **Preserve the trained Random Forest model and original numerical outputs.** The computed features (max, min, mean of neighbor values) will be numerically identical.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# ==============================================================================

# ---------------------------------------------------------------------------
# Step 0: Ensure data is sorted by (id, year) so we can use integer arithmetic.
#         If already sorted, this is a no-op check.
# ---------------------------------------------------------------------------
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

# ---------------------------------------------------------------------------
# Step 1: Build integer-indexed neighbor lookup (vectorized, no strings)
#
# Key insight: if data is sorted by (id, year), then for a cell with
# positional index `c` (1-based among the 344,208 unique cells) and
# year offset `t` (0-based, 0..27 for 1992..2019), the row index is:
#     row = (c - 1) * n_years + (t + 1)
#
# This replaces ALL string paste + named-vector lookups with integer math.
# ---------------------------------------------------------------------------

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  
  unique_ids <- id_order
  n_cells    <- length(unique_ids)
  years      <- sort(unique(data$year))
  n_years    <- length(years)
  
  # Map cell id -> positional index (1..n_cells)
  id_to_pos <- setNames(seq_along(unique_ids), as.character(unique_ids))
  
  # Map year -> offset (1..n_years)
  year_to_offset <- setNames(seq_along(years), as.character(years))
  
  # For each cell position, get its neighbor cell positions
  # neighbors is an nb object: neighbors[[c]] gives integer indices into id_order
  # We will build an edge list: (cell_position, neighbor_position)
  
  # Number of neighbors per cell
  n_nbrs <- lengths(neighbors)
  
  # Expand: for each cell, repeat its index by its number of neighbors
  from_cell <- rep(seq_len(n_cells), times = n_nbrs)
  # The neighbor cell positions (concatenated)
  to_cell   <- unlist(neighbors, use.names = FALSE)
  
  # Total directed neighbor pairs
  n_edges <- length(from_cell)
  
  # Now expand across all years: each edge appears once per year
  # from_row[e, t] = (from_cell[e] - 1) * n_years + t
  # to_row[e, t]   = (to_cell[e]   - 1) * n_years + t
  
  # Vectorize: repeat each edge n_years times, and tile year offsets
  from_cell_exp <- rep(from_cell, each = n_years)
  to_cell_exp   <- rep(to_cell,   each = n_years)
  year_offset   <- rep(seq_len(n_years), times = n_edges)
  
  from_row <- (from_cell_exp - 1L) * n_years + year_offset
  to_row   <- (to_cell_exp   - 1L) * n_years + year_offset
  
  # Return as a data structure: for each "from_row", the list of "to_row" indices

  # But building a 6.46M-element list from edges is itself slow with split().
  # Instead, return the edge vectors sorted by from_row for grouped operations.
  
  ord <- order(from_row)
  list(
    from_row   = from_row[ord],
    to_row     = to_row[ord],
    n_rows     = nrow(data),
    # Precompute group boundaries for fast slicing
    grp_start  = NULL,  # will be filled below
    grp_end    = NULL
  )
}

# ---------------------------------------------------------------------------
# Step 2: Compute neighbor stats vectorized using the edge list
# ---------------------------------------------------------------------------

compute_neighbor_stats_fast <- function(data, edge_from, edge_to, n_rows, var_name) {
  
  vals <- data[[var_name]]
  
  # Get neighbor values along edges
  nbr_vals <- vals[edge_to]
  
  # We need max, min, mean grouped by edge_from
  # edge_from is already sorted, so we can use efficient grouped operations.
  
  # Use data.table for fast grouped aggregation on the edge list
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("data.table package is required for the optimized pipeline.")
  }
  
  dt_edges <- data.table::data.table(
    from = edge_from,
    val  = nbr_vals
  )
  
  # Remove edges where neighbor value is NA
  dt_edges <- dt_edges[!is.na(val)]
  
  # Grouped aggregation
  agg <- dt_edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = from]
  
  # Initialize output with NA
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)
  
  out_max[agg$from]  <- agg$nb_max
  out_min[agg$from]  <- agg$nb_min
  out_mean[agg$from] <- agg$nb_mean
  
  cbind(out_max, out_min, out_mean)
}

# ---------------------------------------------------------------------------
# Step 3: Optimized outer pipeline
# ---------------------------------------------------------------------------

library(data.table)

# Ensure sort order for integer arithmetic to work
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

# Verify panel structure
unique_ids <- id_order  # from the original pipeline
n_cells    <- length(unique_ids)
years      <- sort(unique(cell_data$year))
n_years    <- length(years)
stopifnot(nrow(cell_data) == n_cells * n_years)  # balanced panel check

# Build integer cell-position map
id_to_pos <- setNames(seq_len(n_cells), as.character(unique_ids))

# Verify cell_data$id ordering matches id_order positions
cell_positions <- id_to_pos[as.character(cell_data$id)]
year_offsets   <- match(cell_data$year, years)
expected_rows  <- (cell_positions - 1L) * n_years + year_offsets
stopifnot(all(expected_rows == seq_len(nrow(cell_data))))

# --- Build edge list (one-time cost, ~seconds) ---
n_nbrs    <- lengths(rook_neighbors_unique)
from_cell <- rep(seq_len(n_cells), times = n_nbrs)
to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)
n_edges   <- length(from_cell)

# Expand across years: integer arithmetic, no strings
from_cell_exp <- rep(from_cell, each = n_years)
to_cell_exp   <- rep(to_cell,   each = n_years)
year_offset   <- rep(seq_len(n_years), times = n_edges)

edge_from <- (from_cell_exp - 1L) * n_years + year_offset
edge_to   <- (to_cell_exp   - 1L) * n_years + year_offset

# Sort by from for grouped operations (data.table will also sort internally,
# but pre-sorting helps memory locality)
ord       <- order(edge_from)
edge_from <- edge_from[ord]
edge_to   <- edge_to[ord]

rm(from_cell_exp, to_cell_exp, year_offset, ord)
gc()

# --- Compute neighbor features for all 5 variables ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  
  stats <- compute_neighbor_stats_fast(
    cell_data, edge_from, edge_to, nrow(cell_data), var_name
  )
  
  cell_data[[paste0("nb_max_",  var_name)]] <- stats[, 1]
  cell_data[[paste0("nb_min_",  var_name)]] <- stats[, 2]
  cell_data[[paste0("nb_mean_", var_name)]] <- stats[, 3]
  
  rm(stats)
  gc()
}

# The trained Random Forest model is untouched.
# The numerical outputs (nb_max, nb_min, nb_mean per variable) are identical
# to the original pipeline â€” same integer indexing, same max/min/mean calculations,
# just computed via vectorized grouped aggregation instead of per-row lapply.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup()`: 6.46M Ã— ~4 string pastes + string-keyed hash lookups | Eliminated entirely |
| **Lookup method** | `paste(id, year)` â†’ named character vector probe | Integer arithmetic: `(cell_pos - 1) * n_years + year_offset` |
| **Neighbor stats** | Per-row `lapply` over 6.46M rows | `data.table` grouped aggregation on edge list |
| **Estimated runtime** | 86+ hours | **~2â€“10 minutes** (edge list construction + 5 grouped aggregations) |
| **Memory** | 6.46M-element list of integer vectors | Two integer vectors of length ~38.4M (edge_from, edge_to) â‰ˆ ~307 MB |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical output** | Original estimand | Identical âœ“ |

The key insight is that the panel data has a **regular grid structure** (every cell appears in every year), which means row positions can be computed by integer arithmetic in O(1) per lookup â€” completely eliminating the millions of string allocations and hash probes that constituted the actual bottleneck.
END_RESPONSE>>>
