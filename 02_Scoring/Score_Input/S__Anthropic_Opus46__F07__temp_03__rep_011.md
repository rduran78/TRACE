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
run_id: Anthropic_F07_T03_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, where each element is built by:

1. **Character-key hashing per row** â€” `paste(neighbor_cell_ids, data$year[i], sep = "_")` and a named-vector lookup (`idx_lookup[neighbor_keys]`) are called **once per row**. With ~6.46M rows, this creates and indexes millions of temporary character vectors.
2. **`lapply` over 6.46M rows in pure R** â€” Each iteration does multiple allocations (character coercion, paste, subsetting a named vector, NA filtering). The per-iteration overhead is small, but multiplied by 6.46M it becomes catastrophic (~86+ hours).
3. **Redundant recomputation across years** â€” The neighbor *topology* is time-invariant (rook neighbors don't change year to year), yet the lookup is rebuilt from scratch for every cell-year combination by string-matching year-stamped keys.
4. **`compute_neighbor_stats`** also uses `lapply` over 6.46M elements, but each iteration is cheap arithmetic. Still, it is called 5 times (once per variable), adding up.

**Root cause summary:** The algorithm is O(N_rows Ã— avg_neighbors) in *interpreted R with per-element string operations*, where N_rows â‰ˆ 6.46M. This should be a vectorized matrix/data.table operation that finishes in seconds-to-minutes.

---

## Optimization Strategy

1. **Separate topology from time.** The neighbor graph is over 344,208 cells and is year-invariant. Represent it as an edge list once (two integer columns: `from_cell_idx`, `to_cell_idx`).

2. **Lay out data so that all years for one cell are contiguous and in the same order.** Sort `cell_data` by `(id, year)`. Then cell *i*'s data for year *t* is at a predictable row offset. This eliminates all string-key lookups.

3. **Expand the edge list to cell-year edges using vectorized integer arithmetic.** If cell `c` has neighbor `c'`, then for every year index `t` in `1:28`, row `(c-1)*28 + t` is neighbors with row `(c'-1)*28 + t`. This is a single vectorized operation producing ~1.37M Ã— 28 â‰ˆ 38.5M edge pairs â€” large but manageable.

4. **Use `data.table` grouped aggregation on the edge list** to compute `max`, `min`, `mean` of neighbor values in one vectorized pass per variable. This replaces both `build_neighbor_lookup` and `compute_neighbor_stats`.

5. **Result:** No `lapply` over 6.46M rows, no character key construction, no named-vector lookups. Expected runtime: **a few minutes** on a 16 GB laptop.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Inputs assumed already in memory:
#       cell_data            â€” data.frame / data.table with columns id, year, ntl, ec, â€¦
#       id_order             â€” integer vector of cell IDs in the order used by spdep::nb
#       rook_neighbors_unique â€” spdep nb object (list of integer index vectors)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table if needed
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Sort cell_data by (id, year) and build a fast row-index scheme
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Preserve original row order so we can put results back
cell_data[, orig_row_idx__ := .I]

# Create a canonical ordering: cells in id_order order, years ascending
# Map each id to its position in id_order (1-based "cell index")
id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, cell_idx__ := id_to_cellidx[as.character(id)]]

# Sort by cell_idx__ then year
setorder(cell_data, cell_idx__, year)

# After sorting, record the new row position
cell_data[, sorted_row__ := .I]

# Determine the unique sorted years and number of years
years_vec  <- sort(unique(cell_data$year))
n_years    <- length(years_vec)
year_to_t  <- setNames(seq_along(years_vec), as.character(years_vec))
cell_data[, t_idx__ := year_to_t[as.character(year)]]

# Verify layout: row for (cell_idx c, time index t) should be (c-1)*n_years + t
# This holds exactly when every cell has every year. Check:
n_cells <- length(id_order)
stopifnot(nrow(cell_data) == n_cells * n_years)  
# If this fails, fall back to the merge-based approach below (Section 1b).

# Confirm the layout is as expected (spot check)
stopifnot(all(cell_data$sorted_row__ == (cell_data$cell_idx__ - 1L) * n_years + cell_data$t_idx__))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Build the spatial edge list (topology only, year-invariant)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique[[c]] gives integer indices of neighbors of cell c
# (indices into id_order). We need directed edges: from c -> each neighbor.

edge_from <- rep(seq_along(rook_neighbors_unique),
                 lengths(rook_neighbors_unique))
edge_to   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove any 0-length or self-neighbor artifacts
valid <- edge_to > 0L & edge_to <= n_cells & edge_from != edge_to
edge_from <- edge_from[valid]
edge_to   <- edge_to[valid]

n_edges_spatial <- length(edge_from)
cat(sprintf("Spatial directed edges: %d\n", n_edges_spatial))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Expand to cell-year edges using vectorized integer arithmetic
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Row of (cell c, year-index t) in sorted cell_data = (c - 1) * n_years + t
# We replicate each spatial edge across all n_years time slices.

t_offsets <- seq_len(n_years)  # 1 .. 28

# Outer expansion: each spatial edge Ã— each year
# Result vectors of length n_edges_spatial * n_years
focal_rows    <- rep((edge_from - 1L) * n_years, each = n_years) +
                 rep(t_offsets, times = n_edges_spatial)
neighbor_rows <- rep((edge_to   - 1L) * n_years, each = n_years) +
                 rep(t_offsets, times = n_edges_spatial)

cat(sprintf("Cell-year directed edges: %d\n", length(focal_rows)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Compute neighbor stats per variable using data.table aggregation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Build the edge data.table once (just integer row pointers)
edges_dt <- data.table(focal_row = focal_rows, neighbor_row = neighbor_rows)

# Free the large intermediate vectors
rm(focal_rows, neighbor_rows, edge_from, edge_to, t_offsets, valid)
gc()

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  # Extract the variable values in sorted order
  vals <- cell_data[[var_name]]
  
  # Attach neighbor values to edge table
  edges_dt[, nval := vals[neighbor_row]]
  
  # Aggregate: for each focal row, compute max/min/mean of neighbor values
  agg <- edges_dt[!is.na(nval),
                  .(nb_max  = max(nval),
                    nb_min  = min(nval),
                    nb_mean = mean(nval)),
                  keyby = focal_row]
  
  # Initialize result columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  # Place aggregated values into the correct rows
  cell_data[agg$focal_row, (max_col)  := agg$nb_max]
  cell_data[agg$focal_row, (min_col)  := agg$nb_min]
  cell_data[agg$focal_row, (mean_col) := agg$nb_mean]
  
  rm(agg)
  gc()
}

# Clean up the temporary nval column from edges_dt
edges_dt[, nval := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Restore original row order
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setorder(cell_data, orig_row_idx__)

# Drop helper columns
cell_data[, c("orig_row_idx__", "cell_idx__", "sorted_row__", "t_idx__") := NULL]

cat("Done. Neighbor features added.\n")
```

---

### Fallback: Unbalanced Panel (Section 1b)

If the panel is **not perfectly balanced** (some cells missing some years), the `stopifnot` will fail. Replace Sections 1 and 3 with a merge-based approach:

```r
# â”€â”€ 1b. Build row lookup via merge instead of arithmetic â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, row_id__ := .I]  # preserve current order

# Spatial edge list (same as Section 2 above)
edges_spatial <- data.table(from_cell = edge_from, to_cell = edge_to)

# Cross with years present in data
cell_year_map <- cell_data[, .(row_id__, cell_idx__, year)]

# Merge: for each spatial edge, join on year to get focal and neighbor rows
edges_full <- edges_spatial[, .(from_cell, to_cell)]
# Expand by year via cross join with unique years
edges_full <- edges_full[, CJ(edge_id = .I, year = years_vec)
                         ][edges_full, on = "edge_id", allow.cartesian = TRUE]

# ... then join to cell_year_map for from_cell and to_cell to get row_id__
# (This is more complex but handles missing cell-years gracefully.)
```

For the common balanced-panel case (344,208 Ã— 28 = 9,637,824 â€” note: the problem states ~6.46M rows, so verify), the arithmetic approach in the main code is exact and fastest.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows, string paste + named-vector lookup | Eliminated entirely; replaced by integer arithmetic on sorted layout |
| `compute_neighbor_stats` | `lapply` over 6.46M rows Ã— 5 vars | `data.table` grouped aggregation on edge list Ã— 5 vars |
| Key data structure | List of 6.46M integer vectors | Single `data.table` with ~38.5M rows (two integer columns) |
| Estimated runtime | 86+ hours | **5â€“15 minutes** (dominated by the 5 `data.table` group-by aggregations over ~38.5M edges) |
| RAM peak | Moderate (list overhead) | ~2â€“3 GB for edge table + variable column |
| Numerical result | max, min, mean of rook-neighbor values | **Identical** â€” same neighbor sets, same aggregation functions |
| Trained RF model | Untouched | **Untouched** â€” only feature engineering is changed |
END_RESPONSE>>>
