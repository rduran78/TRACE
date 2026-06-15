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
run_id: Anthropic_F06_T03_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`
This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs string keys by pasting `neighbor_id` and `year`.
4. Looks up row indices via a named character vector (`idx_lookup`).

The string pasting and named-vector lookup for every single row is extremely expensive. With ~6.46M rows, each doing multiple `paste` + hash lookups, this alone can take many hours.

### Bottleneck 2: `compute_neighbor_stats` â€” Row-level `lapply` with R-level loops
For each of the ~6.46M rows, it subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. The `lapply` over 6.46M elements, repeated for 5 variables (so ~32.3M iterations total), is very slow in pure R.

### Why raster focal/kernel operations are **not** a direct replacement
Focal operations assume a regular rectangular grid with a fixed kernel. Here, the grid cells have an irregular neighbor structure (stored as an `nb` object â€” some cells have 2, 3, or 4 rook neighbors depending on boundaries and missing cells), and the data is in long panel format (cell Ã— year). Focal operations would require reshaping each variable into a complete raster for each year, running the focal, then extracting back â€” and would silently produce wrong results at boundaries or for missing cells. The `nb`-based approach is correct and must be preserved.

### Summary
| Component | Current Cost | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~40+ hours | 6.46M string pastes + named vector lookups |
| `compute_neighbor_stats` | ~40+ hours | 6.46M R-level lapply iterations Ã— 5 vars |
| **Total** | **~86+ hours** | Pure-R row-level iteration |

---

## Optimization Strategy

### Strategy 1: Vectorize `build_neighbor_lookup` entirely
Instead of building a per-row list of neighbor row indices, construct a **sparse adjacency structure as two integer vectors** (a "from-row" and "to-row" edge list) using fully vectorized operations. This avoids all per-row string operations.

**Key insight**: Since every cell in a given year has the same neighbors (just in that year's rows), we can:
1. Build a cell-level edge list from the `nb` object (done once, ~1.37M edges).
2. Cross this with all 28 years to get a **row-level edge list** (~1.37M Ã— 28 â‰ˆ 38.5M edges).
3. Use `data.table` joins (hash-based, vectorized in C) to map `(cell_id, year)` â†’ row index.

### Strategy 2: Vectorized grouped aggregation for neighbor stats
Given the row-level edge list `(from_row, to_row)`, computing neighbor stats becomes a **grouped aggregation**:
- For each `from_row`, aggregate `variable[to_row]` by `max`, `min`, `mean`.
- This is exactly what `data.table` does at C speed with `by=` grouping.

### Complexity comparison
| | Current | Optimized |
|---|---|---|
| Lookup build | O(6.46M) R-level iterations | O(38.5M) vectorized join |
| Stats (per var) | O(6.46M) R-level iterations | O(38.5M) data.table group-by |
| **Total wall time** | ~86 hours | **~2â€“5 minutes** |

### Preserving correctness
- The edge list is derived from the same `nb` object.
- `max`, `min`, `mean` are computed over exactly the same neighbor sets.
- Rows with no neighbors get `NA` (same as original).
- The trained Random Forest model is untouched; we only change how predictor columns are computed.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 1: Build a cell-level edge list from the nb object
# =============================================================================
build_cell_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector of cell IDs corresponding to each nb element
  
  from_list <- vector("list", length(neighbors))
  to_list   <- vector("list", length(neighbors))
  
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(nb_i) == 1L && nb_i == 0L) next
    from_list[[i]] <- rep(id_order[i], length(nb_i))
    to_list[[i]]   <- id_order[nb_i]
  }
  
  data.table(
    from_id = unlist(from_list, use.names = FALSE),
    to_id   = unlist(to_list,   use.names = FALSE)
  )
}

# =============================================================================
# STEP 2: Expand to row-level edge list via vectorized join
# =============================================================================
build_row_edge_list <- function(cell_dt, cell_edges) {
  # cell_dt must have columns: .row_idx, id, year
  # cell_edges has columns: from_id, to_id
  
  # Get unique years
  years <- sort(unique(cell_dt$year))
  
  # Cross cell edges with all years
  row_edges <- cell_edges[, .(year = years), by = .(from_id, to_id)]
  
  # Create lookup: (id, year) -> row index
  setkey(cell_dt, id, year)
  
  # Map from_id,year -> from_row
  row_edges[cell_dt, from_row := i..row_idx, on = .(from_id = id, year = year)]
  
  # Map to_id,year -> to_row
  row_edges[cell_dt, to_row := i..row_idx, on = .(to_id = id, year = year)]
  
  # Drop edges where either side is missing (cell not observed that year)
  row_edges <- row_edges[!is.na(from_row) & !is.na(to_row)]
  
  row_edges[, .(from_row, to_row)]
}

# =============================================================================
# STEP 3: Compute neighbor stats via grouped aggregation
# =============================================================================
compute_neighbor_stats_fast <- function(cell_dt, row_edges, var_name) {
  # Extract the variable values for the "to" side of each edge
  vals <- cell_dt[[var_name]]
  
  edge_vals <- data.table(
    from_row = row_edges$from_row,
    val      = vals[row_edges$to_row]
  )
  
  # Remove edges where the neighbor value is NA
  edge_vals <- edge_vals[!is.na(val)]
  
  # Grouped aggregation
  stats <- edge_vals[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = from_row]
  
  # Initialize output columns with NA
  n <- nrow(cell_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)
  
  # Fill in computed values
  out_max[stats$from_row]  <- stats$nb_max
  out_min[stats$from_row]  <- stats$nb_min
  out_mean[stats$from_row] <- stats$nb_mean
  
  # Name columns to match expected RF predictor names
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  set(cell_dt, j = col_max,  value = out_max)
  set(cell_dt, j = col_min,  value = out_min)
  set(cell_dt, j = col_mean, value = out_mean)
  
  invisible(cell_dt)
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

# Convert to data.table if not already (non-destructive copy)
cell_dt <- as.data.table(cell_data)

# Add row index column
cell_dt[, .row_idx := .I]

# Step 1: Cell-level edge list from nb object (~1.37M edges, < 1 second)
cat("Building cell-level edge list...\n")
cell_edges <- build_cell_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Cell-level edges: %s\n", format(nrow(cell_edges), big.mark = ",")))

# Step 2: Expand to row-level edge list (~38.5M edges, ~10-30 seconds)
cat("Building row-level edge list...\n")
row_edges <- build_row_edge_list(cell_dt, cell_edges)
cat(sprintf("  Row-level edges: %s\n", format(nrow(row_edges), big.mark = ",")))

# Free the cell-level edges
rm(cell_edges)
gc()

# Step 3: Compute neighbor features for all 5 variables (~30-60 sec each)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
  t0 <- proc.time()
  compute_neighbor_stats_fast(cell_dt, row_edges, var_name)
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("  Done in %.1f seconds.\n", elapsed))
}

# Free the edge list
rm(row_edges)
gc()

# Remove helper column
cell_dt[, .row_idx := NULL]

# Convert back to data.frame if the RF predict method requires it
cell_data <- as.data.frame(cell_dt)

cat("All neighbor features computed. Ready for prediction.\n")

# =============================================================================
# PREDICTION (unchanged â€” the trained model is preserved as-is)
# =============================================================================
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Is Correct and Equivalent

| Property | Original | Optimized |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[i]]` | Same `nb` object â†’ same edge list |
| Aggregation functions | `max`, `min`, `mean` (NA-removed) | Identical functions, NA-removed before grouping |
| No-neighbor rows | Return `c(NA, NA, NA)` | Rows absent from `stats` â†’ filled with `NA` |
| Column names | `{var}_neighbor_max/min/mean` | Identical naming convention |
| RF model | Not retouched | Not retouched |
| Numerical estimand | Preserved exactly | Floating-point identical results |

**Expected runtime: ~2â€“5 minutes total** (vs. 86+ hours), well within 16 GB RAM on a standard laptop.
END_RESPONSE>>>
