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
run_id: Anthropic_F10_T03_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query on large vectors, and `paste()`-based key construction for 6.46M rows Ã— ~4 neighbors each is extremely expensive. This single step likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting subsets of a numeric vector. While each individual operation is small, the R-level loop overhead across 6.46M iterations, repeated for 5 variables (Ã—3 stats = 15 new columns), compounds severely.

3. **The neighbor topology is year-invariant but the lookup is rebuilt as if it were year-dependent.** Every cell has the same rook neighbors every year. The code re-resolves neighbor row indices per cell-year, even though the spatial adjacency graph is static. This means the topology work is repeated 28Ã— implicitly.

**Root cause summary:** The design conflates the spatial graph (344K nodes, ~1.37M edges) with the panel structure (28 years). The lookup should be built once on the 344K-cell spatial graph and then projected across years via vectorized integer arithmetic, not string matching.

## Optimization Strategy

1. **Build a sparse adjacency structure once** over the 344K cells using a two-column integer edge list (from, to). This is O(E) where E â‰ˆ 1.37M.

2. **Exploit the panel's regular structure.** If data is sorted by `(id, year)` â€” or `(year, id)` â€” then the row index of any `(cell_i, year_t)` can be computed by arithmetic: `offset[cell_i] + (year_t - min_year)`. No string keys needed.

3. **Vectorize the aggregation using sparse matrix multiplication.** Construct a sparse `NÃ—N` adjacency matrix `A` (where N = 344,208). For each year, extract the variable column as a vector over cells, then use `A` to compute neighbor sums and neighbor counts in one matrix-vector multiply. Max and min require a grouped operation, but can be done efficiently with `data.table` or a compiled C++ snippet via `Rcpp`.

4. **For max/min:** Use `data.table` joins on the integer edge list â€” expand edges, join variable values, and group-aggregate. This is highly optimized internally in `data.table` (radix-based, in-place).

5. **Process all 28 years in a vectorized batch** per variable, or loop over 28 years (not 6.46M rows).

**Expected speedup:** From ~86 hours to **minutes** (the bottleneck becomes 5 variables Ã— 28 years Ã— ~1.37M edge expansions with `data.table` group-by, which is trivial).

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats output.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 0: Ensure cell_data is a data.table sorted by (id, year) ---------
cell_dt <- as.data.table(cell_data)
setkeyv(cell_dt, c("id", "year"))

# Unique cell IDs in sorted order and unique years
unique_ids   <- sort(unique(cell_dt$id))
unique_years <- sort(unique(cell_dt$year))
n_cells      <- length(unique_ids)
n_years      <- length(unique_years)

stopifnot(nrow(cell_dt) == n_cells * n_years)  # balanced panel check

# Map cell id -> integer index 1..n_cells
id_to_idx <- setNames(seq_along(unique_ids), as.character(unique_ids))

# Map year -> integer index 1..n_years
year_to_idx <- setNames(seq_along(unique_years), as.character(unique_years))

# Assign a sequential row locator: for sorted (id, year), row of (cell i, year t)
# = (i-1)*n_years + t   where i = id_to_idx[id], t = year_to_idx[year]
# Verify this matches the actual row order:
cell_dt[, row_check := (id_to_idx[as.character(id)] - 1L) * n_years +
                         year_to_idx[as.character(year)]]
stopifnot(all(cell_dt$row_check == seq_len(nrow(cell_dt))))
cell_dt[, row_check := NULL]

# ---- Step 1: Build edge list from rook_neighbors_unique (spdep nb object) --
# rook_neighbors_unique is a list of length n_cells; element [[i]] contains
# integer indices (into id_order) of neighbors of cell i.
# id_order is the vector of cell IDs in the order matching the nb object.

# Map id_order positions to our sorted unique_ids positions
id_order_to_sorted <- id_to_idx[as.character(id_order)]

edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(data.table(from = integer(0), to = integer(0)))
  }
  data.table(
    from = id_order_to_sorted[i],
    to   = id_order_to_sorted[nb]
  )
}))

# Remove any NA edges (boundary cells whose neighbors don't exist)
edges <- edges[!is.na(from) & !is.na(to)]
setkey(edges, from)

cat(sprintf("Edge list: %d directed edges over %d cells\n", nrow(edges), n_cells))

# ---- Step 2: Function to compute neighbor max, min, mean for one variable ---
#
# For each cell i and year t, we need max/min/mean of variable values at
# neighbors of i in year t.
#
# Strategy: loop over 28 years (not 6.46M rows). For each year, extract the
# variable vector (length n_cells), join onto edge list, and group-aggregate.

add_neighbor_features <- function(dt, var_name, edges, n_cells, n_years,
                                  unique_years, year_to_idx) {
  # Pre-extract the full variable column as a matrix: n_cells x n_years
  # Row i, col t = value for cell i in year t
  # Because dt is keyed by (id, year), values are laid out as:

  #   cell1-year1, cell1-year2, ..., cell1-yearT, cell2-year1, ...
  vals_vec <- dt[[var_name]]
  # Reshape to matrix: rows=cells, cols=years (byrow=FALSE reads column-major,
  # which matches our layout since consecutive rows = consecutive years for same cell)
  vals_mat <- matrix(vals_vec, nrow = n_years, ncol = n_cells)
  # vals_mat[t, i] = value for cell i in year t
  # (R fills matrices column-major: first n_years entries -> column 1 = cell 1)

  # Prepare output matrices
  max_mat  <- matrix(NA_real_, nrow = n_years, ncol = n_cells)
  min_mat  <- matrix(NA_real_, nrow = n_years, ncol = n_cells)
  mean_mat <- matrix(NA_real_, nrow = n_years, ncol = n_cells)

  # edges$from, edges$to are integer cell indices
  e_from <- edges$from
  e_to   <- edges$to

  for (t in seq_len(n_years)) {
    # Neighbor values: for each edge (from -> to), get value at 'to' in year t
    nb_vals <- vals_mat[t, e_to]

    # Build a data.table for fast grouped aggregation
    agg_dt <- data.table(from = e_from, val = nb_vals)

    # Remove NAs before aggregation (matches original: neighbor_vals[!is.na()])
    agg_dt <- agg_dt[!is.na(val)]

    if (nrow(agg_dt) > 0L) {
      stats <- agg_dt[, .(nb_max  = max(val),
                           nb_min  = min(val),
                           nb_mean = mean(val)),
                       by = from]

      max_mat[t,  stats$from] <- stats$nb_max
      min_mat[t,  stats$from] <- stats$nb_min
      mean_mat[t, stats$from] <- stats$nb_mean
    }
  }

  # Flatten back to vector (column-major matches our row layout)
  max_col  <- paste0("max_",  var_name)
  min_col  <- paste0("min_",  var_name)
  mean_col <- paste0("mean_", var_name)

  dt[, (max_col)  := as.vector(max_mat)]
  dt[, (min_col)  := as.vector(min_mat)]
  dt[, (mean_col) := as.vector(mean_mat)]

  invisible(dt)
}

# ---- Step 3: Run for all 5 neighbor source variables ------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s ...\n", var_name))
  t0 <- proc.time()
  add_neighbor_features(cell_dt, var_name, edges, n_cells, n_years,
                        unique_years, year_to_idx)
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("  Done in %.1f seconds\n", elapsed))
}

# ---- Step 4: Convert back to data.frame if the RF model expects one ---------
cell_data <- as.data.frame(cell_dt)

# ---- Step 5: Apply the pre-trained Random Forest (unchanged) ----------------
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` extracts neighbor values for cell `i` in year `t` | `vals_mat[t, e_to]` extracts the same values via the same edge pairs | Same values selected |
| `neighbor_vals[!is.na(neighbor_vals)]` | `agg_dt[!is.na(val)]` | Same NA removal |
| `max()`, `min()`, `mean()` on filtered vector | `max(val)`, `min(val)`, `mean(val)` in `data.table` group-by | Identical R functions, identical floating-point results |
| Cells with no valid neighbors â†’ `c(NA, NA, NA)` | Matrix initialized to `NA_real_`; only cells appearing in `stats$from` are overwritten | Same NA pattern |

## Performance Analysis

| Component | Original | Optimized |
|---|---|---|
| Lookup construction | ~6.46M string-paste + named-vector lookups â†’ **hours** | 1.37M-row edge list built once â†’ **seconds** |
| Per-variable aggregation | 6.46M R-level `lapply` iterations Ã— 5 vars â†’ **hours** | 28 `data.table` group-by operations on 1.37M rows Ã— 5 vars = 140 calls â†’ **minutes** |
| Memory | 6.46M-element list of integer vectors (~large) | One 1.37M Ã— 2 integer edge table + temporary matrices (~modest) |
| **Total estimated time** | **86+ hours** | **5â€“15 minutes on 16 GB laptop** |

The key insight: the spatial graph has **344K nodes and 1.37M edges** â€” it's the panel expansion to 6.46M row-level list entries with string-key lookups that created the bottleneck. By separating the static topology (edge list) from the temporal dimension (28-iteration loop), we reduce the problem by a factor of ~4,700Ã— in loop iterations and eliminate all string operations entirely.
END_RESPONSE>>>
