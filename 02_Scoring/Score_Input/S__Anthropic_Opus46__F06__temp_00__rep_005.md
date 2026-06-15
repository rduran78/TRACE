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
run_id: Anthropic_F06_T00_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list with per-row string operations

`build_neighbor_lookup` iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's rook neighbors (fine).
2. Constructs `paste()`-based string keys for every neighbor Ã— that row's year.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

String hashing over 6.46M rows with ~4 neighbors each means **~25 million string constructions and hash lookups**. The resulting `neighbor_lookup` is a list of 6.46M integer vectors â€” large in memory and slow to build.

### Bottleneck B: `compute_neighbor_stats` â€” repeated per variable

`compute_neighbor_stats` iterates over the 6.46M-element list **once per variable**. With 5 variables, that's **~32 million list element accesses**, each extracting a small integer vector, subsetting a column, removing NAs, and computing max/min/mean. The R-level `lapply` loop over millions of tiny vectors is extremely slow.

### Why 86+ hours?

- The `lapply` loops are pure R loops over millions of elements.
- String-keyed lookups (`paste` + named vector indexing) are far slower than integer arithmetic.
- The work is done 5 separate times (once per variable) when it could be vectorized.

### Why not raster focal/kernel operations?

Focal operations assume a regular rectangular grid with a fixed kernel. Here the data is an **irregular spatial panel** with a precomputed `spdep::nb` neighbor object, so cells may have 0â€“4 rook neighbors and the grid may have irregular boundaries. Focal operations would require embedding into a complete raster, handling missing cells, and could introduce subtle numerical differences (e.g., at boundaries). The sparse-matrix approach below is the correct analogy: it is mathematically equivalent to a focal operation but works on arbitrary neighbor structures and **exactly preserves the original numerical results**.

---

## 2. Optimization Strategy

### Key Insight: Express neighbor aggregation as sparse matrix multiplication.

For each cell-year row `i`, the neighbor stats are aggregations over a known set of rows. This is equivalent to a **sparse matrixâ€“vector product**:

- Build a sparse **adjacency matrix** `W` of dimension `(n_rows Ã— n_rows)` where `W[i, j] = 1` if row `j` is a rook neighbor of row `i` in the same year.
- `neighbor_max`, `neighbor_min`, `neighbor_mean` can then be computed using vectorized sparse operations.

**Specifically:**

| Statistic | Vectorized computation |
|---|---|
| **mean** | `W %*% x / neighbor_count` (sparse matrixâ€“vector multiply) |
| **max** | Use the sparse structure to compute group-wise max (via `data.table` group-by on the COO representation) |
| **min** | Same as max but with `min` |

### Steps:

1. **Replace string keys with integer arithmetic**: Encode `(cell_id, year)` â†’ row index via a two-column integer join (using `data.table`), not string pasting.
2. **Build a sparse adjacency matrix once** (COO â†’ `dgCMatrix`), ~25M non-zeros.
3. **Compute all 3 stats Ã— 5 variables in vectorized operations** â€” no R-level row loop.
4. **Estimated speedup**: from 86+ hours to **~2â€“5 minutes**.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# Preserves the exact numerical results of the original implementation.
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # ------------------------------------------------------------------
  # STEP 0: Convert to data.table for fast indexed operations
  # ------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]  # preserve original row order

  # ------------------------------------------------------------------
  # STEP 1: Build integer-keyed mapping from (id, year) -> row index
  # ------------------------------------------------------------------
  # Create a keyed lookup table
  lookup <- dt[, .(id, year, .row_idx)]
  setkey(lookup, id, year)

  n_rows <- nrow(dt)

  # ------------------------------------------------------------------
  # STEP 2: Build sparse adjacency COO (from_row, to_row) in one
  #         vectorized pass â€” no per-row R loop
  # ------------------------------------------------------------------
  cat("Building neighbor edge list...\n")

  # Map cell id -> position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Expand the nb object into an edge list at the cell level
  # Each element of rook_neighbors_unique is an integer vector of
  # neighbor *positions* in id_order (standard spdep::nb format).
  nb_from <- rep(seq_along(rook_neighbors_unique),
                 lengths(rook_neighbors_unique))
  nb_to   <- unlist(rook_neighbors_unique)

  # Remove the 0-neighbor entries (spdep uses integer(0) or 0L sentinel)
  valid <- nb_to > 0L
  nb_from <- nb_from[valid]
  nb_to   <- nb_to[valid]

  # Convert positions back to actual cell IDs
  cell_from <- id_order[nb_from]
  cell_to   <- id_order[nb_to]

  # Create a data.table of directed cell-level edges
  edges_cell <- data.table(id_from = cell_from, id_to = cell_to)

  # Now cross-join with years present in the data to get row-level edges.
  # Instead of a full cross join (expensive), we join through the lookup.
  years <- sort(unique(dt$year))

  cat("Expanding edges across", length(years), "years...\n")

  # Replicate edges for every year
  edges_panel <- edges_cell[, .(id_from, id_to, year = rep(list(years), .N))]
  # More memory-efficient: use CJ-like expansion
  edges_panel <- edges_cell[, .(year = years), by = .(id_from, id_to)]

  # Join to get row indices for "from" rows
  setkey(edges_panel, id_from, year)
  edges_panel[lookup, row_from := i..row_idx, on = .(id_from = id, year)]

  # Join to get row indices for "to" (neighbor) rows
  setkey(edges_panel, id_to, year)
  edges_panel[lookup, row_to := i..row_idx, on = .(id_to = id, year)]

  # Drop edges where either side is missing (cell not observed that year)
  edges_panel <- edges_panel[!is.na(row_from) & !is.na(row_to)]

  cat("Total directed row-level edges:", nrow(edges_panel), "\n")

  # ------------------------------------------------------------------
  # STEP 3: Compute neighbor stats using vectorized group-by
  # ------------------------------------------------------------------
  # We group by row_from and aggregate the neighbor values.
  # This avoids building a 6.46M Ã— 6.46M sparse matrix entirely;
  # instead we work directly on the COO edge list with data.table.

  from_idx <- edges_panel$row_from
  to_idx   <- edges_panel$row_to

  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "\n")

    vals <- dt[[var_name]]

    # Get neighbor values aligned with the edge list
    neighbor_vals <- vals[to_idx]

    # Build a temporary data.table for grouped aggregation
    agg_dt <- data.table(row_from = from_idx, nval = neighbor_vals)

    # Remove edges where the neighbor value is NA
    agg_dt <- agg_dt[!is.na(nval)]

    # Compute max, min, mean grouped by row_from
    stats <- agg_dt[, .(nb_max  = max(nval),
                         nb_min  = min(nval),
                         nb_mean = mean(nval)),
                     by = row_from]

    # Initialize result columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign computed values back to the correct rows
    set(dt, i = stats$row_from, j = max_col,  value = stats$nb_max)
    set(dt, i = stats$row_from, j = min_col,  value = stats$nb_min)
    set(dt, i = stats$row_from, j = mean_col, value = stats$nb_mean)
  }

  # ------------------------------------------------------------------
  # STEP 4: Clean up and return as original class
  # ------------------------------------------------------------------
  dt[, .row_idx := NULL]

  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Property | Original | Optimized | Match? |
|---|---|---|---|
| Neighbor definition | Rook contiguity from `spdep::nb` | Same `spdep::nb` object, same directed edges | âœ… |
| Year matching | `paste(id, year)` string join | Integer join on `(id, year)` â€” identical semantics | âœ… |
| NA handling | `neighbor_vals[!is.na()]` then aggregate | `agg_dt[!is.na(nval)]` then aggregate | âœ… |
| `max` / `min` / `mean` | Base R `max`, `min`, `mean` | `data.table` `max`, `min`, `mean` â€” identical IEEE 754 results | âœ… |
| No-neighbor rows | Return `c(NA, NA, NA)` | Columns initialized to `NA_real_`, only overwritten if edges exist | âœ… |
| RF model | Not retrained | Not retrained | âœ… |

---

## 5. Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| Build lookup | ~30â€“40 min (string ops) | ~30 sec (integer join) |
| Compute stats (Ã—5 vars) | ~80+ hours (R `lapply` Ã—6.46M Ã—5) | ~2â€“3 min (`data.table` grouped agg) |
| Peak RAM | ~8â€“12 GB (6.46M-element list) | ~4â€“6 GB (edge list + columns) |
| **Total** | **~86+ hours** | **~3â€“5 minutes** |

The key transformation is replacing **6.46 million R-level list iterations per variable** with a single **vectorized `data.table` grouped aggregation** over the edge list â€” a speedup of roughly **1,000Ã—**.
END_RESPONSE>>>
