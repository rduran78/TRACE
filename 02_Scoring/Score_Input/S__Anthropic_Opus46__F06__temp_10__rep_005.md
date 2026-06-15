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
run_id: Anthropic_F06_T10_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) list construction
`build_neighbor_lookup` calls `lapply` over **~6.46 million rows**. For each row, it:
- Looks up the cell's reference index, finds its rook neighbors, constructs string keys (`paste(id, year)`), and indexes into a named lookup vector.
- Named vector lookups (`idx_lookup[neighbor_keys]`) in R are hash-table operations, but doing this **6.46 million times** with character key construction and subsetting is extremely slow.
- The function produces a **list of 6.46 million integer vectors** â€” a massive memory and iteration cost.

### Bottleneck 2: `compute_neighbor_stats` â€” Repeated per-row `lapply`
`compute_neighbor_stats` iterates over the 6.46M-element lookup list **once per variable** (Ã—5 variables = ~32.3 million list traversals). Each iteration extracts values, removes NAs, and computes max/min/mean. The R-level loop overhead is enormous.

### Why raster focal/kernel operations don't directly apply
Raster focal operations assume a regular grid with a fixed rectangular kernel and a single time slice. This panel dataset has:
- An irregular subset of grid cells (not all cells in a bounding rectangle are present).
- A temporal dimension (year) that means neighbors must be matched within the same year.
- A precomputed `nb` object (not a simple rectangular window).

However, **the analogy is useful**: focal operations succeed because they are vectorized matrix operations. We can achieve the same by converting the neighbor structure into a **sparse matrix** and using matrix multiplication/operations.

### Summary
| Component | Current Cost | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~hours | 6.46M R-level iterations with string ops |
| `compute_neighbor_stats` | ~hours Ã—5 vars | 6.46M R-level iterations Ã—5 variables |
| **Total** | **86+ hours** | Pure-R row-level loops over millions of rows |

---

## Optimization Strategy

### Key Insight: Sparse-Matrix Vectorization (Focal-Analogy)

Since the neighbor relationships are **identical within each year** (the spatial topology doesn't change), we can:

1. **Build a sparse adjacency matrix `W`** of dimension `N_rows Ã— N_rows` (6.46M Ã— 6.46M) where `W[i,j] = 1` if row `j` is a rook neighbor of row `i` **in the same year**. This matrix is extremely sparse (~1,373,394 neighbor pairs Ã— 28 years â‰ˆ 38.5M nonzero entries out of ~41.7 trillion possible â€” sparsity > 99.9999%).

2. **Compute neighbor stats via sparse matrix operations:**
   - **Mean:** `W %*% x / (W %*% ones)` (sum of neighbor values / count of neighbors)
   - **Max and Min:** Use grouped operations via `data.table` with the edge list, avoiding per-row R loops.

3. **Process all 5 variables in one pass** over the edge structure.

This replaces ~32.3 million R-level loop iterations with vectorized C-level sparse matrix and `data.table` operations.

### Expected Speedup
- `build_neighbor_lookup`: eliminated (replaced by sparse matrix construction, ~seconds).
- `compute_neighbor_stats`: replaced by sparse matrix multiply (mean) and `data.table` grouped aggregation (max, min) â€” ~seconds to low minutes.
- **Overall: from 86+ hours â†’ minutes.**

### Preservation Guarantees
- The **trained Random Forest model** is untouched â€” we only change feature engineering.
- The **numerical results** (max, min, mean of rook neighbors) are identical â€” same arithmetic, different execution strategy.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# =============================================================================
# Requirements: data.table, Matrix
# install.packages(c("data.table", "Matrix"))  # if needed

library(data.table)
library(Matrix)

#' Build neighbor features for all source variables using sparse matrix ops.
#'
#' @param cell_data    data.frame with columns: id, year, and all source vars
#' @param id_order     character/integer vector: the cell IDs in the order
#'                     corresponding to the nb object
#' @param nb_obj       spdep nb object (rook_neighbors_unique)
#' @param source_vars  character vector of variable names to compute stats for
#' @return cell_data with new columns: {var}_nb_max, {var}_nb_min, {var}_nb_mean
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          nb_obj,
                                          source_vars) {

  # ----------------------------
  # Step 0: Convert to data.table for speed; record original row order
  # ----------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]

  # ----------------------------
  # Step 1: Build a spatial edge list from the nb object (year-agnostic)
  # ----------------------------
  # nb_obj[[i]] contains neighbor indices into id_order for cell id_order[i]
  message("Building spatial edge list from nb object...")
  n_cells <- length(id_order)

  from_idx <- rep(seq_len(n_cells),
                  times = vapply(nb_obj, length, integer(1)))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove any 0-length or self-referencing entries (spdep convention)
  valid <- to_idx > 0L & from_idx != to_idx
  spatial_from <- id_order[from_idx[valid]]
  spatial_to   <- id_order[to_idx[valid]]

  spatial_edges <- data.table(from_id = spatial_from, to_id = spatial_to)
  message(sprintf("  %s directed spatial edges", format(nrow(spatial_edges), big.mark = ",")))

  # ----------------------------
  # Step 2: Create a row-index lookup: (id, year) -> row in dt
  # ----------------------------
  message("Creating row index lookup...")
  dt[, .row_idx := .I]
  lookup <- dt[, .(id, year, .row_idx)]
  setkey(lookup, id, year)

  # ----------------------------
  # Step 3: Expand spatial edges across all years to get row-level edges
  #         This is the "same year" join.
  # ----------------------------
  message("Expanding edges across years...")
  years <- sort(unique(dt$year))

  # Cross join spatial edges Ã— years, then look up row indices
  # Memory-efficient approach: process in chunks by year
  edge_list_parts <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Get row indices for 'from' cells in this year
    from_rows <- lookup[.(spatial_edges$from_id, yr), .row_idx, nomatch = 0L, on = .(id, year)]
    to_rows   <- lookup[.(spatial_edges$to_id,   yr), .row_idx, nomatch = 0L, on = .(id, year)]

    # We need matched pairs: both from and to must exist in this year
    # Do a proper paired lookup:
    year_edges <- copy(spatial_edges)
    year_edges[, yr_val := yr]

    # Merge from side
    setnames(year_edges, "from_id", "id")
    year_edges <- lookup[.(years[yi]), on = .(year)][year_edges, on = .(id), nomatch = 0L]
    setnames(year_edges, c(".row_idx", "id"), c("from_row", "from_id"))
    year_edges[, year := NULL]

    # Merge to side
    setnames(year_edges, "to_id", "id")
    year_edges <- lookup[.(years[yi]), on = .(year)][year_edges, on = .(id), nomatch = 0L]
    setnames(year_edges, c(".row_idx", "id"), c("to_row", "to_id"))
    year_edges[, year := NULL]

    edge_list_parts[[yi]] <- year_edges[, .(from_row, to_row)]
  }

  full_edges <- rbindlist(edge_list_parts)
  rm(edge_list_parts)
  message(sprintf("  %s row-level directed edges", format(nrow(full_edges), big.mark = ",")))

  # ----------------------------
  # Step 4: Compute neighbor stats for each variable
  # ----------------------------
  n_rows <- nrow(dt)

  for (var_name in source_vars) {
    message(sprintf("Computing neighbor stats for '%s'...", var_name))

    vals <- dt[[var_name]]

    # Get neighbor values for every edge
    full_edges[, nb_val := vals[to_row]]

    # Remove edges where the neighbor value is NA
    valid_edges <- full_edges[!is.na(nb_val)]

    # Grouped aggregation: for each 'from_row', compute max, min, mean
    stats <- valid_edges[, .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ), by = from_row]

    # Initialize result columns with NA
    max_col  <- rep(NA_real_, n_rows)
    min_col  <- rep(NA_real_, n_rows)
    mean_col <- rep(NA_real_, n_rows)

    # Fill in computed values
    max_col[stats$from_row]  <- stats$nb_max
    min_col[stats$from_row]  <- stats$nb_min
    mean_col[stats$from_row] <- stats$nb_mean

    # Add to data.table
    set(dt, j = paste0(var_name, "_nb_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_nb_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_nb_mean"), value = mean_col)

    message(sprintf("  Done: %s", var_name))
  }

  # Clean up helper columns
  full_edges[, nb_val := NULL]

  # ----------------------------
  # Step 5: Restore original order and return as data.frame
  # ----------------------------
  setorder(dt, .row_order)
  dt[, c(".row_order", ".row_idx") := NULL]
  lookup <- NULL  # free memory

  message("All neighbor features computed successfully.")
  return(as.data.frame(dt))
}


# =============================================================================
# USAGE â€” Drop-in replacement for the original outer loop
# =============================================================================

# Original code replaced:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# New code:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data    = cell_data,
  id_order     = id_order,
  nb_obj       = rook_neighbors_unique,
  source_vars  = neighbor_source_vars
)

# The trained Random Forest model is used as before â€” no changes needed:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Faster Alternative: Sparse Matrix for Mean (optional further speedup)

If profiling shows the `data.table` grouped aggregation is still slow for `mean`, the sparse-matrix approach is even faster for that specific statistic:

```r
# Build sparse adjacency matrix (n_rows Ã— n_rows) â€” one-time cost
library(Matrix)

W <- sparseMatrix(
  i    = full_edges$from_row,
  j    = full_edges$to_row,
  x    = 1,
  dims = c(n_rows, n_rows)
)

# For a given variable:
x <- dt[["ntl"]]
x[is.na(x)] <- 0  # handle NAs carefully

# ones vector for counting non-NA neighbors
non_na <- as.numeric(!is.na(dt[["ntl"]]))

nb_sum   <- as.numeric(W %*% x)
nb_count <- as.numeric(W %*% non_na)
nb_mean  <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)
```

This runs in seconds via compiled sparse BLAS routines but only gives mean directly. Max and min still require the grouped `data.table` approach (or `{collapse}` package `fmax`/`fmin` with grouping).

---

## Performance Summary

| Step | Original | Optimized | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~hours (6.46M R-loop iterations) | ~1â€“2 min (vectorized edge expansion) | ~100Ã— |
| Neighbor stats (5 vars) | ~hours (32.3M R-loop iterations) | ~2â€“5 min (`data.table` grouped agg) | ~500Ã— |
| **Total** | **86+ hours** | **~5â€“10 minutes** | **~500â€“1000Ã—** |
| Peak RAM | ~10â€“14 GB (6.46M-element list) | ~4â€“6 GB (edge table + dt) | Lower |

The trained Random Forest model is completely preserved, and the numerical outputs (max, min, mean of rook-neighbor values per cell-year) are identical to the original implementation.
END_RESPONSE>>>
