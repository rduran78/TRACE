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
run_id: Anthropic_F12_T00_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of ~6.46 million small 3-element vectors is not free, it is a single vectorized operation that completes in seconds. The `lapply` inside `compute_neighbor_stats()` does no list binding at all â€” it returns a fixed-length vector per iteration.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive character key construction and lookup**: For each of the ~6.46 million rows, it calls `paste()` to construct character keys for every neighbor of that cell-year, then performs named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookup in R is **O(n)** per query via hashing overhead at scale. With ~6.46M rows and an average of ~4 rook neighbors each, that's ~25.8 million `paste` + hash-lookup operations against a named vector of length 6.46M.

2. **Repeated `as.character()` and `paste()` calls inside the `lapply`**: Each of the 6.46M iterations does string coercion and concatenation â€” these are extremely expensive in a tight R loop.

3. **The lookup is rebuilt identically for all 5 variables but used 5 times**: This is fine (it's built once), but the build itself is the wall-clock killer. The `lapply` over 6.46M rows with string operations inside is the 86+ hour bottleneck.

4. **`compute_neighbor_stats()` is comparatively cheap**: It indexes a numeric vector by integer positions and computes `max/min/mean` on small subsets. This is fast.

**In summary**: The bottleneck is the per-row string-key construction and named-vector hash lookup inside `build_neighbor_lookup()`. The fix is to replace all string-keyed lookups with pure integer arithmetic.

---

## Optimization Strategy

1. **Replace string-key lookups with integer arithmetic**: Since every `id` appears for every year (balanced panel: 344,208 cells Ã— 28 years = 9,637,824 â€” but the document says ~6.46M rows, so it may be unbalanced). We build an integer matrix mapping `(cell_index, year) â†’ row_number` and look up neighbors via direct integer indexing â€” no strings, no hashing.

2. **Vectorize `compute_neighbor_stats()`**: Replace the per-row `lapply` with a single grouped operation using `data.table` or pre-allocated matrix fills, computing `max`, `min`, `mean` over neighbor indices in bulk.

3. **Preserve the trained Random Forest model**: We only change feature engineering (the pipeline that produces the same numerical columns). The RF model object is untouched.

4. **Preserve the original numerical estimand**: The computed `max`, `min`, `mean` of neighbor values are identical â€” we just compute them faster.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup using integer arithmetic (no string keys)
# =============================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert data to data.table for fast operations
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # Build integer map: id_to_ref (cell id -> position in id_order)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build a lookup matrix: rows = cell_ref_index, cols = year
  # cell_ref_index is position in id_order (1..344208)
  # year is mapped to 1..n_years
  years_unique <- sort(unique(dt$year))
  year_to_col  <- setNames(seq_along(years_unique), as.character(years_unique))
  n_cells      <- length(id_order)
  n_years      <- length(years_unique)
  
  # row_lookup_matrix[cell_ref, year_col] = row index in data (or NA)
  row_lookup_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  
  cell_refs <- id_to_ref[as.character(dt$id)]
  year_cols <- year_to_col[as.character(dt$year)]
  
  row_lookup_matrix[cbind(cell_refs, year_cols)] <- dt$row_idx
  
  # Now build neighbor_lookup: for each row, find neighbor rows
  # Pre-expand neighbors per cell into a flat structure for speed
  n_rows <- nrow(dt)
  
  # Precompute cell_ref and year_col for every row
  row_cell_ref <- as.integer(cell_refs)  # length n_rows
  row_year_col <- as.integer(year_cols)  # length n_rows
  
  # For each row i:
  #   neighbor_cell_refs = neighbors[[ row_cell_ref[i] ]]
  #   neighbor_rows = row_lookup_matrix[ neighbor_cell_refs, row_year_col[i] ]
  #   remove NAs
  
  # Vectorized approach: build flat neighbor table
  # Step 1: Expand neighbors into a data.table of (cell_ref, neighbor_cell_ref)
  neighbor_dt <- rbindlist(lapply(seq_along(neighbors), function(j) {
    nb <- neighbors[[j]]
    if (length(nb) == 0) return(NULL)
    data.table(cell_ref = j, nb_cell_ref = as.integer(nb))
  }))
  
  # Step 2: For each row, get its cell_ref and year_col
  row_info <- data.table(
    row_i    = seq_len(n_rows),
    cell_ref = row_cell_ref,
    year_col = row_year_col
  )
  
  # Step 3: Join row_info with neighbor_dt on cell_ref
  # This gives us: for each row_i, all neighbor cell_refs
  setkey(neighbor_dt, cell_ref)
  setkey(row_info, cell_ref)
  
  expanded <- neighbor_dt[row_info, on = "cell_ref", allow.cartesian = TRUE,
                          nomatch = NULL]
  # expanded has columns: cell_ref, nb_cell_ref, row_i, year_col
  
  # Step 4: Look up the neighbor's row index
  expanded[, nb_row := row_lookup_matrix[cbind(nb_cell_ref, year_col)]]
  
  # Step 5: Remove NAs (neighbor cell-year doesn't exist in data)
  expanded <- expanded[!is.na(nb_row)]
  
  # Step 6: Split into list indexed by row_i
  setkey(expanded, row_i)
  
  # Pre-allocate list
  neighbor_lookup <- vector("list", n_rows)
  
  # Fill using split
  split_result <- split(expanded$nb_row, expanded$row_i)
  
  # split() returns only keys that exist; fill them in
  idx_names <- as.integer(names(split_result))
  for (k in seq_along(idx_names)) {
    neighbor_lookup[[ idx_names[k] ]] <- as.integer(split_result[[k]])
  }
  
  # Rows with no neighbors remain NULL; convert to integer(0)
  null_mask <- vapply(neighbor_lookup, is.null, logical(1))
  neighbor_lookup[null_mask] <- list(integer(0))
  
  neighbor_lookup
}

# =============================================================================
# OPTIMIZED compute_neighbor_stats using vectorized data.table operations
# =============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Build flat table of (row_i, neighbor_row)
  lens <- lengths(neighbor_lookup)
  row_i_vec <- rep.int(seq_len(n), lens)
  nb_row_vec <- unlist(neighbor_lookup, use.names = FALSE)
  
  if (length(nb_row_vec) == 0) {
    # No neighbors at all
    out <- matrix(NA_real_, nrow = n, ncol = 3)
    colnames(out) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
    return(out)
  }
  
  # Get neighbor values
  nb_vals <- vals[nb_row_vec]
  
  # Build data.table and compute grouped stats
  dt_nb <- data.table(row_i = row_i_vec, val = nb_vals)
  dt_nb <- dt_nb[!is.na(val)]
  
  stats <- dt_nb[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = row_i]
  
  # Allocate output matrix (NA by default for rows with no valid neighbors)
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  out[stats$row_i, 1] <- stats$nb_max
  out[stats$row_i, 2] <- stats$nb_min
  out[stats$row_i, 3] <- stats$nb_mean
  
  colnames(out) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  out
}

# =============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# =============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  for (col_idx in seq_len(ncol(stats_mat))) {
    data[[colnames(stats_mat)[col_idx]]] <- stats_mat[, col_idx]
  }
  data
}

# =============================================================================
# MAIN PIPELINE (drop-in replacement)
# =============================================================================

# Build the neighbor lookup ONCE (this is now fast: minutes, not days)
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# The trained Random Forest model is UNCHANGED.
# The numerical estimand is PRESERVED (identical max/min/mean computations).
# Predicted values from the RF model will be numerically identical.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M iterations of `paste()` + named-vector hash lookup | Eliminated |
| **Key construction** | String concatenation (`paste`) per row per neighbor | Integer matrix indexing: `row_lookup_matrix[cbind(cell_ref, year_col)]` |
| **Neighbor lookup** | Named character vector (~6.46M entries), queried ~25.8M times | Integer matrix direct indexing, then vectorized `data.table` join |
| **Stats computation** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | Vectorized `data.table` grouped aggregation on flat neighbor table |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes on the same laptop |
| **RF model** | Preserved (untouched) | Preserved (untouched) |
| **Numerical output** | Original `max/min/mean` | Identical `max/min/mean` |

The colleague misidentified the symptom (`do.call(rbind, ...)`) as the disease. The real pathology is millions of string-construction and hash-lookup operations inside `build_neighbor_lookup()`. Replacing strings with integer arithmetic and replacing row-wise `lapply` with vectorized `data.table` operations eliminates the bottleneck entirely.
END_RESPONSE>>>
