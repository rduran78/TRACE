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
run_id: Anthropic_F12_T10_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis: The Colleague Is Partially Right but Misses the Deeper Bottleneck

The colleague identifies `do.call(rbind, result)` and "repeated list binding" inside `compute_neighbor_stats()` as the main bottleneck. Let's audit this claim against the code:

**`compute_neighbor_stats()` â€” Is it the main bottleneck?**

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is not free, but it is a single vectorized call at the end. The `lapply` inside `compute_neighbor_stats` performs simple index-subsetting and scalar summary operations (`max`, `min`, `mean`) per row. There is **no repeated list binding** inside the function â€” it builds a list via `lapply` (pre-allocated internally by R) and binds once. So the colleague's specific claim about "repeated list binding" is factually wrong for this function. The `do.call(rbind, ...)` on ~6.46M small vectors is mildly expensive but not catastrophic.

**`build_neighbor_lookup()` â€” The True Deep Bottleneck**

The real bottleneck is `build_neighbor_lookup()`. For every one of the **~6.46 million rows**, it:

1. Converts `data$id[i]` to character and looks it up in a named vector (`id_to_ref`): **character conversion + hash lookup per row**.
2. Extracts neighbor cell IDs from the `neighbors` list using the reference index.
3. Calls `paste(..., sep = "_")` to build **multiple** neighbor key strings per row (average ~4 rook neighbors Ã— 6.46M rows â‰ˆ **~25.8 million `paste` calls' worth of string concatenation**).
4. Looks up each key in the named vector `idx_lookup` (which has ~6.46M entries â€” a very large hash table).
5. Filters `NA`s with subsetting.

This is executed **once**, but it iterates over 6.46 million rows, each doing string construction and lookup against a 6.46M-entry named character vector. The string-based keying strategy (`paste(id, year, sep = "_")`) is the architectural bottleneck. It turns what should be a fast integer-indexed operation into millions of string allocations and hash lookups.

**Quantitative comparison:**
- `build_neighbor_lookup`: ~6.46M iterations Ã— (character coercion + paste of ~4 keys + ~4 hash lookups in a 6.46M-key table) â†’ **dominant cost, likely 70â€“80%+ of the 86-hour runtime**.
- `compute_neighbor_stats`: ~6.46M iterations Ã— (integer vector subset + 3 simple scalar ops) Ã— 5 variables â†’ fast by comparison. The `do.call(rbind, ...)` is a single call per variable.

**Verdict: Reject the colleague's diagnosis.** The main bottleneck is `build_neighbor_lookup()`, specifically the per-row string key construction and lookup in a massive named character vector. `compute_neighbor_stats` is secondary, and `do.call(rbind, ...)` is a minor cost.

---

## Optimization Strategy

1. **Eliminate all string-based keying.** Replace the `paste(id, year, sep="_")` lookup with a pure integer-indexed approach using a precomputed integer matrix/mapping. Since `id` and `year` are structured, we can build a 2D integer index: `(id_position, year_position) â†’ row_number`, then look up neighbors by integer arithmetic alone.

2. **Vectorize `build_neighbor_lookup` entirely.** Instead of an R-level loop over 6.46M rows, expand the neighbor list once per cell (not per cell-year), then replicate across years using vectorized integer operations.

3. **Replace `do.call(rbind, lapply(...))` in `compute_neighbor_stats` with direct matrix-column operations** using a flat integer vector + grouping approach (or simply `vapply` which pre-allocates the output matrix).

4. **Preserve the trained Random Forest model and original numerical estimand.** The output columns must be numerically identical (same `max`, `min`, `mean` of neighbor values).

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# =============================================================================

#' Build a fast integer-indexed neighbor lookup using a (cell, year) matrix
#' instead of string-keyed hash tables.
#'
#' Returns a list of two objects:
#'   - nb_rows:  integer vector of all neighbor row indices (flat)
#'   - nb_ptr:   integer vector of length nrow(data)+1 (CSR-style pointers)
#'
#' nb_ptr[i]:(nb_ptr[i+1]-1) indexes into nb_rows to get
#' the neighbor row indices for row i of data.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {

  # --- Step 1: integer mappings (no strings) ---
  n_cells <- length(id_order)
  # Map cell id -> position in id_order (1-based)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)
  # If id_order values are not contiguous positive integers, use a hash:
  # But for spatial grid cell IDs they typically are. We handle the general

  # case with match() as a fallback.
  # id_to_pos <- setNames(seq_len(n_cells), id_order)  # fallback

  # Map year -> position (1-based)
  years <- sort(unique(data$year))
  n_years <- length(years)
  year_min <- min(years)
  # Assuming years are contiguous integers (1992:2019)
  # year_to_pos: simply year - year_min + 1
  # This avoids any lookup.

  # --- Step 2: build row-index matrix: cell_pos Ã— year_pos -> row_number ---
  # This is a matrix of size n_cells Ã— n_years.
  # If a (cell, year) combo doesn't exist, it stays 0.
  row_matrix <- matrix(0L, nrow = n_cells, ncol = n_years)

  cell_positions <- id_to_pos[data$id]
  year_positions <- data$year - year_min + 1L

  row_matrix[cbind(cell_positions, year_positions)] <- seq_len(nrow(data))

  # --- Step 3: expand neighbor relationships per cell into per-row lookups ---
  # For each cell position p, neighbors[[p]] gives positions of neighbor cells.
  # For each row i with cell_position c and year_position y,
  # neighbor rows = row_matrix[neighbors[[c]], y] (dropping zeros).
  #
  # We do this fully vectorized by expanding across years.

  # Pre-compute: for each cell, how many neighbors?
  nb_counts_per_cell <- lengths(neighbors)  # integer vector, length n_cells

  # Total directed neighbor relationships
  total_nb_links <- sum(nb_counts_per_cell)  # ~1.37M

  # For each cell, expand neighbor cell positions into a flat vector
  # and record which cell they belong to.
  nb_cell_flat <- unlist(neighbors, use.names = FALSE)  # neighbor cell positions
  nb_owner_cell <- rep(seq_len(n_cells), nb_counts_per_cell)  # owning cell position

  # Now for each year, we look up row_matrix[nb_cell_flat, y] and
  # row_matrix[nb_owner_cell, y] to map owner->row, neighbor->row.
  # We build CSR (compressed sparse row) structure for the final lookup.

  # Pre-allocate output lists
  # Maximum possible entries: total_nb_links * n_years (but some may be 0/missing)
  # We'll build vectors and then trim.

  # Allocate generously
  max_entries <- as.double(total_nb_links) * n_years
  nb_rows_list <- vector("list", n_years)
  owner_rows_list <- vector("list", n_years)

  for (y in seq_len(n_years)) {
    # For this year, which owner cells have a row?
    owner_row <- row_matrix[nb_owner_cell, y]   # length = total_nb_links
    nb_row    <- row_matrix[nb_cell_flat, y]     # length = total_nb_links

    # Keep only entries where both owner and neighbor have rows in this year
    valid <- owner_row > 0L & nb_row > 0L
    owner_rows_list[[y]] <- owner_row[valid]
    nb_rows_list[[y]]    <- nb_row[valid]
  }

  # Flatten
  all_owner_rows <- unlist(owner_rows_list, use.names = FALSE)
  all_nb_rows    <- unlist(nb_rows_list, use.names = FALSE)
  rm(owner_rows_list, nb_rows_list)

  # Sort by owner row to build CSR pointers
  ord <- order(all_owner_rows)
  all_owner_rows <- all_owner_rows[ord]
  all_nb_rows    <- all_nb_rows[ord]

  n_rows <- nrow(data)
  # Build CSR pointer: nb_ptr of length n_rows + 1

  nb_ptr <- integer(n_rows + 1L)
  if (length(all_owner_rows) > 0) {
    # tabulate counts per owner row
    counts <- tabulate(all_owner_rows, nbins = n_rows)
    nb_ptr[1L] <- 1L
    nb_ptr[2L:(n_rows + 1L)] <- cumsum(counts) + 1L
  } else {
    nb_ptr[] <- 1L
  }

  list(nb_rows = all_nb_rows, nb_ptr = nb_ptr)
}


#' Compute neighbor stats (max, min, mean) using the CSR neighbor structure.
#' Returns a matrix with columns: max, min, mean.

compute_neighbor_stats_fast <- function(data, nb_lookup, var_name) {
  vals <- data[[var_name]]
  nb_rows <- nb_lookup$nb_rows
  nb_ptr  <- nb_lookup$nb_ptr
  n <- nrow(data)

  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  # Vectorized approach: build a grouping vector and use group-level ops
  # Create a vector of "owner row" for each entry in nb_rows
  counts <- diff(nb_ptr)  # length n, count of neighbors per row
  has_nb <- which(counts > 0L)

  if (length(has_nb) == 0L) {
    return(cbind(out_max, out_min, out_mean))
  }

  # Extract all neighbor values at once
  all_nb_vals <- vals[nb_rows]

  # Owner assignment for each entry
  owner <- rep(has_nb, counts[has_nb])

  # Handle NAs in neighbor values
  valid <- !is.na(all_nb_vals)
  owner_valid <- owner[valid]
  vals_valid  <- all_nb_vals[valid]

  if (length(vals_valid) == 0L) {
    return(cbind(out_max, out_min, out_mean))
  }

  # Group-level max, min, sum, count
  # Using data.table for speed if available, otherwise tapply
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::data.table(owner = owner_valid, val = vals_valid)
    stats <- dt[, .(mx = max(val), mn = min(val), s = sum(val), cnt = .N),
                keyby = owner]
    out_max[stats$owner]  <- stats$mx
    out_min[stats$owner]  <- stats$mn
    out_mean[stats$owner] <- stats$s / stats$cnt
  } else {
    # Fallback: tapply (still vectorized, slower than data.table)
    out_max[sort(unique(owner_valid))]  <- tapply(vals_valid, owner_valid, max)
    out_min[sort(unique(owner_valid))]  <- tapply(vals_valid, owner_valid, min)
    out_mean[sort(unique(owner_valid))] <- tapply(vals_valid, owner_valid, mean)
  }

  cbind(out_max, out_min, out_mean)
}


#' Wrapper that matches the original interface: adds neighbor feature columns
#' to cell_data for a given variable.

compute_and_add_neighbor_features_fast <- function(data, var_name, nb_lookup) {
  stats <- compute_neighbor_stats_fast(data, nb_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}


# =============================================================================
# OUTER LOOP â€” drop-in replacement
# =============================================================================

# Build the fast neighbor lookup ONCE (replaces the old build_neighbor_lookup)
nb_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, nb_lookup)
}

# The trained Random Forest model is untouched and can be used for prediction
# on the updated cell_data as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary Table

| Component | Original Cost | Root Cause | Optimized Approach | Expected Speedup |
|---|---|---|---|---|
| `build_neighbor_lookup` | **~70-80% of 86 hrs** â€” 6.46M iterations of `paste()` + string hash lookups in a 6.46M-entry named vector | Per-row string key construction & lookup | Integer matrix `row_matrix[cell_pos, year_pos]`; vectorized expansion across years; CSR pointer structure | **~500â€“1000Ã—** (minutes instead of days) |
| `compute_neighbor_stats` | ~20-30% â€” 6.46M `lapply` iterations Ã— 5 vars; `do.call(rbind)` on 6.46M rows | R-level loop + `rbind` of many small vectors | Vectorized extraction of all neighbor values at once; grouped `max`/`min`/`mean` via `data.table` | **~50â€“200Ã—** |
| **Total** | **~86+ hours** | String-keyed architecture | Pure integer indexing + vectorized group operations | **Target: ~5â€“20 minutes** |

**Key insight:** The colleague was looking at the wrong function. The `do.call(rbind, ...)` in `compute_neighbor_stats` is a single call and is only a minor contributor. The true bottleneck is `build_neighbor_lookup`, where 6.46 million iterations each perform string construction via `paste()` and hash lookups in a ~6.46 million-entry named character vector. Replacing the entire string-key architecture with integer matrix indexing and a CSR neighbor structure eliminates the bottleneck while preserving numerical equivalence and leaving the trained Random Forest model untouched.
END_RESPONSE>>>
