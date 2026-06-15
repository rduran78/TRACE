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
run_id: Anthropic_F06_T10_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row R-level `lapply` calls

`build_neighbor_lookup` iterates over **~6.46 million rows** one at a time in R. For each row, it:
1. Looks up the cell's reference index.
2. Finds that cell's rook neighbors.
3. Pastes neighbor IDs with the current year to form keys.
4. Looks up those keys in a named vector.

This produces a **6.46-million-element list** of integer vectors, all in an interpreted R loop. The `paste()`-based key construction and named-vector lookup are especially expensive at this scale.

### Bottleneck B: `compute_neighbor_stats` â€” repeated per-row extraction

`compute_neighbor_stats` iterates over the 6.46M-element lookup list, subscripts into a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable). That's ~32.3 million R-level iterations with per-element vector subsetting.

### Why raster focal/kernel operations are *not* directly applicable

Focal operations assume a regular grid with a fixed rectangular kernel. Here the grid cells are indexed arbitrarily (not necessarily a complete rectangle), the neighbor structure is an irregular `nb` object (boundary cells have fewer neighbors), and the data is a **panel** (neighbors must come from the same year). A focal approach would require reshaping every variable into a 2D raster per year, running `focal()`, then extracting back â€” feasible but fragile if the grid has holes or irregular boundaries. The better strategy is to **vectorize the neighbor computation directly using sparse matrix algebra**, which perfectly preserves the irregular neighbor structure and the exact numerical results.

### Estimated speedup

| Step | Current | Optimized |
|---|---|---|
| Build lookup | ~20â€“40 hrs | ~10â€“30 sec (sparse matrix construction) |
| Compute stats (Ã—5) | ~40â€“50 hrs | ~2â€“5 min (sparse matrix multiply + group ops) |
| **Total** | **~86 hrs** | **~3â€“6 min** |

---

## 2. Optimization Strategy

1. **Replace the per-row lookup list with a sparse adjacency matrix** (`Matrix::sparseMatrix`) that encodes, for each row in `cell_data`, which other rows (same year) are its rook neighbors. This matrix `W` has dimensions `nrow(cell_data) Ã— nrow(cell_data)` but only ~6.8M non-zero entries (the directed neighbor pairs Ã— 28 years Ã· overlap â€” roughly the number of directed rook relationships times the number of years, minus boundary effects).

2. **Compute neighbor stats via vectorized sparse operations:**
   - **Mean:** `W %*% vals / W %*% ones` (sparse matrix-vector multiply).
   - **Max and Min:** Use `data.table` grouped operations keyed on (id, year), which avoids R-level row iteration entirely. Alternatively, iterate over the sparse matrix column-wise, but the `data.table` approach is simplest and fast.

3. **Preserve exact numerical results:** `max`, `min`, and `mean` of the same neighbor sets produce identical values â€” no approximation, no retraining needed.

---

## 3. Working R Code

```r
library(Matrix)
library(data.table)

# ===========================================================================
# STEP 1: Build a sparse row-adjacency matrix W  (one-time, ~10-30 sec)
# ===========================================================================
build_neighbor_sparse_matrix <- function(cell_data, id_order, rook_neighbors) {
  # cell_data must have columns: id, year
  # id_order: vector of cell IDs in the order matching rook_neighbors (nb object)
  # rook_neighbors: an nb object (list of integer vectors of neighbor indices

  #                 into id_order)

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # Map each (id, year) to its row index
  setkey(dt, id, year)

  n_cells <- length(id_order)
  n_rows  <- nrow(dt)

  # --- Build edge list in cell-ID space --------------------------------
  from_cell <- rep(seq_len(n_cells),
                   times = lengths(rook_neighbors))
  to_cell   <- unlist(rook_neighbors, use.names = FALSE)

  # Convert to IDs
  from_id <- id_order[from_cell]
  to_id   <- id_order[to_cell]

  # --- Expand over years using a non-equi merge in data.table ----------
  edges <- data.table(from_id = from_id, to_id = to_id)

  years <- sort(unique(dt$year))

  # Cross-join edges Ã— years, then look up row indices for both endpoints
  edges_expanded <- edges[, .(from_id, to_id, year = rep(list(years), .N)),
                          by = .I][, .(from_id, to_id, year = unlist(year))]

  # Remove the helper column
  edges_expanded[, I := NULL]

  # Look up row indices for (from_id, year) and (to_id, year)
  # We'll merge twice
  setkey(edges_expanded, from_id, year)
  from_lookup <- dt[, .(id, year, from_row = row_idx)]
  setkey(from_lookup, id, year)
  edges_expanded <- from_lookup[edges_expanded, nomatch = 0L]
  # Now columns: id, year, from_row, from_id, to_id
  # Rename for clarity
  setnames(edges_expanded, "id", "matched_from_id")

  setkey(edges_expanded, to_id, year)
  to_lookup <- dt[, .(id, year, to_row = row_idx)]
  setkey(to_lookup, id, year)
  edges_expanded <- to_lookup[edges_expanded, nomatch = 0L]

  # Build sparse matrix
  W <- sparseMatrix(
    i = edges_expanded$from_row,
    j = edges_expanded$to_row,
    x = 1,
    dims = c(n_rows, n_rows)
  )

  return(W)
}

# ===========================================================================
# STEP 2: Compute neighbor features via sparse ops  (~30 sec per variable)
# ===========================================================================
compute_neighbor_features_sparse <- function(cell_data, W, var_name) {
  vals <- cell_data[[var_name]]

  # Replace NA with 0 for multiplication; track non-NA counts separately
  not_na  <- as.numeric(!is.na(vals))
  vals0   <- ifelse(is.na(vals), 0, vals)

  # Number of non-NA neighbors per row
  n_valid <- as.numeric(W %*% not_na)

  # Sum of neighbor values (NA replaced by 0, so they don't contribute)
  s       <- as.numeric(W %*% vals0)

  # Mean
  nb_mean <- ifelse(n_valid == 0, NA_real_, s / n_valid)

  # ------- Max and Min via row-wise sparse iteration --------------------
  # For max and min we cannot use a simple matrix multiply.
  # Instead, we iterate over rows of W in C-style via its sparse structure.
  # dgCMatrix stores by column; convert to dgRMatrix (row-compressed) for

  # efficient row access, or use summary().

  Wr <- as(W, "RsparseMatrix")  # dgRMatrix: row-compressed

  nb_max <- rep(NA_real_, length(vals))
  nb_min <- rep(NA_real_, length(vals))

  # Wr@p: row pointers (0-based, length nrow+1)
  # Wr@j: column indices (0-based)
  p <- Wr@p
  j <- Wr@j

  # Vectorized approach: extract all neighbor values at once, then
  # split by row using the pointer vector.
  all_neighbor_vals <- vals[j + 1L]  # j is 0-based

  # Build a row-id vector aligned with j
  row_lengths <- diff(p)
  row_id      <- rep(seq_along(row_lengths), times = row_lengths)

  # Remove entries where the neighbor value is NA
  valid       <- !is.na(all_neighbor_vals)
  row_id_v    <- row_id[valid]
  vals_v      <- all_neighbor_vals[valid]

  if (length(vals_v) > 0) {
    # Use data.table for fast grouped max/min
    tmp <- data.table(row_id = row_id_v, val = vals_v)
    agg <- tmp[, .(nb_max = max(val), nb_min = min(val)), by = row_id]
    nb_max[agg$row_id] <- agg$nb_max
    nb_min[agg$row_id] <- agg$nb_min
  }

  # Return the three new columns
  prefix <- var_name
  out <- data.frame(
    nb_max  = nb_max,
    nb_min  = nb_min,
    nb_mean = nb_mean
  )
  names(out) <- paste0("nb_", c("max_", "min_", "mean_"), var_name)
  return(out)
}

# ===========================================================================
# STEP 3: Main pipeline
# ===========================================================================
# -- Build sparse matrix (once) --
W <- build_neighbor_sparse_matrix(cell_data, id_order, rook_neighbors_unique)

# -- Compute neighbor features for each source variable --
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  new_cols <- compute_neighbor_features_sparse(cell_data, W, var_name)
  cell_data <- cbind(cell_data, new_cols)
  cat("Done:", var_name, "\n")
}

# -- Predict with the already-trained Random Forest (unchanged) --
# e.g.:
# cell_data$predicted <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Memory Budget Check

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M Ã— 110 numeric cols) | ~5.7 GB |
| `W` sparse matrix (6.46M Ã— 6.46M, ~38M non-zeros) | ~0.6 GB |
| Intermediate vectors per variable | ~0.15 GB |
| **Total peak** | **~7â€“8 GB** â† fits in 16 GB |

---

## 5. Why This Preserves Correctness

- The sparse matrix `W` encodes **exactly** the same (row_i â†’ row_j) neighbor relationships that the original `build_neighbor_lookup` produced: same cell pairs, same year matching, same handling of boundary cells with fewer neighbors.
- `max`, `min`, and `mean` are computed over **exactly** the same sets of non-NA neighbor values.
- The trained Random Forest model is used as-is for prediction â€” no retraining, no feature changes.
- The numerical estimand is therefore identical to the original implementation.

### Alternative considered and rejected

A raster `focal()` approach would require: (a) reshaping each variable into a complete rectangular raster per year (28 rasters), (b) handling missing/masked cells at boundaries, (c) extracting results back into the panel. While potentially fast for `mean`, it does not naturally handle the irregular `nb` object and risks subtle boundary mismatches. The sparse-matrix approach is both faster to implement correctly and guaranteed to match the original neighbor structure.
END_RESPONSE>>>
