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
run_id: Anthropic_F12_T07_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-time matrix assembly per variable (5 calls total). Each call binds ~6.46M rows Ã— 3 columns â€” a few seconds at most with `do.call(rbind, ...)` on uniform-length numeric vectors.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** The `lapply` iterates over every row, and for each row it:
   - Calls `as.character(data$id[i])` â€” character coercion per row.
   - Looks up `id_to_ref[as.character(...)]` â€” named vector lookup (hashed, but still per-element overhead).
   - Extracts `id_order[neighbors[[ref_idx]]]` â€” subset of a potentially large vector.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” string concatenation for every neighbor of every row.
   - Looks up `idx_lookup[neighbor_keys]` â€” named vector lookup with string keys over a 6.46M-element named vector, **for every neighbor of every row**.

2. **Scale of the problem:** With ~6.46 million rows and an average of ~4 rook neighbors per cell, this inner function performs roughly **25.8 million string paste operations** and **25.8 million named-vector lookups** against a 6.46M-entry hash table â€” all inside a sequential `lapply`. This is the operation taking dozens of hours.

3. `compute_neighbor_stats()`, by contrast, does only cheap numeric indexing (`vals[idx]`) and simple arithmetic (`max`, `min`, `mean`) â€” these are vectorized and fast. The `do.call(rbind, result)` on a list of uniform 3-element numeric vectors is also fast (effectively a matrix reshape).

**Conclusion:** The deep bottleneck is the **string-key construction and lookup strategy in `build_neighbor_lookup()`**. The fix is to eliminate all string operations and replace them with pure integer arithmetic for row indexing.

---

## Optimization Strategy

1. **Replace string-keyed lookup with integer arithmetic.** Since the data has a regular panel structure (each of 344,208 cells Ã— 28 years), we can map `(cell_id, year)` â†’ row index using integer math instead of pasting strings and doing hash lookups. We build integer mappings: `id â†’ integer position` and `year â†’ integer position`, then compute row index as `(id_pos - 1) * n_years + year_pos` (or similar, depending on sort order).

2. **Pre-expand the neighbor lookup to a flat integer vector scheme.** Instead of building a list of 6.46M variable-length integer vectors, we pre-compute a CSR-like (Compressed Sparse Row) structure: two vectors â€” one of concatenated neighbor-row indices, one of offsets â€” enabling fast vectorized access.

3. **Vectorize `compute_neighbor_stats()`.** With the CSR structure, we can use a single vectorized C-level operation (via `data.table` grouping or a simple Rcpp snippet) instead of 6.46M `lapply` iterations.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, not model training. The numerical output (max, min, mean of neighbor values) is identical.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# =============================================================================

library(data.table)

# -------------------------------------------------------------------
# Step 0: Convert to data.table for fast operations (non-destructive)
# -------------------------------------------------------------------
# Assumes cell_data is a data.frame with columns: id, year, and the
# neighbor_source_vars. id_order and rook_neighbors_unique are as before.

cell_dt <- as.data.table(cell_data)

# -------------------------------------------------------------------
# Step 1: Build integer-arithmetic row index mapping
# -------------------------------------------------------------------
# Determine sort order of the data. We require (id, year) to be the
# unique key. We sort to guarantee a known layout.

setorder(cell_dt, id, year)

# Unique ids and years in sorted order
unique_ids   <- sort(unique(cell_dt$id))
unique_years <- sort(unique(cell_dt$year))
n_ids   <- length(unique_ids)
n_years <- length(unique_years)

stopifnot(nrow(cell_dt) == n_ids * n_years)  # confirm balanced panel

# Integer position maps (id -> 1..n_ids, year -> 1..n_years)
id_pos_map   <- setNames(seq_along(unique_ids),   as.character(unique_ids))
year_pos_map <- setNames(seq_along(unique_years),  as.character(unique_years))

# Row index from (id_position, year_position):
#   row = (id_pos - 1) * n_years + year_pos
# This works because data is sorted by (id, year).

# -------------------------------------------------------------------
# Step 2: Build CSR-style neighbor-row structure (integer only)
# -------------------------------------------------------------------
# For each cell id (in id_order), get its neighbor cell ids via
# rook_neighbors_unique, then map to id positions.
#
# id_order is the vector of cell ids in the order matching the nb object.

# Map id_order to positions in our sorted unique_ids
id_order_pos <- id_pos_map[as.character(id_order)]

# Build neighbor list in terms of id positions (not row indices yet)
# rook_neighbors_unique[[k]] gives neighbor indices into id_order
# So id_order[rook_neighbors_unique[[k]]] gives neighbor cell ids
# And id_pos_map[as.character(...)] gives their positions in unique_ids

# We need a mapping from each unique_id's position to its neighbors'
# positions. id_order may differ from unique_ids ordering, so we
# build a bridge.

# pos_in_id_order for each unique_id position:
# id_order_to_uid_pos: for each index k in id_order, the uid position
id_order_to_uid_pos <- id_pos_map[as.character(id_order)]

# For each uid position, which index in id_order does it correspond to?
uid_pos_to_id_order_idx <- integer(n_ids)
uid_pos_to_id_order_idx[id_order_to_uid_pos] <- seq_along(id_order)

# Now build neighbor uid positions for every uid position
cat("Building integer neighbor structure...\n")
neighbor_uid_pos_list <- vector("list", n_ids)
for (k in seq_along(id_order)) {
  uid_pos <- id_order_to_uid_pos[k]
  nb_indices <- rook_neighbors_unique[[k]]
  if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) {
    neighbor_uid_pos_list[[uid_pos]] <- integer(0)
  } else {
    neighbor_uid_pos_list[[uid_pos]] <- id_order_to_uid_pos[nb_indices]
  }
}

# -------------------------------------------------------------------
# Step 3: Expand to row-level CSR structure
# -------------------------------------------------------------------
# For row r corresponding to (uid_pos_r, year_pos_r), its neighbor rows
# are: (neighbor_uid_pos - 1) * n_years + year_pos_r
# We build offset and flat index vectors.

cat("Expanding to row-level CSR structure...\n")

# Precompute uid_pos and year_pos for every row
cell_dt[, uid_pos  := id_pos_map[as.character(id)]]
cell_dt[, year_pos := year_pos_map[as.character(year)]]

# Number of neighbors per uid
n_neighbors_per_uid <- vapply(neighbor_uid_pos_list, length, integer(1))

# Number of neighbors per row = number of neighbors for that row's uid
n_neighbors_per_row <- n_neighbors_per_uid[cell_dt$uid_pos]

total_edges <- sum(as.numeric(n_neighbors_per_row))
cat(sprintf("Total directed neighbor-row edges: %.0f\n", total_edges))

# Build CSR offset vector
offsets <- c(0L, cumsum(as.numeric(n_neighbors_per_row)))

# Build flat neighbor-row index vector
# We process by uid to avoid per-row R overhead
cat("Building flat neighbor index vector...\n")

flat_nb_rows <- integer(total_edges)
write_pos <- 1L

for (u in seq_len(n_ids)) {
  nb_uids <- neighbor_uid_pos_list[[u]]
  n_nb <- length(nb_uids)
  if (n_nb == 0L) next

  # All year positions for this uid
  # Rows for uid u are: ((u-1)*n_years + 1) to (u*n_years)
  base_row_start <- (u - 1L) * n_years

  # For each year (year_pos 1..n_years), the neighbor rows are
  # (nb_uid - 1) * n_years + year_pos
  nb_bases <- (nb_uids - 1L) * n_years  # length n_nb

  for (yp in seq_len(n_years)) {
    # Current row: base_row_start + yp
    # Neighbor rows for this (u, yp): nb_bases + yp
    nb_rows_here <- nb_bases + yp
    idx_range <- write_pos:(write_pos + n_nb - 1L)
    flat_nb_rows[idx_range] <- nb_rows_here
    write_pos <- write_pos + n_nb
  }
}

cat("CSR structure built.\n")

# -------------------------------------------------------------------
# Step 4: Vectorized compute_neighbor_stats using data.table grouping
# -------------------------------------------------------------------
# Strategy: build an edge table (row_idx, neighbor_row_idx), join
# values, group by row_idx, compute max/min/mean.

compute_neighbor_stats_fast <- function(cell_dt, flat_nb_rows, offsets,
                                        n_neighbors_per_row, var_name) {
  cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
  n_rows <- nrow(cell_dt)
  vals <- cell_dt[[var_name]]

  # Identify rows with at least one neighbor
  has_nb <- which(n_neighbors_per_row > 0L)

  # Build row_id vector corresponding to flat_nb_rows
  row_id_vec <- rep.int(has_nb, n_neighbors_per_row[has_nb])

  # Get neighbor values
  nb_vals <- vals[flat_nb_rows[seq_along(row_id_vec)]]

  # Build edge data.table
  edge_dt <- data.table(row_id = row_id_vec, nb_val = nb_vals)

  # Remove NAs in neighbor values
  edge_dt <- edge_dt[!is.na(nb_val)]

  # Group and compute stats
  stats_dt <- edge_dt[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = row_id]

  # Initialize result columns with NA
  col_max  <- rep(NA_real_, n_rows)
  col_min  <- rep(NA_real_, n_rows)
  col_mean <- rep(NA_real_, n_rows)

  col_max[stats_dt$row_id]  <- stats_dt$nb_max
  col_min[stats_dt$row_id]  <- stats_dt$nb_min
  col_mean[stats_dt$row_id] <- stats_dt$nb_mean

  # Add to cell_dt
  max_col_name  <- paste0("max_neighbor_",  var_name)
  min_col_name  <- paste0("min_neighbor_",  var_name)
  mean_col_name <- paste0("mean_neighbor_", var_name)

  cell_dt[, (max_col_name)  := col_max]
  cell_dt[, (min_col_name)  := col_min]
  cell_dt[, (mean_col_name) := col_mean]

  invisible(cell_dt)
}

# -------------------------------------------------------------------
# Step 5: Run for all neighbor source variables
# -------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_dt <- compute_neighbor_stats_fast(
    cell_dt, flat_nb_rows, offsets, n_neighbors_per_row, var_name
  )
}

# Convert back to data.frame if needed downstream (e.g., for predict())
cell_data <- as.data.frame(cell_dt)
# Drop helper columns
cell_data$uid_pos  <- NULL
cell_data$year_pos <- NULL

cat("Done. All neighbor features computed.\n")

# The trained Random Forest model is untouched.
# Use predict(rf_model, cell_data) as before.
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | These are minor (~seconds per variable) |
| **True bottleneck** | â€” | `build_neighbor_lookup()`: 6.46M iterations of `paste()` + named-vector string lookups against a 6.46M-key hash, totaling ~25.8M string operations |
| **Root cause** | â€” | String-based indexing (`paste(id, year, "_")` + named vector lookup) applied per-row inside `lapply` over 6.46M rows |
| **Fix** | â€” | Replace all string operations with integer arithmetic `(uid_pos - 1) * n_years + year_pos`, build a CSR structure once, then use vectorized `data.table` grouping for stats |
| **Expected speedup** | â€” | From ~86+ hours to **minutes** (the CSR build is O(cells Ã— years Ã— avg_neighbors) in integer ops; the stats computation is a single vectorized `data.table` group-by per variable) |
| **Model preservation** | âœ… | RF model is untouched; numerical outputs (max, min, mean of neighbor values) are identical |
END_RESPONSE>>>
