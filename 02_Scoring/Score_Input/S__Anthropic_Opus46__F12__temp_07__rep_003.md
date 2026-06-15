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
run_id: Anthropic_F12_T07_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Audit and Optimization of Spatial Neighbor Pipeline

## Diagnosis

The colleague's claim is **partially correct but misses the deeper bottleneck**. Here is the full audit:

### 1. `do.call(rbind, result)` in `compute_neighbor_stats()`
This binds ~6.46 million 3-element vectors into a matrix. `do.call(rbind, list_of_vectors)` on millions of small vectors is indeed inefficient, but for 6.46M rows Ã— 3 columns, this operation takes seconds to maybe a minute at most. It is called only 5 times (once per variable). **This is not the dominant bottleneck.**

### 2. `compute_neighbor_stats()` inner `lapply`
The per-row lambda does subsetting, NA removal, and three summary statistics across ~6.46 million rows. This is moderately expensive but still manageable â€” each iteration is simple vector indexing. Not the main bottleneck either.

### 3. **The true deep bottleneck: `build_neighbor_lookup()`**
This function runs `lapply` over **every single row** (~6.46 million rows) and inside each iteration:
- Performs a **character key lookup** (`as.character(data$id[i])`) into `id_to_ref` â€” 6.46M `as.character` conversions and named-vector lookups.
- Extracts neighbor cell IDs from the `neighbors` list.
- Calls **`paste()`** to construct `neighbor_keys` for each neighbor of each row â€” this creates millions of temporary character vectors.
- Performs **named-vector lookup** (`idx_lookup[neighbor_keys]`) against a named vector of length 6.46 million â€” named vector lookup is O(n) per probe in the worst case because R uses linear hashing buckets for long named vectors.

The critical insight: `idx_lookup` is a named vector with **~6.46 million entries**. Named vector lookup in R degrades severely at this scale. Each of the ~6.46M rows looks up ~4 neighbors (rook adjacency) in this vector, yielding **~25.8 million character-match lookups against a 6.46M-entry named vector**. This is the dominant cost and explains the 86+ hour runtime.

Additionally, the lookup is **redundant across years**: every cell has the same neighbors in every year, so the neighbor *structure* is identical for all 28 years. Yet the function recomputes string keys and lookups for every cell-year row independently.

**Verdict: REJECT the colleague's diagnosis.** The main bottleneck is `build_neighbor_lookup()`, specifically the repeated `paste()`-based key construction and named-vector lookups at scale. The `do.call(rbind, ...)` is a minor secondary cost.

---

## Optimization Strategy

1. **Replace named-vector lookups with environment/hash-based or integer-indexed lookups.** Use `match()` or, better, direct integer indexing via a pre-built integer matrix.

2. **Exploit the year-invariant neighbor structure.** Build a neighbor lookup at the **cell level** (344,208 entries) once, then expand to cell-year rows via vectorized integer arithmetic â€” not per-row string operations.

3. **Vectorize `compute_neighbor_stats()`** using pre-allocated matrices and vectorized indexing instead of per-row `lapply`.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing identical numerical output.

---

## Optimized Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# Expected runtime: minutes instead of 86+ hours
# Produces numerically identical output; trained RF model is untouched.
# =============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # -------------------------------------------------------------------------
  # Key insight: neighbor relationships are YEAR-INVARIANT.
  # Step 1: Build a cell-level neighbor structure (344K cells, not 6.46M rows).
  # Step 2: Map cell-year rows to cell indices via integer arithmetic.
  # -------------------------------------------------------------------------

  n_cells <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)

  # -- Create integer mappings (no character operations) ---------------------
  # Map cell id -> cell index (1..n_cells)
  # Use match() which is vectorized and fast
  cell_idx_of_id <- match(id_order, id_order)  # identity, but we need the inverse:
  # We need: given an id value, what is its position in id_order?
  # Use a fast integer lookup. If ids are integers, we can use a vector index.
  max_id <- max(id_order)

  # Fast id-to-cell_index map (works if ids are positive integers)
  id_to_cellidx <- integer(max_id)
  id_to_cellidx[id_order] <- seq_along(id_order)

  # Map year -> year index (1..n_years)
  year_to_yearidx <- integer(max(years) - min(years) + 1)
  year_to_yearidx[years - min(years) + 1L] <- seq_along(years)

  # -- Build row index matrix: row_matrix[cell_idx, year_idx] = row in data --
  # This replaces the giant named character vector idx_lookup entirely.
  # Vectorized construction:
  data_cell_idx <- id_to_cellidx[data$id]
  data_year_idx <- year_to_yearidx[data$year - min(years) + 1L]

  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(data_cell_idx, data_year_idx)] <- seq_len(nrow(data))

  # -- Build cell-level neighbor list (integer indices into id_order) --------
  # neighbors (spdep::nb object) already provides this: neighbors[[i]] gives
  # the indices (into id_order) of the neighbors of cell i.
  # We just need to ensure they are integer vectors (they usually are).
  cell_neighbors <- lapply(neighbors, as.integer)

  # -- Expand to row-level neighbor lookup -----------------------------------
  # For each row in data, its neighbors are:
  #   cell_neighbors[[cell_idx]] mapped through row_matrix[, year_idx]
  # We do this vectorized per year to avoid 6.46M iterations.

  neighbor_lookup <- vector("list", nrow(data))

  for (yi in seq_along(years)) {
    # Rows in data for this year
    year_mask <- which(data_year_idx == yi)

    # For this year, the row_matrix column gives us the row index for
    # each cell. Neighbors of cell c are cell_neighbors[[c]], and their
    # row indices for this year are row_matrix[cell_neighbors[[c]], yi].

    for (ri in year_mask) {
      ci <- data_cell_idx[ri]
      nb_cells <- cell_neighbors[[ci]]
      if (length(nb_cells) == 0L) {
        neighbor_lookup[[ri]] <- integer(0)
      } else {
        nb_rows <- row_matrix[nb_cells, yi]
        neighbor_lookup[[ri]] <- nb_rows[!is.na(nb_rows)]
      }
    }
  }

  neighbor_lookup
}


# Even faster: fully vectorized compute_neighbor_stats using sparse expansion
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  n <- nrow(data)
  vals <- data[[var_name]]

  # Pre-allocate output
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  # Build a flat representation for vectorized grouped operations

  # rep_id[k] = which row "owns" the k-th neighbor entry
  # nb_idx[k] = the row index of the k-th neighbor
  lengths_vec <- vapply(neighbor_lookup, length, integer(1))
  total       <- sum(lengths_vec)

  if (total == 0L) {
    return(cbind(out_max, out_min, out_mean))
  }

  rep_id <- rep.int(seq_len(n), lengths_vec)
  nb_idx <- unlist(neighbor_lookup, use.names = FALSE)

  nb_vals <- vals[nb_idx]

  # Remove NAs: mask them out
  valid <- !is.na(nb_vals)
  rep_id  <- rep_id[valid]
  nb_vals <- nb_vals[valid]

  if (length(nb_vals) == 0L) {
    return(cbind(out_max, out_min, out_mean))
  }

  # Grouped max, min, sum, count using tapply or data.table for speed
  # For best performance, use data.table:
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::data.table(g = rep_id, v = nb_vals)
    agg <- dt[, .(mx = max(v), mn = min(v), sm = sum(v), cnt = .N), by = g]
    out_max[agg$g]  <- agg$mx
    out_min[agg$g]  <- agg$mn
    out_mean[agg$g] <- agg$sm / agg$cnt
  } else {
    # Fallback: split + vapply (still much faster than per-row lapply with
    # string lookups, because we've eliminated the lookup bottleneck)
    grp <- split(nb_vals, rep_id)
    grp_ids <- as.integer(names(grp))
    stats <- vapply(grp, function(x) c(max(x), min(x), mean(x)),
                    numeric(3))
    out_max[grp_ids]  <- stats[1, ]
    out_min[grp_ids]  <- stats[2, ]
    out_mean[grp_ids] <- stats[3, ]
  }

  cbind(out_max, out_min, out_mean)
}


# =============================================================================
# Drop-in replacement for the outer loop
# =============================================================================

# Build lookup ONCE (fast version â€” minutes, not days)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  cell_data[[paste0(var_name, "_neighbor_min")]]   <- stats_mat[, 2]
  cell_data[[paste0(var_name, "_neighbor_mean")]]  <- stats_mat[, 3]
}

# The trained Random Forest model is UNCHANGED â€” same features, same values.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M `paste()` + named-vector lookups against a 6.46M-entry vector | Eliminated: integer matrix indexing, year-invariant structure exploited |
| **`compute_neighbor_stats`** | 6.46M `lapply` iterations + `do.call(rbind, ...)` | Flat vectorized grouping via `data.table` |
| **Key construction** | ~25.8M `paste()` string concatenations | Zero string operations â€” pure integer indexing |
| **Lookup method** | Named character vector (O(n) hash collisions at scale) | Direct integer matrix subscript â€” O(1) |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **Numerical output** | Identical | Identical â€” same max/min/mean per row |
| **RF model** | Preserved | Preserved (untouched) |
END_RESPONSE>>>
