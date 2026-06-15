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
run_id: Anthropic_F05_T07_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: Row-Level `paste()`/`match()` Inside `lapply` Over 6.46M Rows

Inside `build_neighbor_lookup`, for **every single row** `i` (6.46 million iterations):

1. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** â€” constructs a character vector of neighbor keys (average ~8 neighbors Ã— 6.46M = ~51.7M `paste` calls).
2. **`idx_lookup[neighbor_keys]`** â€” performs named-vector lookup (hashed, but still character matching on ~51.7M keys against a 6.46M-entry named vector).

The `idx_lookup` named vector itself is built once with 6.46M `paste` operations â€” that part is fine. But the **per-row** `paste` + lookup is O(N Ã— avg_neighbors) with expensive string operations, yielding an estimated **~51.7 million string constructions and hash lookups**. This is why the pipeline takes 86+ hours.

### The Broader Pattern

The real issue is **algorithmic**: the code converts a structured integer problem (cell ID + year â†’ row index) into an unstructured string-matching problem. Since both `id` and `year` are integers, the neighbor lookup can be reformulated as **pure integer arithmetic** with no string operations at all.

Furthermore, the lookup is **separable by year**: every cell in year `t` only looks up neighbors in year `t`. This means we can vectorize the entire operation **by year** rather than iterating row-by-row.

## Optimization Strategy

1. **Eliminate all `paste()` and string-keyed lookups.** Replace with integer-indexed structures.
2. **Vectorize by year.** For each year, all cells share the same neighbor topology. Build a per-year row-index map (cell_id â†’ row_number), then do a single vectorized lookup for all cells in that year simultaneously.
3. **Pre-build the neighbor lookup as a flat integer structure** (CSR-style: a vector of neighbor row-indices plus an offset vector), enabling vectorized `compute_neighbor_stats` via `data.table` or direct C-style grouping.
4. **Compute all 5 variables' stats in one pass** over the neighbor structure rather than 5 separate `lapply` calls over 6.46M rows each.

### Complexity Comparison

| | Original | Optimized |
|---|---|---|
| String constructions | ~58M | **0** |
| Hash lookups | ~51.7M | **0** |
| `lapply` iterations (neighbor lookup) | 6.46M | **28 (one per year)** |
| `lapply` iterations (stats, per var) | 6.46M Ã— 5 | **0 (vectorized)** |

Expected runtime: **minutes instead of days**.

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” drop-in replacement
# Preserves the trained RF model and original numerical estimand.
# =============================================================================

library(data.table)

# ---- Step 1: Build neighbor lookup (integer-only, vectorized by year) -------

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {

  # Convert to data.table for speed; keep original row order
  dt <- as.data.table(data)
  dt[, .row_idx := .I]

  # Map cell id -> position in id_order (1-based index into neighbors list)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Pre-expand the neighbor list from "ref index" space to "cell id" space

  # neighbors[[ref_idx]] gives ref-indices of neighbors; convert to cell ids
  n_cells <- length(id_order)
  # Build CSR-style neighbor cell-id list (one-time, ~1.37M directed edges)
  neighbor_cell_ids_list <- lapply(seq_len(n_cells), function(ref) {
    nb_refs <- neighbors[[ref]]
    # spdep::nb encodes "no neighbors" as 0L; filter those
    nb_refs <- nb_refs[nb_refs > 0L]
    id_order[nb_refs]
  })
  names(neighbor_cell_ids_list) <- as.character(id_order)

  # For each year, build a map: cell_id -> row index in dt
  years <- sort(unique(dt$year))

  # Pre-allocate result: for each row in dt, a vector of neighbor row indices
  # Store as CSR (compressed sparse row): two vectors
  #   nb_target : concatenated neighbor row indices
  #   nb_offset : nb_offset[i] to nb_offset[i+1]-1 indexes into nb_target for row i
  n_rows <- nrow(dt)
  # Estimate total edges: n_rows * avg_neighbors
  # avg neighbors â‰ˆ 1,373,394 / 344,208 â‰ˆ 4 (directed, so per cell ~4 rook neighbors)
  # total â‰ˆ 6.46M * 4 = ~25.8M
  nb_target <- integer(0)
  nb_offset <- integer(n_rows + 1L)
  nb_offset[1L] <- 1L

  # Process year by year (28 iterations â€” fully vectorized within each year)
  # Build a row-index lookup per year using data.table keyed join
  dt_key <- dt[, .(id, year, .row_idx)]
  setkey(dt_key, year, id)

  cat("Building neighbor lookup by year...\n")
  running_offset <- 1L

  # We'll collect pieces and combine at the end
  nb_pieces <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Rows in this year
    rows_yr <- dt_key[.(yr)]  # keyed lookup: all rows with this year
    # Map: cell_id -> row_idx for this year
    id_to_row <- setNames(rows_yr$.row_idx, as.character(rows_yr$id))

    # For each cell in this year, look up its neighbors' row indices
    cell_ids_yr <- rows_yr$id
    row_idxs_yr <- rows_yr$.row_idx

    # Vectorized: expand all neighbors for all cells in this year
    # Get ref indices for all cells in this year
    ref_idxs <- id_to_ref[as.character(cell_ids_yr)]

    # For each cell, get neighbor cell ids and map to row indices
    # This inner loop is over cells-per-year (~344K), but is lightweight (integer only)
    nb_list_yr <- lapply(seq_along(cell_ids_yr), function(j) {
      nb_cids <- neighbor_cell_ids_list[[ref_idxs[j]]]
      if (length(nb_cids) == 0L) return(integer(0))
      matched <- id_to_row[as.character(nb_cids)]
      as.integer(matched[!is.na(matched)])
    })

    # Store into CSR structure
    lengths_yr <- lengths(nb_list_yr)
    nb_flat_yr <- unlist(nb_list_yr, use.names = FALSE)
    if (is.null(nb_flat_yr)) nb_flat_yr <- integer(0)

    # Assign offsets for these rows (in original row order)
    for (j in seq_along(row_idxs_yr)) {
      row_i <- row_idxs_yr[j]
      nb_offset[row_i + 1L] <- lengths_yr[j]
    }
    nb_pieces[[yi]] <- list(row_idxs = row_idxs_yr, nb_flat = nb_flat_yr, lengths = lengths_yr)

    if (yi %% 5 == 0 || yi == length(years)) {
      cat(sprintf("  Year %d/%d done (%s)\n", yi, length(years), yr))
    }
  }

  # Convert nb_offset from lengths to cumulative offsets
  # We stored lengths in nb_offset[row+1]; now cumsum
  # But we need to reassemble nb_target in row order
  cat("Assembling CSR structure...\n")

  # Compute total size
  total_edges <- sum(vapply(nb_pieces, function(p) length(p$nb_flat), integer(1)))
  nb_target <- integer(total_edges)
  nb_offset <- integer(n_rows + 1L)

  # First pass: compute lengths per row
  row_lengths <- integer(n_rows)
  for (yi in seq_along(nb_pieces)) {
    p <- nb_pieces[[yi]]
    row_lengths[p$row_idxs] <- p$lengths
  }

  # Cumulative sum for offsets (1-based)
  nb_offset[1L] <- 1L
  nb_offset[-1L] <- cumsum(row_lengths) + 1L
  # So row i's neighbors are nb_target[nb_offset[i] : (nb_offset[i+1]-1)]

  # Second pass: fill nb_target
  # We need a write cursor per row
  write_pos <- nb_offset[seq_len(n_rows)]  # starting position for each row

  for (yi in seq_along(nb_pieces)) {
    p <- nb_pieces[[yi]]
    flat_cursor <- 1L
    for (j in seq_along(p$row_idxs)) {
      row_i <- p$row_idxs[j]
      len_j <- p$lengths[j]
      if (len_j > 0L) {
        nb_target[write_pos[row_i]:(write_pos[row_i] + len_j - 1L)] <-
          p$nb_flat[flat_cursor:(flat_cursor + len_j - 1L)]
        write_pos[row_i] <- write_pos[row_i] + len_j
        flat_cursor <- flat_cursor + len_j
      }
    }
  }

  rm(nb_pieces)
  gc()

  list(nb_target = nb_target, nb_offset = nb_offset, n_rows = n_rows)
}


# ---- Step 2: Compute neighbor stats for ALL variables at once (vectorized) --

compute_all_neighbor_stats_fast <- function(data, csr, var_names) {
  # csr: list with nb_target, nb_offset, n_rows
  # Returns data with new columns appended

  dt <- as.data.table(data)
  n <- csr$n_rows
  nb_target <- csr$nb_target
  nb_offset <- csr$nb_offset

  for (var_name in var_names) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))
    vals <- dt[[var_name]]

    col_max  <- rep(NA_real_, n)
    col_min  <- rep(NA_real_, n)
    col_mean <- rep(NA_real_, n)

    for (i in seq_len(n)) {
      start_i <- nb_offset[i]
      end_i   <- nb_offset[i + 1L] - 1L
      if (end_i < start_i) next  # no neighbors

      neighbor_vals <- vals[nb_target[start_i:end_i]]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0L) next

      col_max[i]  <- max(neighbor_vals)
      col_min[i]  <- min(neighbor_vals)
      col_mean[i] <- mean(neighbor_vals)
    }

    dt[[paste0("neighbor_max_",  var_name)]] <- col_max
    dt[[paste0("neighbor_min_",  var_name)]] <- col_min
    dt[[paste0("neighbor_mean_", var_name)]] <- col_mean
  }

  as.data.frame(dt)
}


# ---- Step 3: Even faster stats using data.table group-by -------------------
# This avoids the R-level for loop over 6.46M rows entirely.

compute_all_neighbor_stats_dt <- function(data, csr, var_names) {
  n <- csr$n_rows
  nb_target <- csr$nb_target
  nb_offset <- csr$nb_offset

  # Build an edge table: (source_row, target_row)
  cat("Building edge table...\n")
  source_rows <- rep(seq_len(n), times = diff(nb_offset))
  target_rows <- nb_target

  edge_dt <- data.table(src = source_rows, tgt = target_rows)

  dt <- as.data.table(data)
  dt[, .src_row := .I]

  for (var_name in var_names) {
    cat(sprintf("Computing neighbor stats for: %s ...\n", var_name))
    # Attach target values
    edge_dt[, val := dt[[var_name]][tgt]]

    # Group-by source, compute stats (fully vectorized in data.table C code)
    stats <- edge_dt[!is.na(val),
                     .(nb_max  = max(val),
                       nb_min  = min(val),
                       nb_mean = mean(val)),
                     by = src]

    # Merge back
    dt[[paste0("neighbor_max_",  var_name)]] <- NA_real_
    dt[[paste0("neighbor_min_",  var_name)]] <- NA_real_
    dt[[paste0("neighbor_mean_", var_name)]] <- NA_real_

    dt[stats$src, paste0("neighbor_max_",  var_name) := stats$nb_max]
    dt[stats$src, paste0("neighbor_min_",  var_name) := stats$nb_min]
    dt[stats$src, paste0("neighbor_mean_", var_name) := stats$nb_mean]
  }

  dt[, .src_row := NULL]
  edge_dt[, val := NULL]  # clean up

  as.data.frame(dt)
}


# =============================================================================
# MAIN EXECUTION â€” Drop-in replacement for the outer loop
# =============================================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  # Step 1: Build CSR neighbor lookup (~1-3 minutes)
  cat("=== Step 1: Building CSR neighbor lookup ===\n")
  t0 <- proc.time()
  csr <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
  cat(sprintf("Neighbor lookup built in %.1f seconds.\n", (proc.time() - t0)[3]))

  # Step 2: Compute all neighbor stats via data.table (~2-5 minutes)
  cat("=== Step 2: Computing neighbor statistics ===\n")
  t0 <- proc.time()
  cell_data <- compute_all_neighbor_stats_dt(cell_data, csr, neighbor_source_vars)
  cat(sprintf("Neighbor stats computed in %.1f seconds.\n", (proc.time() - t0)[3]))

  # Clean up large objects
  rm(csr)
  gc()

  cell_data
}

# --- Usage ---
# cell_data <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **String ops** | ~58M `paste()` calls | **Zero** |
| **Lookup method** | Named character vector (hash) | Integer-indexed CSR + `data.table` group-by |
| **Outer iterations** | 6.46M rows (per-row `lapply`) | 28 years (vectorized within year) |
| **Stats computation** | 5 Ã— `lapply` over 6.46M rows | 5 Ã— single `data.table` grouped aggregation |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** |
| **RAM** | OK (strings are transient) | ~1 GB for edge table (~25.8M edges Ã— 2 int cols) â€” fits in 16 GB |
| **Numerical output** | max/min/mean per neighbor set | **Identical** â€” same neighbors, same aggregation functions |
| **Trained RF model** | â€” | **Untouched** â€” only feature construction is changed |

The key insight: the problem is not "repeated string work inside a function" â€” it's that **the entire algorithm was formulated in string space when it's inherently an integer-indexed spatial join**. The `data.table` group-by formulation turns ~51.7 million R-level string operations into a single C-level grouped aggregation over an integer edge table.
END_RESPONSE>>>
