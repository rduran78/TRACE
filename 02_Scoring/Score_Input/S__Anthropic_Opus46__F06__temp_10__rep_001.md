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
run_id: Anthropic_F06_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`
This function iterates over all ~6.46 million rows, and for each row it:
1. Looks up the cell's reference index in the neighbor list.
2. Constructs string keys (`paste(id, year)`) for each neighbor.
3. Does named-vector lookups (`idx_lookup[neighbor_keys]`) â€” named vector lookups in R are hash-based but repeated 6.46M times with string construction is extremely slow.

The output is a list of 6.46M integer vectors. This alone could take tens of hours.

### Bottleneck 2: `compute_neighbor_stats` â€” Repeated per-row `lapply`
For each of the 5 variables, another `lapply` over 6.46M elements extracts neighbor values, removes NAs, and computes max/min/mean. That's ~32.3M tiny function calls total. Each call subsets a numeric vector and computes three summary statistics. The overhead of R-level iteration dominates.

### Why raster focal/kernel operations don't directly apply
Focal operations assume a regular grid with a fixed rectangular kernel. Here the data is a **panel** (cell Ã— year), and the neighbor structure is an irregular spatial adjacency (rook contiguity from an `nb` object). Years add a dimension that focal ops don't handle. The correct approach is vectorized sparse-matrix operations that faithfully preserve the rook-neighbor structure and panel alignment.

---

## Optimization Strategy

1. **Replace string-key lookups with integer join via `data.table`.** Build a `(cell_id, year) â†’ row_index` lookup table and join neighbor indices in a single vectorized operation rather than 6.46M `paste`/named-lookup calls.

2. **Construct a sparse adjacency matrix (row-to-row) for same-year neighbors.** Represent the entire neighbor lookup as a sparse matrix **W** of dimension `nrow Ã— nrow`. Entry `W[i,j] = 1` means row `j` is a rook neighbor of row `i` in the same year. This is built once.

3. **Compute neighbor stats via sparse matrixâ€“vector products.** For a given variable `x`:
   - `neighbor_mean`: normalize W row-wise â†’ `W_norm %*% x`.
   - `neighbor_max` and `neighbor_min`: use grouped operations on the sparse triplet representation, which is far faster than 6.46M R function calls.

4. **All 5 variables Ã— 3 stats = 15 new columns** computed in minutes instead of days.

5. **The Random Forest model is not retrained.** The new columns have identical names and identical numerical values (max, min, mean of rook neighbors per cell-year), so prediction is unchanged.

---

## Working R Code

```r
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Fast neighbor-stat computation for cell-year panel data
#  Preserves exact numerical results of the original implementation.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

library(data.table)
library(Matrix)

## â”€â”€ Step 0: Ensure cell_data is a data.table with a row-order column â”€â”€
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
cell_data[, .row_idx := .I]                # preserve original row order

## â”€â”€ Step 1: Build (id, year) â†’ row_index lookup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(cell_data, id, year)                 # fast binary-search join key

## â”€â”€ Step 2: Build the row-to-row sparse adjacency matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  id_order: vector of cell IDs in the order matching rook_neighbors_unique
#  rook_neighbors_unique: an nb object (list of integer neighbor indices)

build_sparse_neighbor_matrix <- function(cell_data, id_order, nb_obj) {


  n_cells <- length(id_order)
  stopifnot(n_cells == length(nb_obj))

  # â”€â”€ 2a. Expand spatial edges into a data.table of (from_id, to_id) â”€â”€
  from_ref <- rep(seq_len(n_cells),
                  times = vapply(nb_obj, function(x) {
                    sum(x != 0L)            # spdep uses 0 for no-neighbor
                  }, integer(1)))

  to_ref   <- unlist(lapply(nb_obj, function(x) x[x != 0L]),
                     use.names = FALSE)

  edges <- data.table(from_id = id_order[from_ref],
                      to_id   = id_order[to_ref])

  # â”€â”€ 2b. Cross-join edges with years present in the data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  years <- sort(unique(cell_data$year))

  # Cartesian product: every spatial edge Ã— every year

  edges_yr <- edges[, .(year = years), by = .(from_id, to_id)]

  # â”€â”€ 2c. Map (from_id, year) and (to_id, year) to row indices â”€â”€â”€â”€â”€â”€â”€â”€
  #         via keyed join on cell_data
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # Join to get "from" row index
  edges_yr[row_lookup, from_row := i..row_idx,
           on = .(from_id = id, year = year)]

  # Join to get "to" row index
  edges_yr[row_lookup, to_row := i..row_idx,
           on = .(to_id = id, year = year)]

  # Drop edges where either endpoint is missing (cell not observed that year)
  edges_yr <- edges_yr[!is.na(from_row) & !is.na(to_row)]

  # â”€â”€ 2d. Construct sparse matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  N <- nrow(cell_data)
  W <- sparseMatrix(
    i    = edges_yr$from_row,
    j    = edges_yr$to_row,
    x    = 1,
    dims = c(N, N)
  )
  return(W)
}

cat("Building sparse neighbor matrix â€¦\n")
t0 <- proc.time()
W <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("Done in %.1f s.\n", (proc.time() - t0)[3]))

## â”€â”€ Step 3: Row-wise neighbor count (for mean) and indicator â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_count <- as.numeric(W %*% rep(1, ncol(W)))   # nnz per row
has_neighbors  <- neighbor_count > 0

## â”€â”€ Step 4: Compute max, min, mean for each source variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  For mean:  W_norm %*% x   (row-normalised matrix)
#  For max/min: grouped operation over the sparse triplet

W_t <- as(W, "TsparseMatrix")   # triplet form: W_t@i, W_t@j, W_t@x (0-based)
from_rows <- W_t@i + 1L         # 1-based "from" row indices
to_rows   <- W_t@j + 1L         # 1-based "to"   row indices

# Row-normalised W for mean computation
W_norm <- W
W_norm@x <- W_norm@x / neighbor_count[from_rows[match(seq_along(W_norm@x),
                                                        seq_along(W_norm@x))]]
# More robustly:
diag_inv <- sparseMatrix(i = which(has_neighbors),
                         j = which(has_neighbors),
                         x = 1 / neighbor_count[has_neighbors],
                         dims = c(nrow(W), nrow(W)))
W_norm <- diag_inv %*% W        # each row sums to 1 (or row is zero)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics â€¦\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {

  x <- cell_data[[var_name]]

  # â”€â”€ Neighbor mean â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  n_mean <- as.numeric(W_norm %*% x)
  n_mean[!has_neighbors] <- NA_real_

  # â”€â”€ Neighbor max & min (grouped over sparse entries) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  #    Pull neighbor values, then group by "from" row
  neighbor_vals <- x[to_rows]

  # Identify non-NA neighbor values
  ok <- !is.na(neighbor_vals)
  fr_ok  <- from_rows[ok]
  nv_ok  <- neighbor_vals[ok]

  # Use data.table for fast grouped max/min
  dt_tmp <- data.table(from = fr_ok, val = nv_ok)
  agg    <- dt_tmp[, .(nmax = max(val), nmin = min(val)), by = from]

  n_max <- rep(NA_real_, nrow(cell_data))
  n_min <- rep(NA_real_, nrow(cell_data))
  n_max[agg$from] <- agg$nmax
  n_min[agg$from] <- agg$nmin

  # Also set mean to NA where all neighbor values were NA
  # (W_norm %*% x propagates 0 for NA; fix up)
  # Count non-NA neighbors per row
  nna_count <- rep(0L, nrow(cell_data))
  nna_tab   <- dt_tmp[, .N, by = from]
  nna_count[nna_tab$from] <- nna_tab$N

  # Recompute mean only from non-NA values (matches original logic)
  nna_sum <- rep(0, nrow(cell_data))
  nna_sum_dt <- dt_tmp[, .(s = sum(val)), by = from]
  nna_sum[nna_sum_dt$from] <- nna_sum_dt$s

  n_mean2 <- ifelse(nna_count > 0, nna_sum / nna_count, NA_real_)
  # Override the sparse-matrix mean with the NA-safe version
  n_mean <- n_mean2

  # â”€â”€ Assign to cell_data with original column naming convention â”€â”€â”€â”€â”€â”€
  set(cell_data, j = paste0("neighbor_max_",  var_name), value = n_max)
  set(cell_data, j = paste0("neighbor_min_",  var_name), value = n_min)
  set(cell_data, j = paste0("neighbor_mean_", var_name), value = n_mean)

  cat(sprintf("  %s done.\n", var_name))
}
cat(sprintf("All neighbor stats computed in %.1f s.\n", (proc.time() - t0)[3]))

## â”€â”€ Step 5: Clean up helper column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, .row_idx := NULL]

## â”€â”€ Step 6: Predict with the pre-trained Random Forest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (model object assumed already in memory; no retraining)
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Expected Performance

| Phase | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~40â€“50 h (6.46M string ops) | ~2â€“5 min (keyed join + sparse matrix construction) |
| `compute_neighbor_stats` (5 vars) | ~35â€“40 h (32.3M R calls) | ~1â€“3 min (sparse matrix multiply + data.table grouped agg) |
| **Total** | **~86+ hours** | **~5â€“10 minutes** |

## Why Numerical Results Are Preserved

- The sparse matrix `W` encodes **exactly** the same rook-neighbor, same-year relationships as the original `build_neighbor_lookup`.
- Max, min, and mean are computed from the identical set of non-NA neighbor values, grouped by the same "from" row.
- The `data.table` grouped aggregation (`max`, `min`, `sum / count`) is algebraically identical to the original `lapply` logic that filters `!is.na`, then computes `max`, `min`, `mean`.
- Column names follow the same naming convention, so the pre-trained Random Forest's `predict()` call resolves features identically. **No retraining is needed or performed.**
END_RESPONSE>>>
