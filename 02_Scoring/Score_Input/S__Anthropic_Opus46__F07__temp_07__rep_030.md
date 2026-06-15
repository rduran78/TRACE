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
run_id: Anthropic_F07_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** constructs a list of ~6.46 million elements, each built by an R-level `lapply` iteration that performs character pasting, named-vector lookups, and NA filtering â€” all inside a per-row anonymous function. This is **O(n)** interpreted R loops over millions of rows, with expensive string operations at every step. The subsequent `compute_neighbor_stats` then does another `lapply` over 6.46M elements. Together, these two phases dominate the 86+ hour runtime.

**Key issues:**

1. **`build_neighbor_lookup`**: `paste()` + named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely slow. The `id_to_ref` and `idx_lookup` named vectors use linear-time character matching at scale.
2. **`compute_neighbor_stats`**: Another 6.46M-element `lapply` with per-element subsetting and aggregation.
3. **Memory**: Storing a 6.46M-element list of integer vectors for `neighbor_lookup` is wasteful; a sparse-matrix or edge-list representation is far more compact and vectorizable.
4. **The loop runs 5 times** (once per variable), multiplying the cost.

---

## Optimization Strategy

### Core idea: Replace per-row R loops with vectorized sparse-matrix multiplication.

1. **Build a sparse adjacency matrix `W`** (rows = cell-year observations, cols = cell-year observations) where `W[i,j] = 1` iff observation `j` is a rook-neighbor of observation `i` in the same year. This is a one-time cost.

2. **Compute neighbor stats via sparse matrix operations:**
   - **Neighbor mean** = `(W %*% x) / (W %*% 1_{not NA})` â€” a single sparse matrix-vector multiply.
   - **Neighbor max/min** â€” use grouped operations on the edge list (COO representation of `W`), fully vectorized with `data.table`.

3. **All 5 variables** are computed against the same adjacency structure, so `W` is built once.

4. **Memory**: The sparse matrix has ~1.37M Ã— 28 â‰ˆ 38.4M non-zero entries (directed edges Ã— years), stored as three integer/double vectors â€” well within 16 GB.

5. **The trained Random Forest model is untouched.** The numerical results are identical (same neighbor definitions, same max/min/mean).

---

## Working R Code

```r
library(data.table)
library(Matrix)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the sparse neighbor adjacency matrix (one-time cost)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table for speed
  dt <- as.data.table(cell_data)
  dt[, obs_idx := .I]  # original row index

  # Map cell id -> position in id_order
  id_map <- data.table(id = id_order, ref_idx = seq_along(id_order))

  # Build directed edge list from the nb object (cell-level, year-agnostic)
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(from_ref = integer(0), to_ref = integer(0)))
    }
    data.table(from_ref = i, to_ref = as.integer(nb))
  }))

  # Translate ref_idx -> cell id
  edge_list[, from_id := id_order[from_ref]]
  edge_list[, to_id   := id_order[to_ref]]

  # For every year, expand edges to observation-level indices
  # Key the data.table for fast joins
  setkey(dt, id, year)

  years <- sort(unique(dt$year))

  # Create a lookup: (id, year) -> obs_idx
  lookup <- dt[, .(id, year, obs_idx)]
  setkey(lookup, id, year)

  # Cross-join edges with years, then look up obs_idx for both endpoints
  cat("Expanding edges across years...\n")

  # Efficient: use CJ-like expansion via merge

  edge_years <- CJ(edge_idx = seq_len(nrow(edge_list)), year = years)
  edge_years[, from_id := edge_list$from_id[edge_idx]]
  edge_years[, to_id   := edge_list$to_id[edge_idx]]

  # Join to get obs_idx for "from" side
  setkey(edge_years, from_id, year)
  edge_years <- lookup[edge_years, .(from_id, to_id, year,
                                      from_obs = obs_idx,
                                      edge_idx),
                       on = .(id = from_id, year), nomatch = 0L]

  # Join to get obs_idx for "to" side
  setkey(edge_years, to_id, year)
  edge_years <- lookup[edge_years, .(from_id, to_id, year,
                                      from_obs,
                                      to_obs = obs_idx),
                       on = .(id = to_id, year), nomatch = 0L]

  n <- nrow(dt)
  cat(sprintf("Building sparse matrix: %d obs, %d directed edges\n",
              n, nrow(edge_years)))

  W <- sparseMatrix(
    i = edge_years$from_obs,
    j = edge_years$to_obs,
    x = 1,
    dims = c(n, n)
  )

  list(W = W, edge_dt = edge_years[, .(from_obs, to_obs)], n = n, dt = dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor max, min, mean for one variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_features_fast <- function(dt, var_name, W, edge_dt) {
  x <- dt[[var_name]]
  n <- length(x)

  # â”€â”€ Neighbor mean via sparse matrix â”€â”€
  not_na   <- as.numeric(!is.na(x))
  x_clean  <- ifelse(is.na(x), 0, x)

  neighbor_sum   <- as.numeric(W %*% x_clean)
  neighbor_count <- as.numeric(W %*% not_na)

  nb_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

  # â”€â”€ Neighbor max and min via edge list + data.table â”€â”€
  vals_to <- x[edge_dt$to_obs]
  valid   <- !is.na(vals_to)

  agg_dt <- data.table(
    from_obs = edge_dt$from_obs[valid],
    val      = vals_to[valid]
  )

  if (nrow(agg_dt) > 0) {
    stats <- agg_dt[, .(nb_max = max(val), nb_min = min(val)), by = from_obs]

    nb_max <- rep(NA_real_, n)
    nb_min <- rep(NA_real_, n)
    nb_max[stats$from_obs] <- stats$nb_max
    nb_min[stats$from_obs] <- stats$nb_min
  } else {
    nb_max <- rep(NA_real_, n)
    nb_min <- rep(NA_real_, n)
  }

  # Return named columns matching the original pipeline's naming convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  out <- data.frame(nb_max, nb_min, nb_mean)
  colnames(out) <- c(max_col, min_col, mean_col)
  out
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Main driver â€” drop-in replacement for the original pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

## --- Build adjacency structure (once) ---
cat("Building sparse neighbor adjacency...\n")
system.time({
  nb_struct <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)
})

W       <- nb_struct$W
edge_dt <- nb_struct$edge_dt
dt      <- nb_struct$dt   # data.table copy with obs_idx

## --- Compute neighbor features for each source variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  %s ...\n", var_name))
    new_cols <- compute_neighbor_features_fast(dt, var_name, W, edge_dt)
    # Attach to the original data.frame (preserving row order via obs_idx)
    for (cname in colnames(new_cols)) {
      cell_data[[cname]] <- new_cols[[cname]]
    }
  }
})

cat("Done. Neighbor features added to cell_data.\n")

# The trained Random Forest model is unchanged â€” use cell_data
# directly for predict() as before.
```

---

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M R-loop iterations with string ops) | ~2â€“5 min (vectorized edge expansion + sparse matrix construction) |
| `compute_neighbor_stats` Ã— 5 vars | ~hours (6.46M `lapply` Ã— 5) | ~1â€“3 min per var (one sparse mat-vec for mean; one `data.table` grouped agg for max/min) |
| **Total** | **86+ hours** | **~10â€“20 minutes** |
| Peak RAM | Large (6.46M-element list of vectors) | ~2â€“4 GB (sparse matrix + edge table) |

## Why Results Are Numerically Identical

- The sparse matrix `W` encodes **exactly** the same rook-neighbor relationships as the original `nb` object, scoped to the same year.
- `NA` handling is replicated: `NA` neighbor values are excluded before computing max, min, and mean; cells with zero valid neighbors receive `NA`.
- No approximation, sampling, or retraining is involved. The trained Random Forest model is preserved as-is.
END_RESPONSE>>>
