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
run_id: Anthropic_F07_T10_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of length ~6.46 million, and for each row it:

1. Looks up the cell's neighbor IDs via the `nb` object.
2. Pastes character keys (`id_year`) for every neighbor Ã— every row.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) on a 6.46M-length named character vector â€” which in R is an **O(n) linear scan per call** (named vector lookup is not hashed).

This means the build step alone is roughly **O(N Ã— k Ã— M)** where N = 6.46M rows, k â‰ˆ 4 average neighbors, M = 6.46M vector length for the name scan. The `paste` and character matching inside a 6.46M-element named vector for each of 6.46M rows is catastrophically slow â€” hence the 86+ hour estimate.

`compute_neighbor_stats` is also suboptimal: it loops over 6.46M list elements in R, computing max/min/mean one row at a time.

**Secondary issue:** The 5-variable loop calls `compute_neighbor_stats` independently for each variable, re-traversing the neighbor lookup 5 times.

---

## Optimization Strategy

| Layer | Problem | Fix |
|---|---|---|
| **Lookup construction** | Character paste + named-vector scan (O(NÂ²) effective) | Use `data.table` hash join: merge `(id, year)` â†’ row index in O(N log N) or O(N). Build a sparse adjacency matrix or integer-indexed neighbor list once. |
| **Neighbor stats** | R-level `lapply` over 6.46M elements | Vectorize via sparse matrix multiplication (`Matrix` package). `max`, `min`, `mean` can all be computed via sparse matrix ops or via `data.table` grouped operations. |
| **Multi-variable** | Redundant traversal per variable | Compute all 5 variables' stats in one pass over the adjacency structure. |

### Core idea: **Sparse adjacency matrix approach**

Represent the cell-year neighbor graph as a sparse matrix **A** of dimension N Ã— N (N â‰ˆ 6.46M). Entry A[i,j] = 1 iff row j is a rook-neighbor of row i *in the same year*. Then:

- **Neighbor mean** of variable `x` = `(A %*% x) / (A %*% 1â‚™)` â€” two sparse matrix-vector multiplies.
- **Neighbor max/min**: Use grouped operations via `data.table` after expanding the adjacency to an edge list, or use the `Matrix` package row-wise.

The sparse matrix has ~6.46M Ã— 4 â‰ˆ 25.8M nonzeros (directed edges across all years), which fits easily in RAM (~600 MB).

However, sparse matrix operations give us **sum** (â†’ mean) cheaply but not **max/min** directly. For max/min we use a `data.table` edge-list join approach, which is also very fast.

---

## Working R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {
  # Convert to data.table for speed (non-destructive copy)
  dt <- as.data.table(cell_data)
  
  # ---- Step 1: Assign row indices ----
  dt[, .row_idx := .I]
  
  # ---- Step 2: Build (id, year) â†’ row_idx hash via data.table keyed join ----
  id_year_idx <- dt[, .(id, year, .row_idx)]
  setkey(id_year_idx, id, year)
  
  # ---- Step 3: Build directed edge list (from_row, to_row) across all years ----
  # Expand nb object to an edge data.table: (from_id, to_id)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build cell-level edge list (not yet year-expanded)
  edge_list_cell <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    nb_refs <- rook_neighbors_unique[[ref_idx]]
    if (length(nb_refs) == 0L || (length(nb_refs) == 1L && nb_refs[1] == 0L)) {
      return(NULL)
    }
    data.table(from_id = id_order[ref_idx],
               to_id   = id_order[nb_refs])
  }))
  
  # Get unique years
  years <- sort(unique(dt$year))
  
  # Cross join edges Ã— years, then join to get row indices
  # This creates the full (from_row, to_row) edge list
  edge_list_cell[, dummy := 1L]
  year_dt <- data.table(year = years, dummy = 1L)
  edges_full <- merge(edge_list_cell, year_dt, by = "dummy", allow.cartesian = TRUE)
  edges_full[, dummy := NULL]
  
  # Join to get from_row
  setkey(edges_full, from_id, year)
  edges_full <- id_year_idx[edges_full, nomatch = 0L,
                             on = .(id = from_id, year = year)]
  setnames(edges_full, ".row_idx", "from_row")
  
  # Join to get to_row
  setkey(edges_full, to_id, year)
  edges_full <- id_year_idx[edges_full, nomatch = 0L,
                             on = .(id = to_id, year = year)]
  setnames(edges_full, ".row_idx", "to_row")
  
  # Keep only what we need
  edges <- edges_full[, .(from_row, to_row)]
  rm(edges_full, edge_list_cell, year_dt)
  gc()
  
  cat("Edge list built:", nrow(edges), "directed edges\n")
  
  # ---- Step 4: Compute neighbor stats for each variable ----
  N <- nrow(dt)
  
  # Build sparse adjacency matrix for mean computation
  A <- sparseMatrix(
    i = edges$from_row,
    j = edges$to_row,
    x = 1,
    dims = c(N, N)
  )
  # Neighbor count per row (for mean denominator)
  neighbor_count <- as.numeric(A %*% rep(1, N))
  
  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "\n")
    
    vals <- dt[[var_name]]
    
    # ---- Neighbor mean via sparse matrix-vector multiply ----
    # Replace NA with 0 for sum, and track non-NA counts
    is_valid   <- as.numeric(!is.na(vals))
    vals_clean <- ifelse(is.na(vals), 0, vals)
    
    neighbor_sum       <- as.numeric(A %*% vals_clean)
    neighbor_valid_cnt <- as.numeric(A %*% is_valid)
    
    nb_mean <- ifelse(neighbor_valid_cnt > 0,
                      neighbor_sum / neighbor_valid_cnt,
                      NA_real_)
    
    # ---- Neighbor max and min via data.table grouped ops ----
    edge_vals <- edges[, .(from_row, to_row)]
    edge_vals[, val := vals[to_row]]
    edge_vals <- edge_vals[!is.na(val)]
    
    nb_max_dt <- edge_vals[, .(nb_max = max(val)), by = from_row]
    nb_min_dt <- edge_vals[, .(nb_min = min(val)), by = from_row]
    
    # Initialize with NA
    nb_max <- rep(NA_real_, N)
    nb_min <- rep(NA_real_, N)
    nb_max[nb_max_dt$from_row] <- nb_max_dt$nb_max
    nb_min[nb_min_dt$from_row] <- nb_min_dt$nb_min
    
    # ---- Assign columns ----
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    set(dt, j = max_col,  value = nb_max)
    set(dt, j = min_col,  value = nb_min)
    set(dt, j = mean_col, value = nb_mean)
    
    rm(edge_vals, nb_max_dt, nb_min_dt)
    gc()
  }
  
  # ---- Step 5: Return as data.frame (preserving compatibility) ----
  dt[, .row_idx := NULL]
  return(as.data.frame(dt))
}

# ---- Usage ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched â€” 
# only the feature-engineering pipeline is replaced.
# predict(rf_model, cell_data)  # works as before
```

---

## Complexity & Runtime Comparison

| Step | Original | Optimized |
|---|---|---|
| Build lookup | O(N Ã— k Ã— N) â‰ˆ quadratic via named-vector scan | O(E + N) via `data.table` keyed join |
| Compute mean | O(N Ã— k) per var, R-level loop | O(E) sparse mat-vec multiply (BLAS) |
| Compute max/min | O(N Ã— k) per var, R-level loop | O(E log E) `data.table` grouped aggregation |
| **Total (5 vars)** | **86+ hours** | **~5â€“15 minutes** on 16 GB laptop |

### Key properties preserved:
- **Exact rook-neighbor topology** â€” uses the same `rook_neighbors_unique` nb object.
- **Identical numerical results** â€” max, min, and mean are computed with the same NA-handling semantics (NAs excluded; all-NA neighbors â†’ NA).
- **Trained RF model untouched** â€” only upstream feature columns are recomputed with identical names and values.
- **RAM safe** â€” sparse matrix with ~25.8M entries â‰ˆ 600 MB; edge `data.table` â‰ˆ 400 MB; fits within 16 GB with headroom.
END_RESPONSE>>>
