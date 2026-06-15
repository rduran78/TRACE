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
run_id: Anthropic_F06_T07_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine).
2. Constructs paste-based string keys for each neighbor Ã— year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) using character matching on a 6.46M-element named vector.

This character-key lookup is **O(k)** per probe in a named vector (R named vectors use linear hashing but with high constant overhead), repeated ~6.46M times with ~4 neighbors each â‰ˆ **25.8M character-match lookups against a 6.46M-length vector**. The `paste` calls also generate enormous temporary string allocations.

### Bottleneck B: `compute_neighbor_stats` â€” R-level loop over 6.46M rows

Each call to `compute_neighbor_stats` iterates over 6.46M list elements, subsetting a numeric vector and computing `max/min/mean`. This is called 5 times (once per source variable). The per-element overhead of R's `lapply` + subsetting + `is.na` filtering is substantial at this scale.

### Why raster focal/kernel operations don't directly apply

The comment in the prompt asks whether raster focal operations offer a useful analogy. They do conceptually (this is a spatial neighborhood aggregation), but the data is in **long panel format** (cell Ã— year), not a raster stack, and the neighbor structure comes from an irregular `spdep::nb` object. Converting to raster would require assuming a regular grid and re-engineering the pipeline. The better approach is to **vectorize the neighbor aggregation using sparse matrix algebra**, which is the generalized equivalent of a focal operation on an irregular lattice.

---

## 2. Optimization Strategy

### Strategy: Sparse Adjacency Matrix Ã— Data Matrix (Vectorized)

1. **Replace `build_neighbor_lookup`** entirely. Instead, build a sparse row-adjacency matrix **W** of dimension `(n_rows Ã— n_rows)` where `n_rows` â‰ˆ 6.46M. Entry `W[i,j] = 1` if row `j` is a rook neighbor of row `i` *in the same year*. This is constructed once using integer arithmetic (no string pasting).

2. **Replace `compute_neighbor_stats`** with sparse matrix operations:
   - **Mean**: `W_norm %*% x` where `W_norm` is row-normalized.
   - **Max and Min**: Use a grouped operation via the sparse structure. Since `dgCMatrix` column-wise operations are efficient, we iterate over rows of W in chunks or use `Matrix` summary + `data.table` grouping.

3. **Key insight for max/min**: True sparse-matrix multiplication only gives sums (and thus means). For max and min, we use `data.table` grouped operations on the sparse triplet representation, which is still fully vectorized in C and avoids any R-level row loop.

### Expected speedup

| Component | Current | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (paste + named vector) | ~30s (integer join via data.table) |
| Stats computation (Ã—5 vars) | ~hours (R lapply Ã—6.46M) | ~2-5 min (sparse mat multiply + data.table group) |
| **Total** | **86+ hours** | **~5-15 minutes** |

### Memory feasibility

- Sparse matrix W: ~6.46M rows, ~25.8M nonzero entries â†’ ~600 MB (triplet: 3 integer/double vectors of length 25.8M). Fits in 16 GB.
- Data matrix for 5 variables: 6.46M Ã— 5 doubles â‰ˆ 258 MB.

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact same numerical output (max, min, mean of rook neighbors)
# Preserves: the trained Random Forest model (no retraining)
# ==============================================================================

library(data.table)
library(Matrix)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # ---- Step 0: Convert to data.table for speed ----
  dt <- as.data.table(cell_data)
  n <- nrow(dt)
  
  # Assign a row index to every observation
  dt[, row_idx := .I]
  
  # ---- Step 1: Build a mapping from (cell_id, year) -> row_idx ----
  # Using integer keys avoids all paste/character overhead
  setkey(dt, id, year)
  
  # ---- Step 2: Expand the nb object into an edge list (spatial only) ----
  # rook_neighbors_unique is a list of length = length(id_order)
  # rook_neighbors_unique[[i]] contains integer indices into id_order
  
  message("Building spatial edge list...")
  
  # Pre-allocate edge list vectors
  n_cells <- length(id_order)
  # Count total edges
  edge_counts <- vapply(rook_neighbors_unique, length, integer(1))
  total_edges <- sum(edge_counts)
  
  from_cell_idx <- integer(total_edges)
  to_cell_idx   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    len_i <- length(nb_i)
    if (len_i > 0L) {
      from_cell_idx[pos:(pos + len_i - 1L)] <- i
      to_cell_idx[pos:(pos + len_i - 1L)]   <- nb_i
      pos <- pos + len_i
    }
  }
  
  # Convert cell indices to actual cell IDs
  edges_spatial <- data.table(
    from_id = id_order[from_cell_idx],
    to_id   = id_order[to_cell_idx]
  )
  
  rm(from_cell_idx, to_cell_idx)
  
  message(sprintf("  %d directed spatial edges.", nrow(edges_spatial)))
  
  # ---- Step 3: Cross-join edges with years to get row-level edges ----
  # For each spatial edge (A -> B), we need the edge for every year
  # where BOTH A and B exist.
  
  message("Joining edges with panel years...")
  
  # Create lookup: cell_id -> row_idx, year
  id_year_lookup <- dt[, .(id, year, row_idx)]
  
  # Join from-side
  setnames(id_year_lookup, c("id", "year", "row_idx"),
           c("from_id", "year", "from_row"))
  edges_full <- merge(edges_spatial, id_year_lookup,
                      by = "from_id", allow.cartesian = TRUE)
  
  # Join to-side
  setnames(id_year_lookup, c("from_id", "year", "from_row"),
           c("to_id", "year", "to_row"))
  edges_full <- merge(edges_full, id_year_lookup,
                      by = c("to_id", "year"), allow.cartesian = FALSE)
  
  rm(id_year_lookup, edges_spatial)
  gc()
  
  message(sprintf("  %d directed row-level edges.", nrow(edges_full)))
  
  # ---- Step 4: Compute neighbor stats using data.table grouped ops ----
  # edges_full now has columns: to_id, year, from_id, from_row, to_row
  # from_row is the "focal" cell-year row
  # to_row is the "neighbor" cell-year row
  
  message("Computing neighbor statistics...")
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing variable: %s", var_name))
    
    # Extract the variable values for all rows
    all_vals <- dt[[var_name]]
    
    # Attach neighbor values to the edge table
    edges_full[, nval := all_vals[to_row]]
    
    # Remove edges where the neighbor value is NA
    edges_valid <- edges_full[!is.na(nval)]
    
    # Compute grouped max, min, mean by from_row (the focal cell-year)
    stats <- edges_valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = from_row]
    
    # Create the column names matching the original pipeline
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    # Initialize with NA, then fill in computed values
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    dt[stats$from_row, (max_col)  := stats$nb_max]
    dt[stats$from_row, (min_col)  := stats$nb_min]
    dt[stats$from_row, (mean_col) := stats$nb_mean]
    
    rm(edges_valid, stats)
  }
  
  # ---- Step 5: Clean up and return ----
  edges_full[, nval := NULL]
  dt[, row_idx := NULL]
  
  # Return as data.frame if the input was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ==============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Requirement | Guarantee |
|---|---|
| **Same neighbor definition** | Uses the identical `rook_neighbors_unique` nb object; same directed edges. |
| **Same statistics** | `max`, `min`, `mean` computed on the identical set of non-NA neighbor values per focal cell-year. |
| **Same column names** | Output columns follow the same `{var}_neighbor_{stat}` naming convention. |
| **Same NA behavior** | Rows with no valid neighbors get `NA` for all three stats, matching the `c(NA, NA, NA)` return in the original. |
| **Trained RF model untouched** | No model retraining; the feature matrix is numerically identical, so `predict()` produces the same results. |

## 5. Summary of Key Changes

| Original | Optimized | Why |
|---|---|---|
| `paste(id, year)` string keys | Integer join via `data.table` merge | Eliminates ~13M string allocations |
| Named vector lookup (`idx_lookup[key]`) | `data.table` binary-search join (`setkey` + `merge`) | O(log n) vs O(n) per probe |
| `lapply` over 6.46M rows | `data.table` grouped `[, .(...), by=]` | C-level vectorized aggregation |
| 5 separate `lapply` passes | 5 grouped aggregations reusing the same edge table | Edge table built once, reused |
| **~86+ hours** | **~5â€“15 minutes** | **~500â€“1000Ã— speedup** |
END_RESPONSE>>>
