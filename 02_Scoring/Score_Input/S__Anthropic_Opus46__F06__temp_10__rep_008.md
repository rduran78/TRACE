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
run_id: Anthropic_F06_T10_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list with per-row `lapply`

This function iterates over all **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in the neighbor object.
2. Translates neighbor cell IDs into row keys (`paste(id, year)`).
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Named character vector lookups in R are **O(n)** per lookup in the worst case because R's named vectors use linear hashing that degrades with millions of entries. With ~6.46M rows and ~4 rook neighbors each, this produces **~25M string-match lookups** against a 6.46M-entry named vector. This alone accounts for the vast majority of the 86+ hour runtime.

### Bottleneck B: `compute_neighbor_stats` â€” per-row `lapply` over 6.46M list elements

For each of the 5 variables, another `lapply` iterates over 6.46M list elements, subsetting a numeric vector and computing `max`, `min`, `mean`. This is slow due to R-level loop overhead and repeated subsetting, but is secondary to Bottleneck A.

### Why raster focal/kernel operations are not directly applicable

Raster focal operations (e.g., `terra::focal`) assume a regular grid with uniform rectangular kernels. While the data is gridded, the panel structure (cell Ã— year), irregular coastlines/boundaries producing variable neighbor counts, and the need to match the exact `spdep::nb` rook-neighbor topology mean that a focal approach could silently change results at boundaries. We must **preserve the original numerical estimand**, so we use the exact same neighbor relationships but compute them efficiently.

---

## 2. Optimization Strategy

| Step | Technique | Speedup Factor |
|------|-----------|---------------|
| **Replace named-vector lookups with integer-indexed hash maps** | Use `data.table` keyed joins or environment-based hashing instead of named character vectors | ~50â€“100Ã— |
| **Pre-build a sparse integer matrix of neighbor row indices** | Build once, reuse for all 5 variables | Eliminates redundant lookup |
| **Vectorize the stats computation** | Use sparse matrix multiplication for mean; vectorized row operations for max/min via `data.table` grouping | ~20â€“50Ã— |
| **Avoid 6.46M-element R lists entirely** | Represent neighbor relationships as a two-column integer edge table (source_row, neighbor_row), then use `data.table` grouped aggregation | Massive memory and speed gain |

**Expected runtime: ~2â€“10 minutes** on 16 GB RAM laptop.

---

## 3. Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build an edge table of (row_index, neighbor_row_index)
#         This replaces build_neighbor_lookup entirely.
# ===========================================================================

build_neighbor_edges <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert cell_data to data.table if not already
  dt <- as.data.table(cell_data)
  
  # Create integer row index
  dt[, .row_idx := .I]
  
  # Create a keyed lookup: (id, year) -> row_idx
  # Using data.table keyed join (O(log n) per lookup, vectorized in C)
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  # Build mapping: cell id -> position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Get unique cell IDs present in the data
  unique_cells <- unique(dt$id)
  
  # Build edge list at the cell level first (cell_id -> neighbor_cell_ids)
  # This is only ~344K cells, very fast
  cell_edges <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    cell_id <- id_order[ref_idx]
    nb_indices <- rook_neighbors_unique[[ref_idx]]
    if (length(nb_indices) == 0 || (length(nb_indices) == 1 && nb_indices[1] == 0L)) {
      return(NULL)
    }
    nb_cell_ids <- id_order[nb_indices]
    data.table(cell_id = cell_id, nb_cell_id = nb_cell_ids)
  }))
  
  # Get all unique years
  all_years <- sort(unique(dt$year))
  
  # Cross join cell-level edges with years to get row-level edges
  # Use CJ and keyed join for efficiency
  cat("Building row-level edge table...\n")
  
  # Expand cell_edges across all years
  cell_edges_expanded <- cell_edges[, .(year = all_years), by = .(cell_id, nb_cell_id)]
  
  # Join to get source row index
  setkey(cell_edges_expanded, cell_id, year)
  cell_edges_expanded[row_lookup, src_row := i..row_idx, on = .(cell_id = id, year = year)]
  
  # Join to get neighbor row index
  cell_edges_expanded[row_lookup, nb_row := i..row_idx, on = .(nb_cell_id = id, year = year)]
  
  # Remove edges where either side is missing (cell doesn't exist in that year)
  edges <- cell_edges_expanded[!is.na(src_row) & !is.na(nb_row), .(src_row, nb_row)]
  
  # Clean up temporary column in dt
  dt[, .row_idx := NULL]
  
  return(edges)
}

# ===========================================================================
# STEP 2: Vectorized neighbor stats using data.table grouped aggregation
# ===========================================================================

compute_neighbor_stats_fast <- function(cell_data, edges, var_name) {
  n <- nrow(cell_data)
  vals <- cell_data[[var_name]]
  
  # Attach neighbor values to edge table
  edge_dt <- copy(edges)
  edge_dt[, nb_val := vals[nb_row]]
  
  # Remove NA neighbor values
  edge_dt <- edge_dt[!is.na(nb_val)]
  
  # Grouped aggregation â€” this is highly optimized in data.table (C-level)
  stats <- edge_dt[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = src_row]
  
  # Initialize output columns with NA
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)
  
  # Fill in computed values
  max_col[stats$src_row]  <- stats$nb_max
  min_col[stats$src_row]  <- stats$nb_min
  mean_col[stats$src_row] <- stats$nb_mean
  
  # Return named list matching original column naming convention
  col_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  result <- data.frame(max_col, min_col, mean_col)
  names(result) <- col_names
  return(result)
}

# ===========================================================================
# STEP 3: Main pipeline â€” drop-in replacement for the outer loop
# ===========================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  cat("Step 1/2: Building neighbor edge table...\n")
  t0 <- proc.time()
  
  edges <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
  
  cat(sprintf("  Edge table: %s edges built in %.1f seconds\n",
              format(nrow(edges), big.mark = ","),
              (proc.time() - t0)[3]))
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  cat("Step 2/2: Computing neighbor statistics...\n")
  
  for (var_name in neighbor_source_vars) {
    t1 <- proc.time()
    new_cols <- compute_neighbor_stats_fast(cell_data, edges, var_name)
    
    # Append columns to cell_data (preserves original data.frame structure)
    cell_data <- cbind(cell_data, new_cols)
    
    cat(sprintf("  %s: done in %.1f seconds\n", var_name, (proc.time() - t1)[3]))
  }
  
  cat("Pipeline complete.\n")
  return(cell_data)
}

# ===========================================================================
# USAGE (replaces the original outer loop)
# ===========================================================================

# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then predict with the existing trained Random Forest model as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Summary of Guarantees

| Requirement | How It Is Preserved |
|---|---|
| **Trained RF model unchanged** | No retraining occurs; we only compute input features and pass them to `predict()`. |
| **Original numerical estimand** | Same `spdep::nb` rook neighbor topology is used; same `max`, `min`, `mean` aggregations; identical results to the original code (up to floating-point associativity, which `data.table` preserves in practice). |
| **Memory (16 GB laptop)** | Edge table: ~1.37M cell-level edges Ã— 28 years â‰ˆ 38.4M rows Ã— 3 integer columns â‰ˆ **~0.9 GB**. Peak memory (with one `nb_val` column copy) â‰ˆ ~2â€“3 GB. Well within 16 GB. |
| **Runtime** | Edge table build: ~30â€“90 seconds. Each variable's stats: ~15â€“30 seconds. **Total: ~2â€“5 minutes** (vs. 86+ hours). |

The key insight is replacing **6.46 million R-level named-vector lookups** with a single vectorized `data.table` keyed join to build the edge table, and then replacing **6.46 million R-level `lapply` iterations** per variable with a single `data.table` grouped aggregation in C. This yields a roughly **~1,000Ã— speedup** while producing numerically identical results.
END_RESPONSE>>>
