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
run_id: Anthropic_F05_T07_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` iterates over every row (`~6.46M`) and for each row:

1. **Constructs paste keys** for that row's neighbors (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” string allocation per row.
2. **Looks up those keys** in a named character vector (`idx_lookup[neighbor_keys]`) â€” named vector lookup is O(n) hash probing per key, repeated millions of times.
3. The `idx_lookup` named vector itself is built from a `paste` over all 6.46M rows.

This means ~6.46M Ã— (avg ~4 rook neighbors) â‰ˆ **25.8M string constructions and hash lookups**, all in an interpreted R `lapply` loop with no vectorization.

### But the deeper structural insight is:

**The neighbor relationship is cell-to-cell, not cell-year-to-cell-year.** Neighbors don't change across years. The string-keyed lookup is reconstructing, for every single row, the mapping "which rows in the data correspond to the same year for my neighbor cells?" â€” but this mapping is **decomposable** into two static structures:

1. **A cellâ†’neighbors mapping** (fixed across years, already given by `rook_neighbors_unique`).
2. **A (cell, year)â†’row mapping** (fixed once the data is sorted).

If the data is **sorted by (id, year)** (or by **(year, id)**), you can replace all string hashing with **integer arithmetic**. Every cell has exactly 28 rows (one per year). If sorted by `(id, year)`, cell `j`'s row for year `t` is simply `(j-1)*28 + (t - 1991)`. No strings, no hash lookups, no `lapply` over 6.46M rows.

Furthermore, `compute_neighbor_stats` is then applied **5 times** over the same `neighbor_lookup` â€” the lookup construction cost is paid once, but the `lapply` over 6.46M entries (each indexing into a numeric vector) is paid 5 times. This too can be fully vectorized.

---

## Optimization Strategy

| Step | What changes | Why it's faster |
|------|-------------|-----------------|
| 1. **Sort data by (id, year)** | Guarantees row = `(cell_index - 1) * n_years + year_offset` | Enables pure integer arithmetic for row lookup |
| 2. **Build a flat integer neighbor-row matrix** | For each cell, store its neighbor cell indices. To get neighbor *rows* for a given year, use arithmetic. | Eliminates all `paste`/string-key work; O(1) per neighbor |
| 3. **Vectorize stats with matrix operations** | Extract all neighbor values at once using matrix indexing, compute stats column-wise | Eliminates 6.46M-iteration `lapply`; leverages C-level R internals |
| 4. **Process all 5 variables in one pass** | Single matrix-index construction, apply to each variable | Amortizes the index work |

**Estimated speedup:** From ~86 hours to **minutes** (the bottleneck becomes memory-bandwidth over ~25.8M integer lookups, which is trivial).

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED FEATURE CONSTRUCTION
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats loop
# Preserves: original numerical estimand (max, min, mean of neighbor values)
# Preserves: trained Random Forest model (no retraining needed)
# =============================================================================

library(data.table)

build_and_compute_all_neighbor_features <- function(cell_data,
                                                     id_order,
                                                     rook_neighbors_unique,
                                                     neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # Step 1: Convert to data.table for fast manipulation; sort by (id, year)
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Record original row order so we can restore it at the end
  dt[, orig_row_idx__ := .I]
  
  # Create a canonical cell index: integer 1..N_cells matching id_order
  n_cells <- length(id_order)
  id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))
  
  dt[, cell_idx__ := id_to_cellidx[as.character(id)]]
  
  # Sort by (cell_idx__, year) â€” this is the key invariant
  setorder(dt, cell_idx__, year)
  
  # Verify: each cell must have the same set of years
  years_vec   <- sort(unique(dt$year))
  n_years     <- length(years_vec)
  year_to_offset <- setNames(seq_len(n_years), as.character(years_vec))
  
  stopifnot(nrow(dt) == n_cells * n_years)  # balanced panel check
  
  # -------------------------------------------------------------------------
  # Step 2: Build flat neighbor structure (integer cell indices only)
  # -------------------------------------------------------------------------
  # rook_neighbors_unique is an nb object: list of length n_cells,
  # each element is an integer vector of neighbor indices into id_order.
  # We need neighbor cell indices (1-based into id_order).
  # nb objects already store indices into the original spatial object order,
  # which matches id_order by construction.
  
  # Flatten into a two-column matrix: (focal_cell_idx, neighbor_cell_idx)
  # for efficient vectorized operations.
  
  n_neighbors_per_cell <- lengths(rook_neighbors_unique)
  total_edges <- sum(n_neighbors_per_cell)
  
  focal_cell_idx <- rep(seq_len(n_cells), times = n_neighbors_per_cell)
  neighbor_cell_idx <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Handle nb objects where 0 means "no neighbors"
  valid <- neighbor_cell_idx > 0L
  focal_cell_idx    <- focal_cell_idx[valid]
  neighbor_cell_idx <- neighbor_cell_idx[valid]
  total_edges <- length(focal_cell_idx)
  
  cat(sprintf("Total directed neighbor edges: %d\n", total_edges))
  
  # -------------------------------------------------------------------------
  # Step 3: For each year, compute row indices via integer arithmetic
  #         Row of cell c in year-offset y: (c - 1) * n_years + y
  #         Then vectorize max/min/mean across all cells at once.
  # -------------------------------------------------------------------------
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }
  
  # We process year-by-year to keep memory bounded.
  # For each year, we only need one value per cell per variable.
  # This is ~344K cells Ã— ~4 neighbors = ~1.37M lookups per year â€” trivial.
  
  for (yi in seq_len(n_years)) {
    yr <- years_vec[yi]
    
    # Row indices for all cells in this year (sorted order)
    # Cell c's row in the sorted dt: (c - 1) * n_years + yi
    focal_rows    <- (focal_cell_idx - 1L) * n_years + yi
    neighbor_rows <- (neighbor_cell_idx - 1L) * n_years + yi
    
    # Rows for all cells in this year (for writing results)
    all_cell_rows <- (seq_len(n_cells) - 1L) * n_years + yi
    
    for (var_name in neighbor_source_vars) {
      vals <- dt[[var_name]]
      
      # Get neighbor values for every edge in this year
      nv <- vals[neighbor_rows]
      
      # We need to aggregate (max, min, mean) by focal_cell_idx.
      # Use data.table for fast grouped aggregation.
      edge_dt <- data.table(
        focal = focal_cell_idx,
        nval  = nv
      )
      
      # Remove NA neighbor values before aggregation
      edge_dt <- edge_dt[!is.na(nval)]
      
      if (nrow(edge_dt) > 0L) {
        agg <- edge_dt[, .(
          nmax  = max(nval),
          nmin  = min(nval),
          nmean = mean(nval)
        ), by = focal]
        
        # Write results into the correct rows of dt
        target_rows <- (agg$focal - 1L) * n_years + yi
        
        set(dt, i = target_rows, j = paste0(var_name, "_neighbor_max"),  value = agg$nmax)
        set(dt, i = target_rows, j = paste0(var_name, "_neighbor_min"),  value = agg$nmin)
        set(dt, i = target_rows, j = paste0(var_name, "_neighbor_mean"), value = agg$nmean)
      }
    }
    
    if (yi %% 5 == 0 || yi == n_years) {
      cat(sprintf("  Processed year %d (%d/%d)\n", yr, yi, n_years))
    }
  }
  
  # -------------------------------------------------------------------------
  # Step 4: Restore original row order and return as data.frame
  # -------------------------------------------------------------------------
  setorder(dt, orig_row_idx__)
  dt[, c("orig_row_idx__", "cell_idx__") := NULL]
  
  as.data.frame(dt)
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

# Original code (slow, ~86 hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_and_compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has the same 15 new columns (5 vars Ã— {max, min, mean})
# with identical numerical values. The trained RF model can be used as-is.
```

---

## Complexity Comparison

| | Original | Optimized |
|---|---|---|
| **String constructions** | ~6.46M (index) + ~25.8M (lookups) = **~32M** | **0** |
| **Hash lookups** | ~25.8M named-vector probes in R interpreter | **0** |
| **R-level loop iterations** | 6.46M (`lapply`) Ã— 2 passes (build + computeÃ—5) | 28 (years) Ã— 5 (vars) = **140** |
| **Core operation** | Interpreted R loop + string ops | Vectorized integer arithmetic + `data.table` grouped aggregation (C-level) |
| **Memory** | 6.46M-element list of integer vectors (~hundreds of MB) | ~1.37M-row edge `data.table` reused per year (~11 MB) |
| **Estimated time** | **~86 hours** | **~2â€“10 minutes** |

### Why the numerical results are identical:

- For every `(cell, year)` pair, the same set of neighbor cells is identified (from the same `rook_neighbors_unique` nb object).
- The same variable values are extracted for those neighbors in that year.
- The same `max`, `min`, `mean` aggregations are computed, with the same `NA` handling (NAs removed before aggregation; if all neighbors are NA or no neighbors exist, result is `NA`).
- The results are written to identically named columns.
- The trained Random Forest model reads these column names at prediction time and receives the same values.
END_RESPONSE>>>
