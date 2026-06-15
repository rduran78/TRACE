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
run_id: Anthropic_F07_T07_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` â€” O(nÂ²) list construction via per-row string matching
- For each of the **~6.46 million rows**, the function:
  - Looks up the cell's neighbor IDs.
  - Pastes strings to create keys (`"id_year"`).
  - Performs named-vector lookups (`idx_lookup[neighbor_keys]`).
- Named vector lookup in R is **hash-based but slow at scale** when called millions of times in an `lapply`. The dominant cost is the **6.46M iterations** of the inner function, each doing string operations and hash lookups.
- Estimated: with ~4 neighbors per cell on average (rook), that's ~25.8M string-paste + hash-lookup operations, all inside an interpreted R loop.

### 2. `compute_neighbor_stats` â€” Repeated per variable, interpreted loop
- For each of the **5 variables**, another `lapply` over 6.46M rows extracts neighbor values and computes `max`, `min`, `mean`.
- That's **5 Ã— 6.46M = 32.3M** R-level function calls.

### Combined effect
- ~6.46M R-level iterations for the lookup build.
- ~32.3M R-level iterations for the stats.
- Total: **~38.8M interpreted R loop iterations**, each with non-trivial work â†’ **86+ hours**.

---

## Optimization Strategy

### Key Insight: Vectorize everything via sparse matrix multiplication and grouped operations.

**Step 1: Replace the row-level lookup with a sparse adjacency matrix.**
- Construct a sparse matrix **W** of dimension `N_cells Ã— N_cells` from `rook_neighbors_unique` (the `nb` object).
- This matrix is time-invariant. For each year, the same adjacency applies.

**Step 2: For each year and each variable, compute neighbor stats via matrix operations.**
- For a given year, extract the variable vector `v` (length = N_cells).
- `W %*% v` gives neighbor sums; `W %*% 1` gives neighbor counts â†’ **mean** = sum/count.
- For **max** and **min**, use a single pass over the sparse matrix structure (CSC/CSR) or use `data.table` grouped operations on an edge list.

**Step 3: Use `data.table` for the edge-list approach (most practical and fast).**
- Convert the `nb` object to an edge list: `(from_id, to_id)`.
- Join with the panel data keyed on `(id, year)` to get neighbor values.
- Group by `(from_id, year)` and compute `max`, `min`, `mean` in one pass.
- This turns 6.46M Ã— 5 R-level loops into **5 vectorized `data.table` grouped aggregations** â€” typically seconds each.

**Estimated speedup: from 86+ hours to ~2â€“5 minutes.**

The numerical results are **identical** (same neighbors, same `max`/`min`/`mean`). The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Convert the nb object to a data.table edge list (one-time, fast)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
nb_to_edge_dt <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector mapping position -> cell id
  from <- rep(
    seq_along(neighbors),
    times = lengths(neighbors)
  )
  to <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    from_id = id_order[from],
    to_id   = id_order[to]
  )
}

edge_dt <- nb_to_edge_dt(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id, to_id
# Each row means: to_id is a rook-neighbor of from_id

cat("Edge list rows:", nrow(edge_dt), "\n")
# Should be ~1,373,394

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year are keyed for fast joins
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Vectorized neighbor feature computation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Build a small lookup: just id, year, and the variable of interest
  lookup <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(lookup, id, year)
  
  # Get all unique years
  years <- sort(unique(cell_dt$year))
  
  # Cross-join edges with years, then look up neighbor values

  # More memory-efficient: expand edges by year via CJ inside a join
  
  # Create edge-year table:
  #   For each edge (from_id, to_id) and each year,
  #   look up the neighbor's (to_id) value in that year.
  
  # Step A: expand edges Ã— years
  edge_year <- edge_dt[, CJ(from_id = from_id, to_id = to_id, year = years, unique = TRUE)]
  # The above would be too large. Instead, do it properly:
  
  # Actually, since every edge exists in every year, we do:
  edge_year <- CJ_dt(edge_dt, years)
  
  # Better approach â€” avoid massive CJ; just do a keyed join per year
  # or replicate edges for all years at once:
  
  # Most efficient: replicate edge list across years
  edge_year <- edge_dt[, .(from_id, to_id, year = rep(years, each = .N)), 
                        by = .EACHI]  # This won't work directly.
  
  # Correct and simple approach:
  edge_year <- edge_dt[, {
    .(from_id = from_id, to_id = to_id)
  }]
  edge_year <- edge_year[rep(seq_len(.N), length(years))]
  edge_year[, year := rep(years, each = nrow(edge_dt))]
  
  # Join to get neighbor value
  setkey(edge_year, to_id, year)
  setkey(lookup, id, year)
  edge_year[lookup, val := i.val, on = .(to_id = id, year)]
  
  # Aggregate: for each (from_id, year), compute max, min, mean of neighbor vals
  agg <- edge_year[!is.na(val), 
                    .(nb_max  = max(val),
                      nb_min  = min(val),
                      nb_mean = mean(val)),
                    by = .(from_id, year)]
  
  # Rename columns to match expected output
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Join back to cell_dt
  setkey(agg, from_id, year)
  setkey(cell_dt, id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  cell_dt[agg, (c(max_col, min_col, mean_col)) := 
            mget(paste0("i.", c(max_col, min_col, mean_col))),
          on = .(id = from_id, year)]
  
  invisible(cell_dt)
}
```

The above has a memory problem: `edge_dt` Ã— 28 years = ~38.5M rows, which is fine for 16 GB, but let's be more careful and do it **year-by-year** to keep peak memory low:

```r
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. FINAL OPTIMIZED VERSION â€” year-by-year, memory-safe
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns with NA
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Build a fast lookup: id -> value, per year
  setkey(cell_dt, id, year)
  years <- sort(unique(cell_dt$year))
  
  for (yr in years) {
    # Extract this year's values
    yr_data <- cell_dt[year == yr, .(id, val = get(var_name))]
    setkey(yr_data, id)
    
    # Join neighbor values onto edge list
    # edge_dt: (from_id, to_id)
    # We want val of to_id in this year
    edges_with_val <- edge_dt[yr_data, on = .(to_id = id), nomatch = 0L]
    # edges_with_val now has: from_id, to_id, val
    
    # Aggregate by from_id
    agg <- edges_with_val[!is.na(val),
                          .(nb_max  = max(val),
                            nb_min  = min(val),
                            nb_mean = mean(val)),
                          by = .(from_id)]
    
    # Write results back into cell_dt for this year
    # Find row indices in cell_dt for this year
    idx_dt <- cell_dt[year == yr, .(id, .I)]
    setnames(idx_dt, ".I", "row_idx")
    setkey(idx_dt, id)
    
    # Join agg onto idx_dt
    idx_dt[agg, on = .(id = from_id), 
           `:=`(nb_max = i.nb_max, nb_min = i.nb_min, nb_mean = i.nb_mean)]
    
    # Update cell_dt in place using row indices
    rows <- idx_dt[!is.na(nb_max)]
    if (nrow(rows) > 0) {
      set(cell_dt, i = rows$row_idx, j = max_col,  value = rows$nb_max)
      set(cell_dt, i = rows$row_idx, j = min_col,  value = rows$nb_min)
      set(cell_dt, i = rows$row_idx, j = mean_col, value = rows$nb_mean)
    }
  }
  
  invisible(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Run it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)

# Convert nb object to edge list (once)
edge_dt <- nb_to_edge_dt(id_order, rook_neighbors_unique)

# Convert to data.table if needed
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor features
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "...\n")
  t0 <- proc.time()
  compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  elapsed <- (proc.time() - t0)[3]
  cat("  Done in", round(elapsed, 1), "seconds.\n")
}

# Convert back to data.frame if downstream code expects it
# cell_data <- as.data.frame(cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup build** | 6.46M `lapply` iterations with string ops | One-time `nb` â†’ edge-list conversion (~1 sec) |
| **Stats computation** | 5 Ã— 6.46M `lapply` iterations | 5 vars Ã— 28 years = 140 vectorized `data.table` aggregations |
| **Peak memory** | ~6.46M-element list of integer vectors | ~1.37M-row edge `data.table` + one year slice at a time |
| **Estimated runtime** | 86+ hours | **~2â€“5 minutes total** |
| **Numerical equivalence** | Baseline | **Identical** â€” same neighbors, same max/min/mean |
| **Trained RF model** | â€” | **Untouched** â€” only feature columns are recomputed with identical values |

The key insight is replacing millions of interpreted R loop iterations with vectorized `data.table` keyed joins and grouped aggregations, while processing one year at a time to stay within 16 GB RAM.
END_RESPONSE>>>
