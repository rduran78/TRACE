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
run_id: Anthropic_F05_T03_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts the cell ID to a character and looks it up in `id_to_ref` â€” **6.46M character coercions + named-vector lookups**.
2. Extracts neighbor cell IDs from the `nb` object.
3. **Pastes** each neighbor ID with the current year (`paste(..., sep="_")`) â€” this creates ~1.37M Ã— 28 â‰ˆ **tens of millions of temporary strings**.
4. Looks up each string key in `idx_lookup` (a named vector of length 6.46M) â€” **named-vector lookup is O(N) in the worst case** because R's named vectors use linear hashing that degrades with millions of entries.

The total work is roughly: 6.46M rows Ã— (average ~4 neighbors) Ã— (1 paste + 1 named-vector probe into a 6.46M-entry table) â‰ˆ **25.8 million string constructions and hash probes**. With R's overhead on character operations and the large hash table, this alone accounts for many hours.

Then `compute_neighbor_stats` is called 5 times (once per variable), but since it reuses the integer `neighbor_lookup`, it is comparatively cheap. **The bottleneck is `build_neighbor_lookup`.**

### Why It's a Broader Algorithmic Issue

The entire string-keying scheme is unnecessary. The data has a regular panel structure (344,208 cells Ã— 28 years). This means:

- Every cell appears exactly once per year (or can be made to).
- If the data is sorted by `(year, id)` â€” or even `(id, year)` â€” the row index for any `(cell, year)` pair can be computed **arithmetically** with zero string operations.
- The neighbor relationships are **time-invariant**: cell `A`'s rook neighbors are the same in every year. So the neighbor lookup only needs to be built once at the **cell level** (344K entries), not the **cell-year level** (6.46M entries).

The current code conflates two orthogonal dimensions (spatial adjacency and temporal indexing) into a single flat string-key lookup, which is the root cause.

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Index scheme | 6.46M-entry named character vector | Arithmetic index: `(cell_position - 1) Ã— T + year_position` |
| Loop granularity | `lapply` over 6.46M rows in R | Vectorized: build integer neighbor-row matrix for all cell-years at once |
| Neighbor lookup | Per-row string paste + hash probe | Pre-expand spatial neighbors to cell-year rows via vectorized integer arithmetic |
| Stat computation | `lapply` over 6.46M lists | Vectorized split-apply using `data.table` grouped operations or matrix indexing |

**Expected speedup**: From ~86+ hours to **minutes** (the main operations become vectorized integer arithmetic and `data.table` grouped aggregations over ~25.8M neighbor-pair-year rows).

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# 
# Preserves: trained Random Forest model (untouched), original numerical 
# estimand (max, min, mean of neighbor values per variable per cell-year).
# =============================================================================

library(data.table)

build_and_compute_all_neighbor_features <- function(cell_data, 
                                                     id_order, 
                                                     rook_neighbors_unique, 
                                                     neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # 1. Convert to data.table for fast grouped operations
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # -------------------------------------------------------------------------
  # 2. Build a fast integer mapping: cell id -> position in id_order
  #    id_order is the vector of 344,208 cell IDs in the order matching

  #    the nb object.
  # -------------------------------------------------------------------------
  n_cells <- length(id_order)
  id_to_pos <- integer(0)  # We'll use a data.table join instead for safety
  
  id_map <- data.table(
    id     = id_order,
    id_pos = seq_len(n_cells)
  )
  
  # -------------------------------------------------------------------------
  # 3. Build the spatial edge list (time-invariant) from the nb object.
  #    rook_neighbors_unique[[i]] gives the neighbor indices (into id_order)
  #    for the i-th cell in id_order.
  # -------------------------------------------------------------------------
  # Pre-compute lengths for vectorized expansion
  n_neighbors <- lengths(rook_neighbors_unique)  # integer vector, length n_cells
  total_edges <- sum(n_neighbors)  # ~1,373,394 directed edges
  
  cat(sprintf("Cells: %d | Years: %d | Directed edges: %d\n",
              n_cells, length(unique(dt$year)), total_edges))
  
  # Build edge list: (focal_id_pos, neighbor_id_pos)
  focal_pos    <- rep(seq_len(n_cells), times = n_neighbors)
  neighbor_pos <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  edge_dt <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  rm(focal_pos, neighbor_pos)  # free memory
  
  # -------------------------------------------------------------------------
  # 4. Cross edges with years to get the full cell-year neighbor table.
  #    This is ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows.
  #    On 16GB RAM this is feasible (~1-2 GB for the integer columns).
  # -------------------------------------------------------------------------
  years <- sort(unique(dt$year))
  n_years <- length(years)
  
  # Instead of a full cross join (which would be large), we join through
  # the actual data. This naturally handles any missing cell-years.
  
  # Create a minimal keyed version of dt for joining
  # We need: id, year, and the source variables
  keep_cols <- c("id", "year", neighbor_source_vars)
  dt_slim <- dt[, ..keep_cols]
  
  # -------------------------------------------------------------------------
  # 5. For each focal cell-year, find neighbor values by joining.
  #    Strategy: 
  #      a) Join dt_slim with edge_dt on focal_id = id  -> gives (focal_id, year, neighbor_id)
  #      b) Join result with dt_slim on (neighbor_id, year) -> gives neighbor values
  #      c) Group by (focal_id, year) and compute max/min/mean
  # -------------------------------------------------------------------------
  
  # Step 5a: Expand edges to cell-years
  # For each row in dt_slim, attach its neighbors
  setnames(dt_slim, "id", "focal_id")
  
  # Key for fast join
  setkey(edge_dt, focal_id)
  setkey(dt_slim, focal_id)
  
  # Join: for each (focal_id, year) row, get all neighbor_ids
  # We only need focal_id, year, and neighbor_id at this stage
  cat("Expanding edges to cell-years...\n")
  
  # This join replicates each (focal_id, year) row for each neighbor
  expanded <- edge_dt[dt_slim[, .(focal_id, year)], 
                      on = "focal_id", 
                      allow.cartesian = TRUE,
                      nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, year
  # Rows: ~38.5M (total_edges Ã— n_years, minus any missing cell-years)
  
  cat(sprintf("Expanded edge-year table: %d rows\n", nrow(expanded)))
  
  # Step 5b: Attach neighbor variable values
  # Prepare a lookup copy of the original data keyed by (id, year)
  setnames(dt_slim, "focal_id", "id")
  neighbor_vals <- copy(dt_slim)
  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  cat("Joining neighbor values...\n")
  merged <- neighbor_vals[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # merged has: neighbor_id, year, focal_id, + all source variable columns
  
  rm(expanded, neighbor_vals)
  gc()
  
  # -------------------------------------------------------------------------
  # 6. Compute grouped statistics: max, min, mean per (focal_id, year, var)
  # -------------------------------------------------------------------------
  cat("Computing neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0(v, "_neighbor_max")]]  <- bquote(
      as.numeric(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0(v, "_neighbor_min")]]  <- bquote(
      as.numeric(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0(v, "_neighbor_mean")]] <- bquote(
      mean(.(v_sym), na.rm = TRUE))
  }
  
  # Handle the -Inf / Inf from max/min on all-NA groups: replace with NA
  # We'll do this after aggregation.
  
  stats <- merged[, 
    lapply(agg_exprs, eval, envir = .SD), 
    by = .(focal_id, year),
    .SDcols = neighbor_source_vars
  ]
  
  # Replace Inf/-Inf with NA (from max/min of empty-after-na.rm groups)
  for (col_name in names(stats)) {
    if (is.numeric(stats[[col_name]])) {
      set(stats, which(is.infinite(stats[[col_name]])), col_name, NA_real_)
    }
  }
  
  rm(merged)
  gc()
  
  # -------------------------------------------------------------------------
  # 7. Join statistics back to the original data
  # -------------------------------------------------------------------------
  cat("Joining results back to cell_data...\n")
  
  # Convert original cell_data to data.table if not already
  result_dt <- as.data.table(cell_data)
  setnames(stats, "focal_id", "id")
  setkey(stats, id, year)
  setkey(result_dt, id, year)
  
  # Merge
  result_dt <- stats[result_dt, on = c("id", "year")]
  
  # -------------------------------------------------------------------------
  # 8. Return as data.frame to preserve downstream compatibility
  # -------------------------------------------------------------------------
  cat("Done.\n")
  setDF(result_dt)
  return(result_dt)
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
# 
# # Original code (86+ hours):
# # neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# # for (var_name in neighbor_source_vars) {
# #   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# # }
# 
# # Optimized replacement (minutes):
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# 
# cell_data <- build_and_compute_all_neighbor_features(
#   cell_data, 
#   id_order, 
#   rook_neighbors_unique, 
#   neighbor_source_vars
# )
# 
# # The trained Random Forest model is untouched â€” just use cell_data 
# # with predict() as before. Column names match the original pattern:
# #   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean, etc.
```

## Memory Budget Check

| Object | Approximate Size |
|--------|-----------------|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `edge_dt` (1.37M Ã— 2 int cols) | ~11 MB |
| `expanded` (38.5M Ã— 3 cols: 2 int + 1 int/year) | ~460 MB |
| `merged` (38.5M Ã— 8 cols: 3 keys + 5 numeric vars) | ~2.3 GB |
| `stats` (6.46M Ã— 17 cols) | ~880 MB |
| **Peak** | **~9.4 GB** âœ“ fits in 16 GB |

## Why This Preserves the Original Estimand

The original code computes, for each cell-year row `i`:
- Finds all rook neighbors of cell `i` that exist in the same year.
- Computes `max`, `min`, `mean` of each variable across those neighbors (excluding `NA`).

The optimized code does **exactly the same join logic** â€” `(focal_id, year)` â†” `(neighbor_id, year)` â€” but executes it as a vectorized `data.table` equi-join + grouped aggregation instead of 6.46 million R-level loop iterations with string concatenation. The numerical results are identical.

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Indexing** | 6.46M-entry named character vector, string paste per row | Vectorized integer join via `data.table` |
| **Loop** | `lapply` over 6.46M rows in R | Zero R-level row loops |
| **Neighbor expansion** | Per-row: paste + hash probe | Bulk cross-join of edge list Ã— years |
| **Stat computation** | `lapply` + per-list subsetting | `data.table` grouped `max/min/mean` (C-level) |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** |
| **RF model** | Unchanged | Unchanged |
END_RESPONSE>>>
