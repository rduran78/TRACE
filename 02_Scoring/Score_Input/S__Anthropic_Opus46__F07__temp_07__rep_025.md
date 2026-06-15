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
run_id: Anthropic_F07_T07_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` â€” O(nÂ²)-scale overhead
The function iterates over **~6.46 million rows** with `lapply`, and for each row it:
- Looks up the cell's neighbor IDs (fine).
- Constructs **character key strings** via `paste()` (expensive at scale).
- Performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) â€” each lookup is O(n) in the worst case for named vectors because R's named vector indexing uses linear hashing that degrades with millions of entries.

The `idx_lookup` vector has ~6.46M entries. Doing ~6.46M lookups into it, each with multiple keys, produces billions of character-match operations. This is the primary reason the pipeline is estimated at 86+ hours.

### 2. `compute_neighbor_stats` â€” Repeated per variable but structurally fine
This function loops over 6.46M entries 5 times (once per variable). Each iteration extracts a small vector of neighbor values and computes max/min/mean. The loop itself is O(nÂ·kÌ„) where kÌ„ â‰ˆ average number of neighbors (~4 for rook). This is tolerable but still slow in pure-R `lapply`. It can be vectorized.

### 3. Memory
The `neighbor_lookup` list of 6.46M integer vectors is large but feasible in 16 GB. The real problem is speed, not memory.

---

## Optimization Strategy

| Step | Current | Optimized |
|---|---|---|
| Key construction | `paste(id, year)` character keys | Integer arithmetic: `id * 100000L + year` or use `data.table` keyed joins |
| Index lookup | Named vector (slow hash at scale) | `data.table` binary-search join â€” O(log n) per lookup |
| Neighbor lookup build | Row-by-row `lapply` over 6.46M rows | Vectorized: explode neighbor pairs into an edge table, join once for all rows |
| Neighbor stats | Row-by-row `lapply` per variable | Vectorized `data.table` grouped aggregation on the edge table |
| Number of passes | 5 separate loops | Single grouped aggregation computes all 5 variables at once |

**Expected speedup**: From ~86 hours to **~2â€“10 minutes** on a standard laptop.

**Numerical equivalence**: The aggregation functions (max, min, mean) are applied to exactly the same neighbor sets, so the estimand is preserved bit-for-bit (up to floating-point associativity of `mean`, which `data.table` computes identically for the same group).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build a vectorized edge table (one-time, ~seconds)
# ============================================================
build_edge_table <- function(cell_data_dt, id_order, rook_neighbors_unique) {


  # --- Map each cell id to its position in id_order ---
  n_cells <- length(id_order)

  # Explode the nb object into a two-column edge list (focal_pos, neighbor_pos)
  focal_pos <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  neighbor_pos <- unlist(rook_neighbors_unique)

  # Remove the 0-neighbor sentinel that spdep::nb uses (integer(0) becomes nothing via unlist,
  # but some nb objects encode "no neighbors" as 0L)
  valid <- neighbor_pos > 0L
  focal_pos <- focal_pos[valid]
  neighbor_pos <- neighbor_pos[valid]

  # Convert positions to actual cell IDs
  focal_id    <- id_order[focal_pos]
  neighbor_id <- id_order[neighbor_pos]

  # Build a small edge table of unique directed neighbor pairs (cell-level, no year yet)
  edges <- data.table(focal_id = focal_id, neighbor_id = neighbor_id)

  # --- Cross with years present in the data ---
  years <- sort(unique(cell_data_dt$year))

  # Expand edges Ã— years via cross join
  edges_yearless <- unique(edges)  # should already be unique, but be safe
  edge_year <- edges_yearless[, .(year = years), by = .(focal_id, neighbor_id)]

  return(edge_year)
}

# ============================================================
# STEP 2: Compute all neighbor stats in one vectorized pass
# ============================================================
compute_all_neighbor_stats <- function(cell_data_dt, edge_year, neighbor_source_vars) {

  # Key the main data for fast join
  setkeyv(cell_data_dt, c("id", "year"))

  # --- Attach neighbor values via join ---
  # Join edge table to cell_data to get neighbor-row values
  # We join on (neighbor_id, year) -> (id, year) in cell_data
  neighbor_vals <- edge_year[
    cell_data_dt,
    on = .(neighbor_id = id, year = year),
    # Select only the columns we need
    mget(c("focal_id", "year", neighbor_source_vars)),
    nomatch = NULL,
    allow.cartesian = TRUE
  ]

  # --- Aggregate: group by (focal_id, year), compute max/min/mean per variable ---
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  stats <- neighbor_vals[,
    lapply(agg_exprs, eval, envir = .SD),
    by = .(focal_id, year),
    .SDcols = neighbor_source_vars
  ]

  # Handle Inf/-Inf from max/min of all-NA groups (shouldn't happen if nomatch=NULL, but be safe)
  for (col in agg_names) {
    vals <- stats[[col]]
    set(stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
  }

  return(stats)
}

# ============================================================
# STEP 3: Main pipeline
# ============================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  # Convert to data.table if needed (by reference â€” no copy)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  message("Building edge table...")
  edge_year <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge-year rows: %s", format(nrow(edge_year), big.mark = ",")))

  message("Computing neighbor statistics (vectorized)...")
  stats <- compute_all_neighbor_stats(cell_data, edge_year, neighbor_source_vars)
  message(sprintf("  Stats rows: %s", format(nrow(stats), big.mark = ",")))

  # --- Merge back into cell_data ---
  # Drop any pre-existing neighbor columns to avoid duplication
  neighbor_cols <- grep("^neighbor_(max|min|mean)_", names(cell_data), value = TRUE)
  if (length(neighbor_cols) > 0) {
    cell_data[, (neighbor_cols) := NULL]
  }

  cell_data <- merge(
    cell_data,
    stats,
    by.x = c("id", "year"),
    by.y = c("focal_id", "year"),
    all.x = TRUE,
    sort = FALSE
  )

  message("Done. Neighbor features added.")
  return(cell_data)
}

# ============================================================
# USAGE (drop-in replacement for the original outer loop)
# ============================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then predict with the already-trained Random Forest as before:
# predictions <- predict(trained_rf_model, cell_data)
```

---

## Why This Preserves Everything

| Concern | Guarantee |
|---|---|
| **Trained RF model** | No retraining. The code only adds the same feature columns the model expects. |
| **Numerical estimand** | The same `max`, `min`, `mean` are computed over exactly the same neighbor sets (rook-contiguous cells sharing the same year). `na.rm = TRUE` mirrors the original `neighbor_vals[!is.na(...)]` filtering. |
| **Irregular topology** | The edge table is derived directly from `rook_neighbors_unique` (the `spdep::nb` object), so masked cells, irregular boundaries, and missing neighbors are handled identically to the original. |
| **Column names** | Output columns follow the pattern `neighbor_{max,min,mean}_{var}`, matching what `compute_and_add_neighbor_features` would have produced. Adjust the naming pattern if your trained model expects a different convention. |

## Performance Estimate

| Component | Time (est.) |
|---|---|
| Edge table construction | ~5â€“15 seconds |
| Cross join with 28 years | ~10â€“30 seconds |
| Single `data.table` grouped aggregation | ~1â€“5 minutes |
| Merge back | ~10â€“30 seconds |
| **Total** | **~2â€“7 minutes** |

This replaces the 86+ hour pipeline with a fully vectorized `data.table` solution that avoids all per-row character operations and named-vector lookups.
END_RESPONSE>>>
