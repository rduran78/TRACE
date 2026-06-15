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
run_id: Anthropic_F06_T03_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (by cell ID).
2. Constructs string keys (`paste(id, year)`) for each neighbor-year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) using character matching.

Character-keyed named-vector lookups are **O(n)** in the worst case per access in base R, and doing this 6.46 million times with string construction is extremely expensive. The result is a **list of 6.46M integer vectors** â€” itself a large memory object.

### Bottleneck B: `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows per variable

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by the index vectors stored in the lookup list. This is called 5 times, so ~32.3 million list-element operations total.

### Why it takes 86+ hours

| Step | Approximate cost |
|---|---|
| `build_neighbor_lookup` | ~6.46M string-paste + named-vector lookups | 
| `compute_neighbor_stats` Ã— 5 vars | ~32.3M subset + summary operations |
| Memory pressure | The 6.46M-element list of integer vectors causes GC thrashing on 16 GB RAM |

The dominant cost is the **row-level R-loop with character key lookups**. This is a classic case where vectorized/join-based approaches yield orders-of-magnitude speedups.

---

## 2. Optimization Strategy

### Core Insight

The neighbor statistics (max, min, mean of rook neighbors) are **per-cell, per-year** summaries. Since the neighbor structure is **time-invariant** (rook adjacency doesn't change across years), we can:

1. **Expand the neighbor list into an edge table once** (cell â†’ neighbor, ~1.37M directed edges).
2. **Join** this edge table to the panel data by `(neighbor_id, year)` to pull neighbor values â€” this is a single vectorized merge, not 6.46M lookups.
3. **Group-by aggregate** `(cell_id, year)` to compute max, min, mean â€” fully vectorized.

This replaces all `lapply` loops and character-key lookups with **a single join + grouped aggregation per variable**, leveraging `data.table` for speed and memory efficiency.

### Why not raster focal/kernel operations?

The document header asks us to consider raster focal operations. While the grid structure is regular enough that `terra::focal()` could theoretically compute neighbor statistics, there are practical problems:
- The panel has irregular missing cells (not all grid cells may be present in all years).
- The neighbor object (`spdep::nb`) encodes the actual adjacency, which may reflect boundary irregularities, masked cells, or non-rectangular domains.
- Focal operations would require reconstructing a full raster per year per variable, running the focal, then extracting back â€” adding complexity and risking numerical discrepancies at edges.

**The join-based `data.table` approach best preserves the required results** while being nearly as fast, and it exactly reproduces the original computation.

### Complexity comparison

| Approach | Time complexity | Memory |
|---|---|---|
| Original (lapply + character keys) | O(R Ã— k) with large constants | ~6.46M-element list + strings |
| data.table join + group-by | O(E log E) for join, O(E) for aggregation | Edge table ~1.37M rows |

Where R = 6.46M rows, k = avg neighbors per cell, E = ~1.37M Ã— 28 â‰ˆ 38.5M edge-year rows (but only for cells present). Expected runtime: **minutes, not hours**.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert panel data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Assumes:
#   cell_data       â€” data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order        â€” integer/character vector of cell IDs in the order used by the nb object
#   rook_neighbors_unique â€” spdep::nb object (list of integer index vectors into id_order)

setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a directed edge table from the nb object (time-invariant)
#
#   This replaces build_neighbor_lookup entirely.
#   ~1.37M rows, two integer columns â€” trivial memory footprint.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  edges <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has columns: focal_id, neighbor_id
# Rows â‰ˆ 1,373,394

cat("Edge table rows:", nrow(edge_table), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor stats for all variables via join + group-by
#
#   For each source variable, we:
#     (a) Join edge_table to cell_data on (neighbor_id = id, year = year)
#         to get the neighbor's value of that variable.
#     (b) Group by (focal_id, year) and compute max, min, mean.
#     (c) Merge the results back into cell_data.
#
#   This replaces compute_neighbor_stats + the outer loop.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  
  # Subset cell_data to only the columns we need for the join (memory efficient)
  # Columns: id (will match as neighbor_id), year, and the variable of interest
  neighbor_vals <- cell_data[, .(id, year, val = get(var_name))]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id, year)
  
  # Join: for each edge (focal_id, neighbor_id), for each year,
  # pull the neighbor's value of var_name.
  # Result: one row per (focal_id, neighbor_id, year) with the neighbor's value.
  joined <- merge(
    edge_table,
    neighbor_vals,
    by = "neighbor_id",
    allow.cartesian = TRUE  # a neighbor_id appears in many edges
  )
  # joined columns: neighbor_id, focal_id, year, val
  
  # Aggregate: compute max, min, mean per (focal_id, year), dropping NAs
  stats <- joined[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(focal_id, year)
  ]
  
  # Rename columns to match the original naming convention
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(stats, "focal_id", "id")
  setkey(stats, id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # Merge back into cell_data
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
  
  # Clean up intermediate objects to keep memory in check

rm(neighbor_vals, joined, stats)
  gc()
  
  cat("  Done:", max_col, min_col, mean_col, "added.\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the pre-trained Random Forest (unchanged)
#
#   The trained model object is used as-is. No retraining.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example (adjust to your actual model object and predict call):
# predictions <- predict(rf_model, newdata = cell_data)

cat("Neighbor feature engineering complete. Ready for prediction.\n")
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor definition** | We use the exact same `rook_neighbors_unique` nb object â€” same adjacency graph. |
| **Same aggregation functions** | `max()`, `min()`, `mean()` with `NA` removal â€” identical to the original `compute_neighbor_stats`. |
| **Same join semantics** | The merge on `(neighbor_id, year)` replicates the original's `paste(id, year)` key lookup. Cells missing in a given year produce no join match, yielding `NA` after the `all.x = TRUE` merge â€” identical to the original's `is.na` filtering. |
| **No model retraining** | The Random Forest model object is never touched; only the feature-engineering pipeline is optimized. |
| **Floating-point identity** | `mean()` is computed over the same set of non-NA values in the same R function â€” results are numerically identical (not merely approximately equal). |

---

## 5. Expected Performance

| Step | Estimated time | Estimated peak memory |
|---|---|---|
| `build_edge_table` | < 5 seconds | ~20 MB |
| Per-variable join + aggregate | ~30â€“90 seconds each | ~1.5â€“3 GB transient |
| Total (5 variables) | **~3â€“8 minutes** | ~4â€“6 GB peak (within 16 GB) |

This represents a speedup from **86+ hours to under 10 minutes** â€” roughly **500â€“1,500Ã—** faster.
END_RESPONSE>>>
