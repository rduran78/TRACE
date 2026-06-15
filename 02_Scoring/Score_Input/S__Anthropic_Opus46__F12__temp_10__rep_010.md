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
run_id: Anthropic_F12_T10_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially correct but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` â€” Colleague's Claim
`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed costly (it dispatches `rbind` iteratively). And the per-element `lapply` does redundant work. **But this is a secondary bottleneck**, not the primary one. With 5 variables, this function runs 5 times. Even naively, binding 6.46M short vectors with `do.call(rbind, ...)` takes minutes, not hours.

### `build_neighbor_lookup()` â€” The True Primary Bottleneck
The real killer is `build_neighbor_lookup()`. It runs an `lapply` over **every one of the ~6.46 million rows**, and inside each iteration it:

1. **Calls `as.character()` and does a named-vector lookup** (`id_to_ref[as.character(data$id[i])]`) â€” 6.46M character coercions + name-matching lookups.
2. **Constructs `paste()` keys for every neighbor of every row** â€” ~6.46M calls to `paste(..., sep="_")`, each producing a small character vector.
3. **Does named-vector indexing** on `idx_lookup` (a named vector of length 6.46M) â€” this is **O(n) per lookup** via linear hashing in R's named vectors, repeated for every neighbor of every row.

The `idx_lookup` named vector has 6.46 million entries. Named-vector lookup in R uses internal hashing, but constructing and querying it millions of times with `paste`-generated keys is extremely slow. With ~1.37 million neighbor relationships spread across 344K cells and 28 years, the total number of neighbor-key lookups is roughly **6.46M Ã— avg_neighbors â‰ˆ 6.46M Ã— 4 â‰ˆ 25.8 million** string-match lookups against a 6.46M-entry named vector. This is the **dominant cost**, easily accounting for the 86+ hour estimate.

**Verdict: REJECT the colleague's diagnosis.** The main bottleneck is `build_neighbor_lookup()` â€” specifically, the per-row string construction (`paste`) and repeated named-vector lookups against a 6.46M-entry index. The `do.call(rbind, ...)` in `compute_neighbor_stats` is a minor secondary issue.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()` entirely** â€” eliminate the row-level `lapply`. Use integer-indexed matching via `data.table` or `match()` on integer-encoded keys instead of character `paste` keys. Pre-expand the neighbor list at the cell level (344K entries), then join on year via vectorized operations.

2. **Vectorize `compute_neighbor_stats()`** â€” replace `lapply` + `do.call(rbind, ...)` with grouped vectorized aggregation using `data.table`.

3. **Preserve**: the trained Random Forest model (no retraining), all original numerical outputs (same estimand â€” max, min, mean of neighbor values).

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED build_neighbor_lookup (vectorized, no per-row lapply)
# ==============================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table for fast operations
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # Build cell-level neighbor edge list (directed):
  # For each cell index in id_order, list its neighbor cell IDs.
  # neighbors is an nb object: neighbors[[i]] gives integer indices
  # into id_order for the neighbors of id_order[i].
  
  n_cells <- length(id_order)
  
  # Expand neighbor list into an edge table: (focal_cell_id, neighbor_cell_id)
  # Use integer cell IDs from id_order
  from_idx <- rep(seq_len(n_cells), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  
  edge_dt <- data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  
  # Get unique years
  years <- sort(unique(dt$year))
  
  # Cross-join edges with years: each edge applies to every year
  edge_year <- edge_dt[, CJ(focal_id = focal_id, year = years), 
                        by = .(neighbor_id)]
  # The above is wrong for large data â€” let's do it properly:
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year[, focal_id    := edge_dt$focal_id[edge_idx]]
  edge_year[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
  edge_year[, edge_idx := NULL]
  
  # Map focal (id, year) -> row index in original data
  setkey(dt, id, year)
  
  # Map neighbor (id, year) -> row index in original data
  # focal row index
  edge_year <- merge(
    edge_year,
    dt[, .(id, year, focal_row = row_idx)],
    by.x = c("focal_id", "year"),
    by.y = c("id", "year"),
    all.x = FALSE  # inner join: drop edges where focal cell-year doesn't exist
  )
  
  # neighbor row index
  edge_year <- merge(
    edge_year,
    dt[, .(id, year, neighbor_row = row_idx)],
    by.x = c("neighbor_id", "year"),
    by.y = c("id", "year"),
    all.x = FALSE  # drop edges where neighbor cell-year doesn't exist
  )
  
  # Return the edge table â€” this replaces the list-of-vectors lookup
  # Columns: focal_row, neighbor_row (both are integer indices into data)
  edge_year[, .(focal_row, neighbor_row)]
}

# ==============================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table)
# ==============================================================

compute_neighbor_stats_fast <- function(data, edge_table, var_name) {
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  n <- nrow(dt)
  
  # Extract neighbor values
  work <- copy(edge_table)
  work[, val := dt[[var_name]][neighbor_row]]
  
  # Remove NAs
  work <- work[!is.na(val)]
  
  # Grouped aggregation
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]
  
  # Initialize result columns with NA
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)
  
  max_col[agg$focal_row]  <- agg$nb_max
  min_col[agg$focal_row]  <- agg$nb_min
  mean_col[agg$focal_row] <- agg$nb_mean
  
  # Return as matrix to match original interface (n x 3)
  cbind(max_col, min_col, mean_col)
}

# ==============================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, edge_table) {
  stats <- compute_neighbor_stats_fast(data, edge_table, var_name)
  
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  
  data
}

# ==============================================================
# OPTIMIZED Outer Pipeline
# ==============================================================

# Build edge table ONCE (replaces neighbor_lookup list)
message("Building vectorized neighbor edge table...")
edge_table <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, edge_table
  )
}

message("Done. Proceed to prediction with the existing trained Random Forest model.")

# The trained RF model is untouched â€” use predict() as before:
# predictions <- predict(rf_model, cell_data)
```

---

## Complexity Comparison

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | O(N Ã— k) string ops + named-vector lookups on 6.46M entries; ~6.46M `paste` + `match` calls | One vectorized edge expansion + two `data.table` keyed merges. O(E Ã— T) with hash joins. |
| `compute_neighbor_stats` (per var) | 6.46M `lapply` iterations + `do.call(rbind, 6.46M)` | Single `data.table` grouped aggregation over ~25.8M edge rows |
| **Estimated total time** | **86+ hours** | **~5â€“20 minutes** (dominated by the merge and grouped aggregations) |
| Memory | Millions of small character vectors + 6.46M-element list | One ~25.8M-row integer edge table (~400 MB) â€” fits in 16 GB |
| Trained RF model | Preserved âœ“ | Preserved âœ“ |
| Numerical outputs | max/min/mean | Identical max/min/mean âœ“ |

### Key insight

The 86-hour runtime was overwhelmingly caused by `build_neighbor_lookup()` performing **~6.46 million iterations** of string construction and named-vector lookups against a 6.46M-entry lookup table â€” not by `do.call(rbind, ...)`. The fix is to replace the row-level `lapply` with a single vectorized edge-table construction via `data.table` keyed joins, and to replace the per-row aggregation with `data.table` grouped operations.
END_RESPONSE>>>
