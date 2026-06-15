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
run_id: Anthropic_F02_T00_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with millions of keys creates enormous overhead. The `compute_neighbor_stats` function then iterates over the same 6.46 million rows again, once per variable (Ã—5 variables). Specifically:

1. **`build_neighbor_lookup`**: Creates ~6.46M character keys for `idx_lookup`, then for each of the 6.46M rows, pastes neighbor IDs with the year, and indexes into the named vector. This is ~6.46M Ã— avg_neighbors string operations and hash lookups. The `lapply` loop is interpreted R â€” no vectorization, no parallelism.

2. **`compute_neighbor_stats`**: Iterates 6.46M list elements per variable (Ã—5 = ~32.3M iterations), each time subsetting a numeric vector by indices, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is also slow.

3. **Memory**: Storing a 6.46M-element list of integer vectors (`neighbor_lookup`) plus the full data frame with 110+ columns is feasible in 16 GB, but the intermediate character vectors and the `do.call(rbind, ...)` on millions of small vectors create memory pressure and GC thrashing.

**Estimated cost**: The `build_neighbor_lookup` alone does ~6.46M Ã— ~4 (avg rook neighbors) = ~25.8M string paste + lookup operations in an interpreted loop. `compute_neighbor_stats` does ~32.3M list iterations with subsetting. Total: ~86+ hours is consistent with this analysis.

---

## Optimization Strategy

### Principle: Replace interpreted R loops with vectorized and `data.table`-based joins.

**Key insight**: The neighbor lookup is essentially a **join operation**. Each row `(id, year)` needs to be joined to its neighbors' rows `(neighbor_id, same year)` to aggregate their variable values. This is a classic relational operation that `data.table` handles in seconds.

### Steps:

1. **Expand the neighbor list into an edge table** (`data.table` with columns `id`, `neighbor_id`) â€” ~1.37M rows (directed). This is done once.

2. **Join the edge table with the data** on `(neighbor_id, year)` to pull neighbor values â€” this is a keyed `data.table` merge, highly optimized in C.

3. **Group-by aggregate** `(id, year)` to compute max, min, mean of neighbor values â€” again a native `data.table` operation.

4. **Merge results back** into the main data.

This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` entirely. No 6.46M-element list, no interpreted loop, no string pasting.

**Expected speedup**: From ~86 hours to **minutes** (the join is ~6.46M Ã— ~4 = ~25.8M rows after expansion, and `data.table` aggregates this trivially).

**Memory**: The expanded join table is ~25.8M rows Ã— a few columns â€” well within 16 GB.

**Preserves**: The trained Random Forest model is untouched. The numerical output (max, min, mean of neighbor values per cell-year) is identical.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the edge table from the nb object (one-time)
# ============================================================
# rook_neighbors_unique is a list of length 344,208 (one entry per cell).
# id_order is the vector mapping position -> cell id.
# neighbors[[i]] contains integer indices into id_order of cell i's neighbors.

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    n  <- length(nb)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb]
      pos <- pos + n
    }
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows with columns: id, neighbor_id

# ============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ============================================================
cell_dt <- as.data.table(cell_data)

# Ensure id and year are keyed for fast joins
# We'll create a minimal lookup table for neighbor values per variable

# ============================================================
# STEP 3: Compute neighbor features for all variables at once
# ============================================================
compute_all_neighbor_features <- function(cell_dt, edge_dt, neighbor_source_vars) {
  
  # Columns needed from the data for the neighbor lookup:
  # id, year, and each of the neighbor_source_vars
  lookup_cols <- c("id", "year", neighbor_source_vars)
  
  # Minimal table for joining: neighbor values keyed by (id, year)
  # This is the table we join TO (looking up neighbor_id's values)
  val_dt <- cell_dt[, ..lookup_cols]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)
  
  # Expand edges by year:
  # For each (id, year) row, we need (id, neighbor_id, year).
  # Instead of a massive cross join, we join edge_dt to the distinct
  # (id, year) pairs, then look up neighbor values.
  
  # Get distinct (id, year) pairs
  id_year <- cell_dt[, .(id, year)]
  setkey(id_year, id)
  
  # Join: for each (id, year), attach all neighbors
  # Result: (id, year, neighbor_id) â€” ~25.8M rows
  setkey(edge_dt, id)
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  # Now join to get neighbor variable values
  setkey(expanded, neighbor_id, year)
  expanded <- val_dt[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has: neighbor_id, year, ntl, ec, ..., id
  
  # Aggregate: for each (id, year), compute max/min/mean of each variable
  agg_exprs <- list()
  for (var in neighbor_source_vars) {
    var_sym <- as.name(var)
    agg_exprs[[paste0("neighbor_max_", var)]]  <- bquote(
      as.numeric(max(.(var_sym), na.rm = TRUE))
    )
    agg_exprs[[paste0("neighbor_min_", var)]]  <- bquote(
      as.numeric(min(.(var_sym), na.rm = TRUE))
    )
    agg_exprs[[paste0("neighbor_mean_", var)]] <- bquote(
      mean(.(var_sym), na.rm = TRUE)
    )
  }
  
  # Build the aggregation call
  agg_list <- as.call(c(as.name("list"), agg_exprs))
  
  result <- expanded[, eval(agg_list), by = .(id, year)]
  
  # Replace -Inf/Inf from max/min on all-NA groups with NA
  num_cols <- setdiff(names(result), c("id", "year"))
  for (col in num_cols) {
    set(result, which(is.infinite(result[[col]])), col, NA_real_)
  }
  
  return(result)
}

neighbor_features <- compute_all_neighbor_features(
  cell_dt, edge_dt,
  c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# ============================================================
# STEP 4: Merge neighbor features back into cell_data
# ============================================================
# If cell_dt already has old neighbor columns, remove them first
old_neighbor_cols <- grep("^neighbor_", names(cell_dt), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_dt[, (old_neighbor_cols) := NULL]
}

cell_dt <- merge(cell_dt, neighbor_features, by = c("id", "year"), all.x = TRUE)

# Convert back to data.frame if the downstream RF predict() expects one
cell_data <- as.data.frame(cell_dt)

# ============================================================
# STEP 5: Predict with the existing trained Random Forest
# ============================================================
# The trained model object (e.g., rf_model) is unchanged.
# Predictions proceed as before:
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M interpreted R loop iterations with string paste + named vector lookup | Single `data.table` keyed join (C-level) |
| **Neighbor stats** | 6.46M Ã— 5 = 32.3M `lapply` iterations + `do.call(rbind, ...)` | Single grouped aggregation in `data.table` |
| **Estimated runtime** | ~86+ hours | **~2â€“10 minutes** (depending on disk I/O) |
| **Peak memory** | High due to 6.46M-element list + character vectors + GC pressure | ~25.8M-row intermediate table â‰ˆ 1â€“2 GB |
| **Numerical output** | max, min, mean per (id, year) per variable | **Identical** â€” same aggregation functions |
| **Trained RF model** | â€” | **Untouched** â€” only the feature preparation changes |

The key transformation is recognizing that the neighbor lookup + aggregation is a **join-then-group-by** operation, which is exactly what `data.table` is optimized for at the C level, eliminating millions of interpreted R loop iterations entirely.
END_RESPONSE>>>
