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
run_id: Anthropic_F05_T00_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` â€” **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **allocates a new character vector per row** (~4 neighbor keys Ã— 6.46M rows â‰ˆ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` â€” **named-vector lookup via string matching on a 6.46M-element named vector**, repeated 6.46M times.

The `idx_lookup` named vector has 6.46M entries. R's named-vector lookup is **O(N)** per query (linear scan or hashing with overhead). Doing this 6.46M times with ~4 keys each yields roughly **25.8 billion character comparisons** in the worst case. This is why the pipeline takes 86+ hours.

### The Broader Pattern

The entire design conflates two independent structures:

- **Spatial topology** (which cells neighbor which) â€” fixed across years.
- **Temporal alignment** (same year) â€” trivially handled by grouping.

By encoding both into a single string key (`id_year`), the code forces an expensive join at every row. The neighbor relationships are **identical across all 28 years**, so the lookup should be built **once on cell IDs** and then broadcast across years.

### `compute_neighbor_stats` Is Efficient *Given* the Lookup

Once `neighbor_lookup` (a list of integer row indices) is built, `compute_neighbor_stats` is a simple vectorized operation. The bottleneck is entirely in `build_neighbor_lookup`.

---

## Optimization Strategy

1. **Separate spatial topology from temporal alignment.** Build a cell-ID â†’ neighbor-cell-IDs mapping once (344K entries, not 6.46M).
2. **Use `data.table` for fast equi-joins.** Instead of string-key named-vector lookups, use integer-keyed joins.
3. **Vectorize the neighbor-stats computation.** Expand the neighbor relationships into an edge table, join variable values, and compute grouped aggregates â€” all vectorized, zero R-level loops.
4. **Process all 5 variables in one pass** over the edge table rather than 5 separate passes.

### Complexity Comparison

| Step | Original | Optimized |
|---|---|---|
| Build lookup | O(R Ã— K Ã— N) string ops (R=rows, K=avg neighbors, N=lookup size) | O(E) integer join (E=edgesÃ—years) |
| Compute stats | O(R Ã— K) â€” already fine | O(E) vectorized grouped aggregation |
| Total string allocs | ~25.8M | 0 |

Expected speedup: **~1000Ã—** or more (minutes instead of days).

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED PIPELINE â€” drop-in replacement
# Preserves the exact same numerical output columns.
# Preserves the trained Random Forest model (no retraining).
# ============================================================

build_neighbor_edge_table <- function(id_order, neighbors) {

  # Build a data.table of directed spatial edges: focal_id -> neighbor_id

  # This is year-independent (topology is fixed).
  # neighbors is an nb object (list of integer index vectors into id_order).

  focal_indices <- which(lengths(neighbors) > 0)

  focal_ids <- rep(id_order[focal_indices], times = lengths(neighbors[focal_indices]))
  neighbor_ids <- id_order[unlist(neighbors[focal_indices])]

  data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
}

compute_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                          neighbor_source_vars) {
  # cell_data: data.frame or data.table with columns id, year, and the source vars
  # id_order: integer vector of cell IDs matching the nb object indexing

  # neighbors: spdep nb object (list of integer neighbor indices)
  # neighbor_source_vars: character vector of variable names

  dt <- as.data.table(cell_data)

  # 1. Build spatial edge table (year-independent): ~1.37M rows
  edges <- build_neighbor_edge_table(id_order, neighbors)

  # 2. Cross with years to get full edge table: ~1.37M Ã— 28 â‰ˆ 38.5M rows
  #    But we only need edges where both focal and neighbor exist in the data.
  #    Instead of a cross-join, we join through the data itself.

  # Create a row-key table: (id, year) -> row index + variable values
  dt[, row_idx := .I]

  # Subset to only the columns we need for the join
  value_cols <- neighbor_source_vars
  key_cols <- c("id", "year", value_cols)
  dt_key <- dt[, ..key_cols]

  # 3. For each focal row, find its neighbors in the same year.
  #    Join edges to focal rows to get (focal_id, year, neighbor_id),
  #    then join to neighbor rows to get neighbor variable values.

  # Step A: Join focal rows to edges on focal_id
  #   Result: each focal (id, year) is expanded to its neighbors
  setnames(edges, c("focal_id", "neighbor_id"))

  # Focal side: get (focal_id, year) pairs
  focal_dt <- dt[, .(focal_id = id, year)]

  # Merge: focal_dt Ã— edges on focal_id â†’ (focal_id, year, neighbor_id)
  # This is the big expansion: ~6.46M rows Ã— ~4 neighbors = ~25.8M rows
  setkey(edges, focal_id)
  setkey(focal_dt, focal_id)
  expanded <- edges[focal_dt, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, year

  # Step B: Join neighbor values on (neighbor_id, year)
  setnames(dt_key, "id", "neighbor_id")
  setkey(dt_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  merged <- dt_key[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # merged has: neighbor_id, year, <value_cols>, focal_id

  # 4. Compute grouped stats: max, min, mean per (focal_id, year) per variable
  #    We do all variables in one grouped operation.

  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in value_cols) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]] <- substitute(
      as.numeric(max(x[!is.na(x)], na.rm = FALSE)),
      list(x = v_sym)
    )
    agg_exprs[[paste0("n_min_", v)]] <- substitute(
      as.numeric(min(x[!is.na(x)], na.rm = FALSE)),
      list(x = v_sym)
    )
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(
      as.numeric(mean(x[!is.na(x)], na.rm = FALSE)),
      list(x = v_sym)
    )
  }

  # Custom aggregation that handles all-NA groups correctly (return NA)
  # We use a single lapply-based aggregation for clarity and correctness.
  stat_fn <- function(vals) {
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) return(list(NA_real_, NA_real_, NA_real_))
    list(max(vals), min(vals), mean(vals))
  }

  # Aggregate all variables
  result_list <- vector("list", length(value_cols))
  names(result_list) <- value_cols

  for (v in value_cols) {
    cat("Computing neighbor stats for:", v, "\n")
    agg <- merged[, {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(n_max = NA_real_, n_min = NA_real_, n_mean = NA_real_)
      } else {
        list(n_max = max(vals), n_min = min(vals), n_mean = mean(vals))
      }
    }, by = .(focal_id, year)]

    setnames(agg, c("n_max", "n_min", "n_mean"),
             paste0(c("n_max_", "n_min_", "n_mean_"), v))
    result_list[[v]] <- agg
  }

  # 5. Merge all results back to the original data
  #    Successive joins on (focal_id = id, year)
  for (v in value_cols) {
    agg <- result_list[[v]]
    setnames(agg, "focal_id", "id")
    dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)
  }

  # 6. Handle rows with no neighbors (isolated cells): they get NA, which is correct.

  # Clean up helper column
  dt[, row_idx := NULL]

  return(dt)
}

# ============================================================
# USAGE â€” replaces the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now contains the same columns as before:
#   n_max_ntl, n_min_ntl, n_mean_ntl,
#   n_max_ec,  n_min_ec,  n_mean_ec,
#   ... etc.
#
# The trained Random Forest model can be used directly for prediction
# on this data â€” no retraining needed.
```

---

## Memory Considerations (16 GB Laptop)

| Object | Estimated Size |
|---|---|
| `dt` (cell_data as data.table) | ~5.7 GB (6.46M Ã— 110 cols) |
| `edges` (spatial) | ~10.5 MB (1.37M Ã— 2 int cols) |
| `expanded` (edges Ã— years) | ~580 MB (38.4M Ã— 3 cols) |
| `merged` (expanded + 5 value cols) | ~2.3 GB (38.4M Ã— 8 cols) |
| Per-variable `agg` | ~150 MB each |

**Peak: ~10 GB** â€” fits in 16 GB with headroom. If memory is tight, process variables one at a time and drop the `merged` column after each aggregation:

```r
# Memory-conservative variant: process one variable at a time
for (v in value_cols) {
  cat("Processing:", v, "\n")

  # Minimal merge: only bring the one variable we need
  dt_one <- dt[, .(neighbor_id = id, year, val = get(v))]
  setkey(dt_one, neighbor_id, year)

  merged_one <- dt_one[expanded, on = c("neighbor_id", "year"), nomatch = NA]

  agg <- merged_one[, {
    vals <- val[!is.na(val)]
    if (length(vals) == 0L) {
      list(n_max = NA_real_, n_min = NA_real_, n_mean = NA_real_)
    } else {
      list(n_max = max(vals), n_min = min(vals), n_mean = vals))
    }
  }, by = .(focal_id, year)]

  setnames(agg, c("n_max", "n_min", "n_mean"),
           paste0(c("n_max_", "n_min_", "n_mean_"), v))
  setnames(agg, "focal_id", "id")

  dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)
  rm(merged_one, agg, dt_one); gc()
}
```

*(Note: there's a typo in the memory-conservative snippet above â€” `mean(vals))` has an extra paren. Corrected version below.)*

```r
# Corrected memory-conservative variant
for (v in value_cols) {
  cat("Processing:", v, "\n")

  dt_one <- dt[, .(neighbor_id = id, year, val = get(v))]
  setkey(dt_one, neighbor_id, year)

  merged_one <- dt_one[expanded, on = c("neighbor_id", "year"), nomatch = NA]

  agg <- merged_one[, {
    vals <- val[!is.na(val)]
    if (length(vals) == 0L) {
      list(n_max = NA_real_, n_min = NA_real_, n_mean = NA_real_)
    } else {
      list(n_max = max(vals), n_min = min(vals), n_mean = mean(vals))
    }
  }, by = .(focal_id, year)]

  setnames(agg, c("n_max", "n_min", "n_mean"),
           paste0(c("n_max_", "n_min_", "n_mean_"), v))
  setnames(agg, "focal_id", "id")

  dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)
  rm(merged_one, agg, dt_one); gc()
}
```

## Expected Runtime

| Phase | Estimated Time |
|---|---|
| Build edge table | < 1 second |
| Expand edges Ã— focal rows | ~10 seconds |
| Merge neighbor values (per var) | ~20 seconds |
| Grouped aggregation (per var) | ~30 seconds |
| Final merge back (per var) | ~10 seconds |
| **Total (5 variables)** | **~5â€“10 minutes** |

**Down from 86+ hours to under 10 minutes** â€” roughly a **500â€“1000Ã— speedup** â€” by eliminating the row-level string-key lookups and replacing them with vectorized `data.table` integer-keyed joins.
END_RESPONSE>>>
