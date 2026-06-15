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
run_id: Anthropic_F04_T03_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for every neighbor of every row.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookups on character keys are **O(n)** or poorly-cached hash operations in base R. With ~6.46M rows Ã— ~4 neighbors each â‰ˆ **~26 million string constructions and hash lookups**, all inside a sequential `lapply`, this is extremely slow.

`compute_neighbor_stats` is a secondary bottleneck: another `lapply` over 6.46M elements, each allocating small vectors and calling `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list is also costly.

**Summary of root causes:**

| Cause | Location | Impact |
|---|---|---|
| Per-row `paste()` key construction | `build_neighbor_lookup` | ~26M string allocs |
| Named character vector lookup (hash on strings) | `build_neighbor_lookup` | Slow hashing Ã—26M |
| Row-wise `lapply` over 6.46M rows | Both functions | No vectorization |
| `do.call(rbind, list_of_6.46M)` | `compute_neighbor_stats` | Massive listâ†’matrix coercion |
| Repeated per-variable overhead | Outer loop | 5Ã— the stats computation |

## Optimization Strategy

**Core idea:** Replace the row-level `lapply` with a fully vectorized `data.table` merge-and-group-by approach.

1. **Build a neighbor edge table once** â€” a two-column `data.table` mapping each `(cell_id)` to its `(neighbor_cell_id)`. This is small (~1.37M rows).
2. **Join the edge table to the panel data by `(cell_id, year)`** to produce an expanded table where each row is a `(cell_id, year, neighbor_cell_id)` tuple, then join again to get the neighbor's variable value. This replaces all string-key construction and lookup.
3. **Group-by `(cell_id, year)`** to compute `max`, `min`, `mean` in one vectorized pass per variable (or all variables at once).
4. **Join the aggregated stats back** to the original data.

This eliminates all per-row R-level iteration, all `paste` key construction, and all named-vector lookups. Expected runtime: **minutes, not days**.

## Working R Code

```r
library(data.table)

# â”€â”€ Step 1: Build the neighbor edge table (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

build_neighbor_edges <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors_nb))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }
  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edges_dt <- build_neighbor_edges(id_order, rook_neighbors_unique)

# â”€â”€ Step 2: Convert panel data to data.table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure key columns exist and are properly typed
stopifnot(all(c("id", "year") %in% names(cell_dt)))

# â”€â”€ Step 3: Compute neighbor stats for all variables at once â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset the neighbor value table: only the columns we need from the panel
# We will join edges Ã— years to the panel to get neighbor values.

# Create a keyed lookup of (id, year) -> variable values
value_cols <- c("id", "year", neighbor_source_vars)
values_dt <- cell_dt[, ..value_cols]
setnames(values_dt, "id", "neighbor_id")
setkey(values_dt, neighbor_id, year)

# Expand edges by year: each edge applies to every year the focal cell has data.
# Instead of a full cross join (expensive), we merge edges onto the panel.
# For each row in the panel, find its neighbors, then look up neighbor values.

# Step 3a: Map each (cell_id, year) to its neighbors
#   panel_edges = cell_dt[, .(id, year)] joined to edges_dt on cell_id
panel_keys <- cell_dt[, .(cell_id = id, year)]
setkey(edges_dt, cell_id)
setkey(panel_keys, cell_id)

# This is the big join: ~6.46M rows Ã— ~4 neighbors = ~26M rows
# data.table handles this efficiently via binary merge
panel_edges <- edges_dt[panel_keys, on = "cell_id", allow.cartesian = TRUE, nomatch = NA]
# Result columns: cell_id, neighbor_id, year

# Step 3b: Look up neighbor variable values
setkey(panel_edges, neighbor_id, year)
panel_edges <- values_dt[panel_edges, on = .(neighbor_id, year)]
# Now panel_edges has: neighbor_id, year, ntl, ec, ..., cell_id

# Step 3c: Aggregate by (cell_id, year)
# Compute max, min, mean for each variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call dynamically
agg_list <- setNames(agg_exprs, agg_names)

# Use a single grouped aggregation
neighbor_stats <- panel_edges[
  !is.na(neighbor_id),
  {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_max_", v)]]  <- NA_real_
        out[[paste0("neighbor_min_", v)]]  <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_", v)]]  <- max(vals)
        out[[paste0("neighbor_min_", v)]]  <- min(vals)
        out[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    out
  },
  by = .(cell_id, year)
]

# â”€â”€ Step 4: Join back to the main data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- intersect(names(cell_dt), agg_names)
if (length(existing_neighbor_cols) > 0) {
  cell_dt[, (existing_neighbor_cols) := NULL]
}

cell_dt <- merge(cell_dt, neighbor_stats,
                 by.x = c("id", "year"),
                 by.y = c("cell_id", "year"),
                 all.x = TRUE)

# â”€â”€ Step 5: Convert back to data.frame if downstream code expects it â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)

# â”€â”€ The trained Random Forest model is untouched. â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Proceed directly to prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Same `max`, `min`, `mean` on the same neighbor sets â€” identical estimand. |
| **Trained RF model** | Not retouched; only the feature-construction step is replaced. |
| **Column names** | `neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc. â€” adjust the naming pattern to match whatever your existing `compute_and_add_neighbor_features` produces. |
| **Memory** | The largest intermediate (`panel_edges`) is ~26M rows Ã— ~8 columns â‰ˆ 1.5 GB, well within 16 GB RAM. |
| **Runtime** | Two `data.table` keyed joins + one grouped aggregation. Expected: **5â€“15 minutes** on a standard laptop vs. the original 86+ hours. |

### Optional Further Speedup

If even the `get(v)` inside the `by` group is a concern, you can unroll the five variables explicitly or use `.SDcols`:

```r
# Alternative: .SDcols approach (avoids get() overhead)
neighbor_stats <- panel_edges[
  !is.na(neighbor_id),
  lapply(.SD, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0L) c(NA_real_, NA_real_, NA_real_)
    else c(max(x), min(x), mean(x))
  }),
  by = .(cell_id, year),
  .SDcols = neighbor_source_vars
]
# This returns 3 rows per group; reshape afterward with a row-index trick.
```

The primary solution above is the most straightforward drop-in replacement. The key transformation is: **replace 6.46 million R-level iterations with two data.table binary joins and one vectorized group-by aggregation**.
END_RESPONSE>>>
