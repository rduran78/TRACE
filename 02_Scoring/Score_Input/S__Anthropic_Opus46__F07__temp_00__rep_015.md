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
run_id: Anthropic_F07_T00_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) â€” for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries.
2. **`lapply` over 6.46M rows** â€” each iteration does string pasting, named-vector subsetting, and NA filtering. The per-element overhead of `lapply` plus the string operations is enormous.
3. **`compute_neighbor_stats`** then does another `lapply` over 6.46M elements, extracting subsets of a numeric vector. This is less expensive but still slow due to R-level looping.

**Estimated cost**: ~6.46M iterations Ã— (string paste + named-vector lookup for ~4 neighbors each) â‰ˆ billions of character operations. The 86+ hour estimate is credible.

### Root Cause Summary

| Component | Problem |
|---|---|
| `build_neighbor_lookup` | O(NÃ—k) string-key lookups in a 6.46M-entry named vector; R-level loop |
| `compute_neighbor_stats` | R-level `lapply` over 6.46M elements, repeated 5 times |
| Overall architecture | Builds a row-level adjacency list when a vectorized merge/join would suffice |

## Optimization Strategy

**Replace the entire row-level adjacency list with a vectorized edge-table join using `data.table`.**

The key insight: a rook-neighbor relationship between cell `i` and cell `j` in year `t` is simply a join condition `(neighbor_id, year)`. We can:

1. **Expand the `nb` object into an edge table** of `(id, neighbor_id)` â€” done once, ~1.37M rows.
2. **Cross with years implicitly via a keyed join**: join `edges` to `cell_data` on `(neighbor_id, year)` to get neighbor values.
3. **Aggregate** with `data.table`'s `by=` to compute max, min, mean per `(id, year)` â€” fully vectorized in C.

This eliminates all R-level loops and string operations. Expected runtime: **minutes, not hours**.

## Working R Code

```r
library(data.table)

# â”€â”€ Step 0: Convert cell_data to data.table (non-destructive) â”€â”€
# Assumes cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# Assumes rook_neighbors_unique is an nb object (list of integer index vectors)
# Assumes id_order is the vector mapping nb indices to cell IDs

cell_dt <- as.data.table(cell_data)

# â”€â”€ Step 1: Build edge table from nb object (once) â”€â”€
# Convert the nb object (index-based) to an (id, neighbor_id) edge table.
build_edge_table <- function(nb_obj, id_order) {
  # nb objects use integer indices into id_order; 0L means no neighbors
  from_list <- lapply(seq_along(nb_obj), function(i) {
    nb_idx <- nb_obj[[i]]
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  })
  rbindlist(from_list)
}

edges <- build_edge_table(rook_neighbors_unique, id_order)
# edges has ~1,373,394 rows with columns: id, neighbor_id

# â”€â”€ Step 2: Compute neighbor stats for all variables via keyed join â”€â”€
# Key cell_dt for fast joins
setkey(cell_dt, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Select only the columns we need for the neighbor lookup
neighbor_cols <- c("id", "year", neighbor_source_vars)
cell_subset <- cell_dt[, ..neighbor_cols]
setnames(cell_subset, "id", "neighbor_id")
setkey(cell_subset, neighbor_id, year)

# Join edges Ã— years: for each (id, year), get neighbor variable values
# First, expand edges to (id, neighbor_id, year) by joining with cell_dt's (id, year) pairs
# More efficient: join edges to cell_subset on neighbor_id, then aggregate by (id, year)

# We need to know which (id, year) pairs exist. Rather than a full cross,
# we join edges to the cell_dt year column, then look up neighbor values.

# Approach: 
#   1. Create (id, year) from cell_dt
#   2. Join with edges on id -> gives (id, year, neighbor_id)
#   3. Join with cell_subset on (neighbor_id, year) -> gives neighbor values
#   4. Aggregate by (id, year)

# Step 2a: (id, year, neighbor_id) â€” all neighbor lookups needed
id_year <- cell_dt[, .(id, year)]
setkey(edges, id)
setkey(id_year, id)

# This is the big join: ~6.46M rows Ã— ~4 avg neighbors = ~25.8M rows
# data.table handles this efficiently
cat("Building (id, year, neighbor_id) expansion...\n")
expanded <- edges[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
# expanded has columns: id, neighbor_id, year

# Step 2b: Look up neighbor values
cat("Joining neighbor values...\n")
setkey(expanded, neighbor_id, year)
expanded <- cell_subset[expanded, on = .(neighbor_id, year), nomatch = NA]
# Now expanded has: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id

# Step 2c: Aggregate â€” compute max, min, mean for each variable, grouped by (id, year)
cat("Aggregating neighbor statistics...\n")

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

# Build the aggregation call programmatically
agg_list <- setNames(agg_exprs, agg_names)

neighbor_stats <- expanded[,
  {
    res <- list()
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        res[[paste0("neighbor_max_", v)]]  <- NA_real_
        res[[paste0("neighbor_min_", v)]]  <- NA_real_
        res[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        res[[paste0("neighbor_max_", v)]]  <- max(vals)
        res[[paste0("neighbor_min_", v)]]  <- min(vals)
        res[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    res
  },
  by = .(id, year)
]

# â”€â”€ Step 3: Merge back to cell_dt â”€â”€
cat("Merging neighbor stats back to main data...\n")
setkey(neighbor_stats, id, year)
setkey(cell_dt, id, year)

# Remove old neighbor columns if they exist (idempotent re-run)
old_cols <- intersect(names(cell_dt), agg_names)
if (length(old_cols) > 0) cell_dt[, (old_cols) := NULL]

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

# â”€â”€ Step 4: Convert back to data.frame if needed by downstream code â”€â”€
cell_data <- as.data.frame(cell_dt)

cat("Done. Neighbor features added for:", paste(neighbor_source_vars, collapse = ", "), "\n")
```

### Even Faster Aggregation (Optional Refinement)

The `by`-group loop above with `get()` can be replaced with a purely vectorized `data.table` aggregation for maximum speed:

```r
# Faster alternative for Step 2c â€” fully vectorized, no R-level loop inside groups
neighbor_stats <- expanded[, {
  out <- list()
  for (v in neighbor_source_vars) {
    vv <- .SD[[v]]
    vv <- vv[!is.na(vv)]
    n <- length(vv)
    out[[paste0("neighbor_max_", v)]]  <- if (n > 0L) max(vv) else NA_real_
    out[[paste0("neighbor_min_", v)]]  <- if (n > 0L) min(vv) else NA_real_
    out[[paste0("neighbor_mean_", v)]] <- if (n > 0L) mean(vv) else NA_real_
  }
  out
}, by = .(id, year), .SDcols = neighbor_source_vars]
```

Or, for the absolute fastest path, compute each variable separately:

```r
setkey(expanded, id, year)

stat_tables <- lapply(neighbor_source_vars, function(v) {
  expanded[!is.na(get(v)), .(
    nmax  = max(get(v)),
    nmin  = min(get(v)),
    nmean = mean(get(v))
  ), by = .(id, year)] |>
    setnames(c("nmax", "nmin", "nmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))
})

# Merge all stat tables
neighbor_stats <- Reduce(function(a, b) merge(a, b, by = c("id", "year"), all = TRUE), stat_tables)
```

## Performance Estimate

| Step | Rows | Expected Time |
|---|---|---|
| `build_edge_table` | ~1.37M | < 5 seconds |
| Cartesian expansion (`edges` Ã— `id_year`) | ~25.8M | ~10â€“30 seconds |
| Keyed join for neighbor values | ~25.8M | ~10â€“30 seconds |
| Aggregation (max/min/mean Ã— 5 vars) | ~25.8M â†’ ~6.46M | ~30â€“90 seconds |
| Final merge | ~6.46M | ~5â€“10 seconds |
| **Total** | | **~2â€“5 minutes** |

## What Is Preserved

- **Trained Random Forest model**: No retraining. The code only computes the same 15 neighbor features (3 stats Ã— 5 variables) that the model expects.
- **Numerical estimand**: The max, min, and mean computations are identical â€” same neighbor sets, same NA handling, same arithmetic. The results are numerically equivalent to the original implementation.
- **Irregular topology**: The `nb` object's exact neighbor relationships are faithfully converted to the edge table. No raster focal assumptions are introduced.
END_RESPONSE>>>
