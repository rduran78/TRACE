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
run_id: Anthropic_F02_T03_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with millions of keys is extremely slow. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again, once per variable. Combined:

1. **`build_neighbor_lookup`**: ~6.46M iterations, each doing string concatenation and named-vector subsetting against a 6.46M-length named vector. This is the dominant cost â€” estimated at tens of hours alone.
2. **`compute_neighbor_stats`**: Called 5 times (once per variable), each time iterating 6.46M rows with `lapply` and `do.call(rbind, ...)` on a list of 6.46M small vectors. The `do.call(rbind, ...)` on millions of 3-element vectors is also very slow.
3. **Memory**: Storing a 6.46M-element list of integer vectors (the neighbor lookup) plus intermediate string vectors consumes significant RAM on a 16 GB machine.

**Root causes in summary:**
- Row-level `lapply` loops over millions of rows in pure R.
- Repeated string construction (`paste`) and named-vector lookups at scale.
- `do.call(rbind, list_of_millions)` is notoriously slow.
- No vectorization or use of efficient join/merge operations.

---

## Optimization Strategy

### Key Insight
The neighbor lookup is **year-invariant**: the spatial neighbor structure is the same for every year. We should exploit this by separating the spatial topology from the temporal dimension. Instead of building a 6.46M-row lookup, we build a ~344K-cell spatial neighbor edge list once, then use **vectorized merge/join operations** via `data.table` to compute neighbor statistics across all years simultaneously.

### Steps

1. **Convert the `spdep::nb` neighbor list into a flat edge-list `data.table`** with columns `(id, neighbor_id)`. This has ~1.37M rows â€” tiny and fast.
2. **Join the edge list to the panel data by `(neighbor_id, year)`** to pull neighbor values for all cell-years in one vectorized merge. This produces a long table of ~(1.37M Ã— 28) â‰ˆ 38.5M rows.
3. **Group-by aggregate** `(id, year)` to compute `max`, `min`, `mean` in one pass per variable using `data.table`'s optimized grouped operations.
4. **Join the aggregated stats back** to the main data.
5. Repeat for each of the 5 variables (or batch them).

This replaces ~86 hours of row-wise R loops with a handful of vectorized `data.table` joins and group-by operations that should complete in **minutes**.

### Why This Preserves Correctness
- The neighbor relationships are identical (same `rook_neighbors_unique` nb object).
- The statistics computed (`max`, `min`, `mean` of non-NA neighbor values) are identical.
- The main data and the trained Random Forest model are untouched.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Convert spdep::nb object to a flat edge-list
# ============================================================
# id_order is the vector of cell IDs corresponding to positions
# in rook_neighbors_unique (the nb object).

build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: a list of integer vectors (positional indices)
  # id_order maps position -> cell id
  from_ids <- rep(id_order, times = lengths(neighbors))
  to_positions <- unlist(neighbors)
  to_ids <- id_order[to_positions]
  
  data.table(id = from_ids, neighbor_id = to_ids)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (id, neighbor_id)

# ============================================================
# STEP 2: Convert main data to data.table (in place if possible)
# ============================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist and are properly typed
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# ============================================================
# STEP 3: Compute neighbor features for all variables
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a minimal lookup table: (id, year, var1, var2, ...) for neighbor values
# We only need the neighbor source variables plus id and year for the join.
neighbor_val_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..neighbor_val_cols]

# Rename 'id' to 'neighbor_id' so we can join on the neighbor side
setnames(neighbor_vals_dt, "id", "neighbor_id")

# ============================================================
# STEP 4: Join edge list with panel data to get all neighbor
#          observations across all years (vectorized)
# ============================================================
# For each (id, year), we want the values of all neighbors in that same year.
# Join: edge_dt[neighbor_vals_dt] on neighbor_id, then we get
#       (id, neighbor_id, year, ntl, ec, ...)

# Set keys for fast join
setkey(edge_dt, neighbor_id)
setkey(neighbor_vals_dt, neighbor_id)

# This is an inner join: for each edge (id, neighbor_id), attach all years
# of data for that neighbor_id.
# Result: one row per (id, neighbor_id, year) combination.
expanded <- edge_dt[neighbor_vals_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NULL]
# expanded has columns: id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2
# Approximate rows: ~1.37M edges Ã— 28 years = ~38.4M rows

# ============================================================
# STEP 5: Aggregate neighbor stats per (id, year) for each var
# ============================================================
# We compute max, min, mean of non-NA neighbor values grouped by (id, year).

# Build aggregation expressions dynamically
agg_exprs <- lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
})

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

agg_calls <- unlist(agg_exprs, recursive = FALSE)

# Build a single aggregation call
# Using a simpler, robust approach:
agg_stats <- expanded[, {
  out <- list()
  for (v in neighbor_source_vars) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      out[[paste0("neighbor_", v, "_max")]]  <- NA_real_
      out[[paste0("neighbor_", v, "_min")]]  <- NA_real_
      out[[paste0("neighbor_", v, "_mean")]] <- NA_real_
    } else {
      out[[paste0("neighbor_", v, "_max")]]  <- max(vals)
      out[[paste0("neighbor_", v, "_min")]]  <- min(vals)
      out[[paste0("neighbor_", v, "_mean")]] <- mean(vals)
    }
  }
  out
}, by = .(id, year)]

# ============================================================
# STEP 6: Faster alternative using pre-built expressions
#          (replaces STEP 5 if the loop-in-j is too slow)
# ============================================================
# For better data.table optimization, use explicit column expressions:

# Uncomment below and comment out STEP 5 if preferred:
#
# agg_stats <- expanded[, .(
#   neighbor_ntl_max         = ifelse(all(is.na(ntl)), NA_real_, max(ntl, na.rm = TRUE)),
#   neighbor_ntl_min         = ifelse(all(is.na(ntl)), NA_real_, min(ntl, na.rm = TRUE)),
#   neighbor_ntl_mean        = mean(ntl, na.rm = TRUE),
#   neighbor_ec_max          = ifelse(all(is.na(ec)), NA_real_, max(ec, na.rm = TRUE)),
#   neighbor_ec_min          = ifelse(all(is.na(ec)), NA_real_, min(ec, na.rm = TRUE)),
#   neighbor_ec_mean         = mean(ec, na.rm = TRUE),
#   neighbor_pop_density_max = ifelse(all(is.na(pop_density)), NA_real_, max(pop_density, na.rm = TRUE)),
#   neighbor_pop_density_min = ifelse(all(is.na(pop_density)), NA_real_, min(pop_density, na.rm = TRUE)),
#   neighbor_pop_density_mean= mean(pop_density, na.rm = TRUE),
#   neighbor_def_max         = ifelse(all(is.na(def)), NA_real_, max(def, na.rm = TRUE)),
#   neighbor_def_min         = ifelse(all(is.na(def)), NA_real_, min(def, na.rm = TRUE)),
#   neighbor_def_mean        = mean(def, na.rm = TRUE),
#   neighbor_usd_est_n2_max  = ifelse(all(is.na(usd_est_n2)), NA_real_, max(usd_est_n2, na.rm = TRUE)),
#   neighbor_usd_est_n2_min  = ifelse(all(is.na(usd_est_n2)), NA_real_, min(usd_est_n2, na.rm = TRUE)),
#   neighbor_usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
# ), by = .(id, year)]

# ============================================================
# STEP 7: Merge aggregated neighbor features back to cell_data
# ============================================================
setkey(agg_stats, id, year)
setkey(cell_data, id, year)

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

cell_data <- agg_stats[cell_data, on = .(id, year)]
# Left join: all rows of cell_data are preserved.
# Cells with no neighbors will have NA for neighbor stats (correct behavior).

# ============================================================
# STEP 8: Clean up large intermediate objects to free RAM
# ============================================================
rm(expanded, neighbor_vals_dt, agg_stats, edge_dt)
gc()

# cell_data now has all 15 neighbor feature columns appended.
# The trained Random Forest model is untouched.
# The numerical values are identical to the original implementation.
```

---

## Expected Performance

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M row-wise `lapply` with string ops | Eliminated; replaced by ~1.37M-row edge list built in <1 sec |
| `compute_neighbor_stats` | 5 Ã— 6.46M row-wise `lapply` + `do.call(rbind)` | 1 vectorized `data.table` grouped aggregation over ~38.4M rows |
| **Estimated wall time** | **86+ hours** | **5â€“15 minutes** |
| **Peak RAM** | High (6.46M-element list + string vectors) | ~3â€“5 GB for the expanded join table (fits in 16 GB) |

### Why It's Correct
- The edge list is derived from the same `rook_neighbors_unique` nb object and `id_order`.
- The join on `(neighbor_id, year)` replicates the exact same lookup the original code performed row by row.
- `max`, `min`, `mean` with `na.rm = TRUE` and the `all(is.na(...))` guard reproduce the original `c(NA, NA, NA)` fallback for cells with no valid neighbor values.
- The trained Random Forest model is never touched â€” only the input feature table is augmented.
END_RESPONSE>>>
