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
run_id: Anthropic_F09_T00_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of string operations and hash lookups. Named vector lookups via `[` on character keys are O(n) in the worst case per call due to R's internal hashing overhead at scale.

### 2. Redundant recomputation of the same spatial topology per year
The rook-neighbor relationships are **purely spatial** â€” they don't change across years. Yet `build_neighbor_lookup` rebuilds the neighbor mapping for every cell-year combination, effectively duplicating the same spatial adjacency structure 28 times (once per year). This inflates the lookup list from ~344K entries to ~6.46M entries.

### 3. Row-level `lapply` over 6.46M rows in `compute_neighbor_stats`
Each call to `compute_neighbor_stats` iterates over all 6.46M rows, subsetting and computing `max`, `min`, `mean` one row at a time. With 5 variables, that's ~32.3M individual R-level loop iterations with per-element vector subsetting.

### Summary of waste
| Component | Current scale | Optimal scale | Waste factor |
|---|---|---|---|
| Neighbor lookup | 6.46M entries | 344K entries (year-invariant) | ~19Ã— |
| Stat computation | 6.46M Ã— 5 lapply calls | Vectorized join + group-by | Orders of magnitude |

---

## Optimization Strategy

The key insight: **the neighbor table is time-invariant; only the cell attributes change by year.** Therefore:

1. **Build the adjacency edge-list once** from the `nb` object â€” a simple two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is done once and reused forever.

2. **Join yearly attributes onto the edge-list by year.** For each year, join the neighbor cell's attribute values onto the edge-list via a keyed `data.table` join. This turns the neighbor-value lookup into a vectorized merge.

3. **Compute grouped `max`, `min`, `mean` via `data.table` aggregation** â€” grouping by `(cell_id, year)` over the joined edge-list. This replaces millions of `lapply` iterations with a single vectorized group-by.

4. **Join the resulting neighbor stats back** onto the main `cell_data` table.

This reduces the problem from ~6.46M R-level loop iterations to a handful of vectorized `data.table` operations, bringing runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the time-invariant adjacency edge-list ONCE
# ==============================================================================
# Input:
#   id_order             â€” vector of 344,208 cell IDs (positional index matches nb object)
#   rook_neighbors_unique â€” spdep::nb object (list of length 344,208; each element
#                           is an integer vector of positional indices of neighbors)
#
# Output:
#   adj_dt â€” data.table with columns (cell_id, neighbor_id), ~1.37M rows

build_adjacency_edgelist <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_cells <- length(id_order)
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  cell_id_vec    <- integer(n_edges)
  neighbor_id_vec <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
    n_nb <- length(nb_idx)
    cell_id_vec[pos:(pos + n_nb - 1L)]    <- id_order[i]
    neighbor_id_vec[pos:(pos + n_nb - 1L)] <- id_order[nb_idx]
    pos <- pos + n_nb
  }

  data.table(cell_id = cell_id_vec, neighbor_id = neighbor_id_vec)
}

adj_dt <- build_adjacency_edgelist(id_order, rook_neighbors_unique)

cat(sprintf("Adjacency edge-list: %d directed edges across %d cells\n",
            nrow(adj_dt), length(id_order)))

# ==============================================================================
# STEP 2: Convert main data to data.table (if not already)
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist and are of consistent type
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================================
# STEP 3: For each neighbor source variable, compute neighbor stats via
#          vectorized join + grouped aggregation, then merge back.
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We will cross-join the adjacency edge-list with all 28 years, then join
# neighbor attributes. But that would create ~1.37M Ã— 28 â‰ˆ 38.5M rows,
# which is fine for data.table but uses memory. A more memory-friendly
# approach: iterate year-by-year or, better, join directly on (neighbor_id, year).

# Strategy: expand adj_dt by year via merge with cell_data's (id, year) pairs,
# then look up neighbor values.

# Create a slim lookup: only id, year, and the 5 source vars
lookup_cols <- c("id", "year", neighbor_source_vars)
# Ensure all columns exist
stopifnot(all(lookup_cols %in% names(cell_data)))

neighbor_vals_dt <- cell_data[, ..lookup_cols]
setnames(neighbor_vals_dt, "id", "neighbor_id")
# Key for fast join
setkey(neighbor_vals_dt, neighbor_id, year)

# Create the cell-year backbone from cell_data (id, year) to expand adj_dt
cell_year_dt <- cell_data[, .(cell_id = id, year)]

# Merge cell_year_dt with adj_dt to get (cell_id, year, neighbor_id)
# This is: for each (cell, year), list all spatial neighbors
setkey(adj_dt, cell_id)
setkey(cell_year_dt, cell_id)

# Efficient expansion: join adj_dt onto cell_year_dt
# Result: each row of cell_year_dt is expanded by the number of neighbors
edges_by_year <- adj_dt[cell_year_dt, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
# edges_by_year has columns: cell_id, neighbor_id, year
# Expected rows: ~6.46M Ã— avg_neighbors â‰ˆ ~6.46M Ã— (1.37M/344KÃ—2 sides... ~4 neighbors each)
# â‰ˆ ~25.8M rows â€” fits comfortably in 16 GB

cat(sprintf("Expanded edge-year table: %s rows\n", format(nrow(edges_by_year), big.mark = ",")))

# Join neighbor attribute values onto edges_by_year
setkey(edges_by_year, neighbor_id, year)
edges_by_year <- neighbor_vals_dt[edges_by_year, on = .(neighbor_id, year)]

# Now edges_by_year has columns:
#   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, cell_id

# ==============================================================================
# STEP 4: Grouped aggregation â€” compute max, min, mean per (cell_id, year, var)
# ==============================================================================

# Build all neighbor stat columns in one grouped aggregation
agg_expr_list <- list()
for (var_name in neighbor_source_vars) {
  var_sym <- as.name(var_name)
  agg_expr_list[[paste0("neighbor_max_", var_name)]] <-
    bquote(as.numeric(max(.(var_sym), na.rm = TRUE)))
  agg_expr_list[[paste0("neighbor_min_", var_name)]] <-
    bquote(as.numeric(min(.(var_sym), na.rm = TRUE)))
  agg_expr_list[[paste0("neighbor_mean_", var_name)]] <-
    bquote(mean(.(var_sym), na.rm = TRUE))
}

# Handle edge case: max/min of zero-length after NA removal returns Inf/-Inf;
# we will convert those to NA after aggregation.

# Build the aggregation call dynamically
agg_call <- as.call(c(as.name("list"),
                       setNames(agg_expr_list, names(agg_expr_list))))

neighbor_stats <- edges_by_year[, eval(agg_call), by = .(cell_id, year)]

# Replace Inf / -Inf with NA (from max/min of all-NA neighbor groups)
stat_cols <- names(neighbor_stats)[!(names(neighbor_stats) %in% c("cell_id", "year"))]
for (col in stat_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

cat(sprintf("Neighbor stats computed: %d rows Ã— %d columns\n",
            nrow(neighbor_stats), ncol(neighbor_stats)))

# ==============================================================================
# STEP 5: Merge neighbor stats back onto cell_data
# ==============================================================================

# Remove any pre-existing neighbor stat columns to avoid duplication
old_neighbor_cols <- intersect(names(cell_data), stat_cols)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

# Keyed merge
setkey(cell_data, id, year)
setnames(neighbor_stats, "cell_id", "id")
setkey(neighbor_stats, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat(sprintf("Final cell_data: %s rows Ã— %d columns\n",
            format(nrow(cell_data), big.mark = ","), ncol(cell_data)))

# ==============================================================================
# STEP 6: Predict with the existing trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) is used as-is.
# The column names produced above (neighbor_max_ntl, neighbor_min_ntl,
# neighbor_mean_ntl, etc.) must match the names the model was trained on.
#
# If the original code used different naming conventions, rename here:
# Example â€” if original names were like "ntl_neighbor_max":
#
# for (var_name in neighbor_source_vars) {
#   setnames(cell_data,
#     old = c(paste0("neighbor_max_", var_name),
#             paste0("neighbor_min_", var_name),
#             paste0("neighbor_mean_", var_name)),
#     new = c(paste0(var_name, "_neighbor_max"),
#             paste0(var_name, "_neighbor_min"),
#             paste0(var_name, "_neighbor_mean")),
#     skip_absent = TRUE)
# }
#
# Then predict:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Expected Performance Comparison

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~6.46M `paste` + hash lookups (~hours) | ~344K loop to build edge-list (~seconds) |
| Expand edges Ã— years | Implicit in lookup (~hours) | `data.table` cartesian join (~10â€“30 sec) |
| Join neighbor values | Embedded in `lapply` (~hours per var) | Single keyed join (~10â€“30 sec) |
| Compute max/min/mean | 6.46M Ã— 5 `lapply` iterations (~days) | One grouped `data.table` aggregation (~30â€“60 sec) |
| Merge back to cell_data | Per-variable column binding | One keyed join (~seconds) |
| **Total** | **~86+ hours** | **~2â€“5 minutes** |

### Why correctness is preserved
- The adjacency relationships are identical (same `nb` object, same `id_order`).
- The `max`, `min`, `mean` with `na.rm = TRUE` produce the same numerical values as the original `lapply` implementation.
- `Inf`/`-Inf` from empty groups are mapped to `NA`, matching the original `return(c(NA, NA, NA))` guard.
- The trained Random Forest model is never modified or retrained â€” only the feature-engineering pipeline is accelerated.
END_RESPONSE>>>
