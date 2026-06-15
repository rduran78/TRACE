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
run_id: Anthropic_F09_T10_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs string pasting (`paste(id, year, sep="_")`), named-vector lookups, and `NA` filtering. With ~6.46M rows, this creates millions of small R-level operations. Named vector lookups via character keys (`idx_lookup[neighbor_keys]`) are hash-table lookups repeated millions of times inside an interpreted loopâ€”extremely slow.

### 2. The neighbor topology is **year-invariant**, but the lookup is rebuilt per cell-year
The rook-neighbor relationships between cells never change across years. Yet `build_neighbor_lookup` produces a list of length 6.46M (one entry per cell-year) rather than a list of length 344,208 (one entry per cell). This 28Ã— blowup is entirely unnecessary.

### 3. `compute_neighbor_stats` iterates over 6.46M list elements in R
Even after the lookup is built, computing max/min/mean by iterating `lapply` over 6.46M entriesâ€”each performing subsetting, `NA` removal, and three summary statisticsâ€”is extremely slow in interpreted R.

**Bottom line:** The entire approach conflates *spatial topology* (fixed) with *temporal attribute data* (varying). The fix is to separate the two, build the adjacency table once, then use fast vectorized joins and grouped aggregations.

---

## Optimization Strategy

1. **Build a static, two-column adjacency edge table once** from the `nb` object: `data.table(cell_id, neighbor_id)`. This table has ~1.37M rows and never changes.

2. **Join yearly attributes onto the edge table** by `(neighbor_id, year)` using `data.table` keyed joins. This is a single vectorized merge per variableâ€”no R-level loops.

3. **Compute grouped summary statistics** `(max, min, mean)` by `(cell_id, year)` using `data.table`'s `by=` grouped aggregationâ€”fully vectorized in C.

4. **Merge the resulting features back** onto the main dataset with a single keyed join per variable.

5. The trained Random Forest model is **never touched**â€”only the feature-engineering pipeline changes. The numerical outputs (max, min, mean of neighbor values) are **identical** to the original.

**Expected speedup:** From ~86 hours to **minutes** (typically 2â€“10 minutes on a 16 GB laptop), because all heavy operations become vectorized C-level `data.table` operations over ~38M edge-year rows (1.37M edges Ã— 28 years).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# plus all other predictor columns. We keep everything intact.
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order so the final dataset aligns with any
# downstream indexing (e.g., the trained Random Forest expects this order).
cell_data[, .row_order := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static adjacency edge table ONCE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique : spdep nb object (list of integer index vectors)
# id_order              : vector of cell IDs in the same order as the nb object

build_adjacency_table <- function(id_order, neighbors_nb) {
  # Each element i of neighbors_nb is an integer vector of neighbor indices

  # into id_order. Index 0 (spdep convention for no neighbors) is handled.
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  return(edges)
}

cat("Building static adjacency table...\n")
adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)
# adj_table has ~1,373,394 rows: (cell_id, neighbor_id)
cat(sprintf("  Adjacency table: %s directed edges\n", format(nrow(adj_table), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each neighbor source variable, compute neighbor stats
#         via vectorized join + grouped aggregation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a slim lookup keyed by (id, year) for joining neighbor attributes.
# We only need the neighbor source variable columns plus id and year.
attr_cols <- unique(c("id", "year", neighbor_source_vars))
attr_lookup <- cell_data[, ..attr_cols]
setkey(attr_lookup, id, year)

# We will also key adj_table on neighbor_id for the join.
setkey(adj_table, neighbor_id)

cat("Computing neighbor features...\n")

for (var_name in neighbor_source_vars) {

  cat(sprintf("  Processing: %s\n", var_name))

  # Slim attribute table for this variable: (id, year, value)
  var_dt <- attr_lookup[, .(id, year, value = get(var_name))]
  setnames(var_dt, "id", "neighbor_id")
  setkey(var_dt, neighbor_id)

  # Cross join: expand adj_table Ã— years by joining on neighbor_id

  # Result: for every (cell_id, neighbor_id) edge and every year,
  # we get the neighbor's attribute value.
  # Efficient approach: merge adj_table with var_dt on neighbor_id,
  # which naturally replicates across all years the neighbor appears in.
  edge_vals <- var_dt[adj_table, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # edge_vals columns: neighbor_id, year, value, cell_id

  # Remove rows where the neighbor's value is NA (matches original logic)
  edge_vals <- edge_vals[!is.na(value)]

  # Grouped aggregation: compute max, min, mean per (cell_id, year)
  stats <- edge_vals[, .(
    nb_max  = max(value),
    nb_min  = min(value),
    nb_mean = mean(value)
  ), by = .(cell_id, year)]

  # Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(stats, "cell_id", "id")

  # Merge back onto cell_data by (id, year)
  # First, remove these columns if they already exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)

  # Clean up

  rm(var_dt, edge_vals, stats)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Restore original row order and clean up
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

cat("Neighbor feature engineering complete.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the existing trained Random Forest
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., rf_model) is unchanged.
# Predictions proceed exactly as before:
#
#   predictions <- predict(rf_model, newdata = cell_data)
#
# The numerical estimand is preserved because the neighbor max, min,
# and mean values are computed identicallyâ€”only the computational
# method (vectorized joins vs. row-level R loops) has changed.
```

---

## Why This Is Correct and Equivalent

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor topology | Embedded in 6.46M-element list | Explicit 1.37M-row edge table |
| Attribute lookup | Character-key hash per row | `data.table` keyed binary-search join |
| Aggregation | `lapply` + per-element `max/min/mean` | `data.table` grouped `by=` (C-level) |
| NA handling | `neighbor_vals[!is.na(...)]` | `edge_vals[!is.na(value)]` before grouping |
| Result | 3 columns per variable (max, min, mean) | Identical 3 columns per variable |
| RF model | Untouched | Untouched |
| Numerical output | Identical | Identical |

**Expected runtime:** 2â€“10 minutes on a 16 GB RAM laptop instead of 86+ hoursâ€”a roughly **500â€“2,500Ã— speedup**.
END_RESPONSE>>>
