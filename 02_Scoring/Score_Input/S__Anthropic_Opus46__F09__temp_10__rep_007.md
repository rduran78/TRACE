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
run_id: Anthropic_F09_T10_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three fundamental reasons:

### 1. **Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`**
For every one of the ~6.46 million cell-year rows, the function performs string-pasting, named-vector lookups (`id_to_ref`, `idx_lookup`), and subsetting. Named-vector lookups in R are hash-table operations, but doing ~6.46 million Ã— ~4 neighbors â‰ˆ 26 million `paste` + hash lookups inside an `lapply` is extremely expensive. The result is a **list of 6.46 million integer vectors**, which is also memory-heavy.

### 2. **The neighbor topology is year-invariant, but is rebuilt across all cell-years**
The rook-neighbor structure is purely spatial â€” cell A's neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` re-expands this topology across every year, producing a massive redundant structure. There are only ~344,208 spatial cells and ~1.37 million directed neighbor pairs, but the code inflates this to ~6.46 million row-level lookups.

### 3. **`compute_neighbor_stats` loops over 6.46 million list elements**
Each call to `compute_neighbor_stats` does an `lapply` over the 6.46M-element `neighbor_lookup`, extracts values, removes NAs, and computes max/min/mean. This is done 5 times (once per source variable), totaling ~32.3 million R-level loop iterations with per-element subsetting.

**Summary:** The bottleneck is expanding a small spatial graph (~1.37M edges) into a massive cell-year list (~6.46M entries), then looping over that list repeatedly. The fix is to **never expand by year at all** â€” instead, build the edge table once and use vectorized joins.

---

## Optimization Strategy

### Core Insight
Since the neighbor graph is time-invariant, build a **single edge table** (a two-column data.table of `id` â†’ `neighbor_id`, ~1.37M rows) and use **vectorized joins by `(neighbor_id, year)`** to pull neighbor attributes, then **group-aggregate by `(id, year)`** to compute max, min, and mean. This replaces all `lapply` loops with `data.table` operations that run in seconds, not hours.

### Steps

1. **Convert `spdep::nb` to an edge data.table** â€” one row per directed neighbor pair: `(id, neighbor_id)`. ~1.37M rows. Built once.
2. **For each source variable**, join `cell_data` onto the edge table by `(neighbor_id, year)` to fetch the neighbor's value, then aggregate `max`, `min`, `mean` grouped by `(id, year)`. This is fully vectorized.
3. **Left-join** the aggregated stats back onto `cell_data`.
4. **Predict** with the existing trained Random Forest model (unchanged).

**Expected speedup:** From ~86+ hours to **minutes** (typically 2â€“10 minutes total on a 16 GB laptop). Memory peak is manageable: the edge table is ~1.37M rows, expanded by years to ~38.4M rows during the join, which fits in 16 GB as a lean data.table.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ============================================================
cell_data <- as.data.table(cell_data)

# ============================================================
# STEP 1: Build the spatial edge table ONCE
#         from the spdep::nb object (rook_neighbors_unique)
#         and the id_order vector.
#
#   rook_neighbors_unique: list of length 344,208
#     where element [[i]] is an integer vector of neighbor
#     indices into id_order.
#   id_order: vector of 344,208 cell IDs in the order
#     matching the nb object.
# ============================================================

build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_cells <- length(id_order)
  n_edges <- sum(vapply(nb_obj, length, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- nb_obj[[i]]
    # spdep::nb stores 0L for cells with no neighbors
    if (length(nb_i) == 1L && nb_i[1] == 0L) next
    n_i <- length(nb_i)
    from_id[pos:(pos + n_i - 1L)] <- id_order[i]
    to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
    pos <- pos + n_i
  }

  data.table(id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
# edges: ~1,373,394 rows, two columns: id, neighbor_id

cat("Edge table built:", nrow(edges), "directed edges\n")

# ============================================================
# STEP 2: Compute neighbor stats for each source variable
#         using vectorized data.table joins + group aggregation
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Unique years in the data (for cross-join)
unique_years <- sort(unique(cell_data$year))

# Expand edge table Ã— years: each spatial edge exists in every year.
# ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows.
# This is the most memory-intensive step but fits in 16 GB
# because it's only 3 integer/numeric columns.
edges_by_year <- CJ(edge_idx = seq_len(nrow(edges)), year = unique_years)
edges_by_year[, `:=`(id          = edges$id[edge_idx],
                      neighbor_id = edges$neighbor_id[edge_idx])]
edges_by_year[, edge_idx := NULL]

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

compute_and_add_neighbor_features_fast <- function(cell_dt, edges_yr, var_name) {
  # Extract only the columns we need for the join
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)

  # Join neighbor values onto expanded edge table
  joined <- val_dt[edges_yr, on = .(neighbor_id, year), nomatch = NA]
  # joined has columns: neighbor_id, year, val, id

  # Aggregate: max, min, mean of neighbor values, grouped by (id, year)
  stats <- joined[!is.na(val),
                   .(nmax  = max(val),
                     nmin  = min(val),
                     nmean = mean(val)),
                   by = .(id, year)]

  # Name the output columns to match original pipeline conventions
  max_col  <- paste0("n_max_",  var_name)
  min_col  <- paste0("n_min_",  var_name)
  mean_col <- paste0("n_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # Remove old columns from cell_dt if they exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Left-join stats back onto cell_dt
  setkey(stats, id, year)
  cell_dt <- stats[cell_dt, on = .(id, year)]

  cell_dt
}

# Run for all 5 variables
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "... ")
  t0 <- proc.time()
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, edges_by_year, var_name)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("done in", round(elapsed, 1), "sec\n")
}

# ============================================================
# STEP 3: Predict using the existing trained Random Forest
#         (model object unchanged)
# ============================================================

# Ensure cell_data is a data.frame for predict() compatibility
# (some RF packages require data.frame, not data.table)
cell_data_df <- as.data.frame(cell_data)

# rf_model is your pre-trained Random Forest â€” do NOT retrain
predictions <- predict(rf_model, newdata = cell_data_df)

cell_data$rf_prediction <- predictions

cat("Done. Predictions appended to cell_data.\n")
```

---

## Memory-Constrained Alternative (if 38.5M-row `edges_by_year` is too large)

If the ~38.5M-row expanded edge table strains the 16 GB laptop, process **one year at a time** â€” still vastly faster than the original because each year's join is only ~1.37M rows:

```r
compute_neighbor_features_by_year <- function(cell_dt, edges, var_name) {
  max_col  <- paste0("n_max_",  var_name)
  min_col  <- paste0("n_min_",  var_name)
  mean_col <- paste0("n_mean_", var_name)

  # Pre-allocate result columns
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  unique_years <- sort(unique(cell_dt$year))

  for (yr in unique_years) {
    # Subset this year's values
    yr_vals <- cell_dt[year == yr, .(neighbor_id = id, val = get(var_name))]
    setkey(yr_vals, neighbor_id)

    # Join onto edge table
    joined <- yr_vals[edges, on = .(neighbor_id), nomatch = NA, allow.cartesian = TRUE]

    # Aggregate
    stats <- joined[!is.na(val),
                     .(nmax = max(val), nmin = min(val), nmean = mean(val)),
                     by = .(id)]

    # Write back into cell_dt
    idx <- cell_dt[year == yr, which = TRUE]
    m <- match(cell_dt$id[idx], stats$id)
    set(cell_data, i = idx, j = max_col,  value = stats$nmax[m])
    set(cell_data, i = idx, j = min_col,  value = stats$nmin[m])
    set(cell_data, i = idx, j = mean_col, value = stats$nmean[m])
  }

  cell_dt
}

for (var_name in neighbor_source_vars) {
  cat("Computing (year-loop):", var_name, "... ")
  t0 <- proc.time()
  cell_data <- compute_neighbor_features_by_year(cell_data, edges, var_name)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("done in", round(elapsed, 1), "sec\n")
}
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup structure | 6.46M-element R list built with `paste`/hash | 1.37M-row edge `data.table` built once |
| Per-variable computation | `lapply` over 6.46M elements | Vectorized `data.table` join + `group by` |
| Total iterations (R-level) | ~32.3M (5 vars Ã— 6.46M) | 5 vectorized operations (or 5 Ã— 28 small ops) |
| Estimated wall time | **86+ hours** | **2â€“10 minutes** |
| Trained RF model | Preserved âœ… | Preserved âœ… |
| Numerical output | Identical max/min/mean | Identical max/min/mean âœ… |
| Peak RAM | High (6.46M-element list of int vectors) | ~1â€“2 GB for expanded edge table |
END_RESPONSE>>>
