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
run_id: Anthropic_F09_T07_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all cell-year rows (~6.46M), but it rebuilds neighbor relationships redundantly across years.

The neighbor topology is **purely spatial** â€” cell A's neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` creates a lookup entry **per cell-year row**, iterating over ~6.46M rows, performing character key construction and hash lookups for each. This means the same spatial neighbor set is resolved **28 times** (once per year) for each of the 344,208 cells. That's ~9.6 million redundant key-paste-and-match operations.

### 2. `compute_neighbor_stats` uses an R-level `lapply` over ~6.46M entries.

Each iteration indexes into a numeric vector and computes `max`, `min`, `mean` â€” all in interpreted R. With 5 variables Ã— 6.46M rows, that's ~32.3 million R-level function calls with per-element vector subsetting. This is the dominant bottleneck.

### 3. Memory bloat from the lookup structure.

`neighbor_lookup` is a list of 6.46M integer vectors. The list overhead alone (~6.46M SEXP pointers + individual vector allocations) can consume several GB of RAM on a 16 GB laptop, causing GC pressure and swapping.

**Root cause summary:** The spatial topology is conflated with the temporal panel. The code treats each cell-year as a unique entity needing its own neighbor resolution, when in fact the neighbor graph is time-invariant.

---

## Optimization Strategy

**Core idea:** Separate the *spatial topology* (built once) from the *temporal attributes* (joined per year), then compute neighbor statistics using vectorized `data.table` operations instead of row-wise R loops.

### Step-by-step plan:

1. **Build a spatial edge table once.** Convert the `spdep::nb` object into a two-column `data.table` of `(cell_id, neighbor_id)` â€” roughly 1.37M rows. This is done once and is year-invariant.

2. **For each year, join cell attributes onto the edge table.** This gives each edge the neighbor's attribute value. Then group by `(cell_id, year)` and compute `max`, `min`, `mean` in one vectorized pass.

3. **Join the resulting statistics back to the main data.** This replaces the per-row R-level `lapply`.

### Complexity comparison:

| | Current | Optimized |
|---|---|---|
| Neighbor resolution | ~6.46M R-level iterations | ~1.37M-row edge table (built once) |
| Stats computation | ~32.3M R-level `lapply` calls | 5 vectorized `data.table` group-by operations over ~38.4M edge-year rows |
| Expected time | ~86+ hours | **~2â€“10 minutes** |
| RAM for lookup | Several GB (list of 6.46M vectors) | ~200â€“400 MB (edge table + joins) |

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table with columns: id, year, and all
#         predictor variables. The trained RF model object is untouched.
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================================
# STEP 1: Build the time-invariant spatial edge table ONCE from the nb object.
#
#   rook_neighbors_unique : spdep nb object (list of integer index vectors)
#   id_order              : vector of cell IDs in the same order as the nb object
#
#   Result: edges_dt â€” a data.table with columns (cell_id, neighbor_id)
#           representing every directed rook-neighbor pair.
# ==============================================================================

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_cells <- length(id_order)
  n_edges <- sum(vapply(neighbors, length, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
      to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edges_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Edge table built: %s directed neighbor pairs for %s cells.\n",
  format(nrow(edges_dt), big.mark = ","),
  format(length(id_order), big.mark = ",")
))

# ==============================================================================
# STEP 2: For each neighbor source variable, compute neighbor max, min, mean
#         by joining yearly attributes onto the edge table, then grouping.
#
#   This replaces build_neighbor_lookup + compute_neighbor_stats entirely.
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Extract only the columns we need for the join: id, year, and the source vars.
# This keeps memory lean during the join operations.
join_cols <- unique(c("id", "year", neighbor_source_vars))
attr_dt   <- cell_data[, ..join_cols]

# We will join edges_dt (cell_id -> neighbor_id) with attr_dt on neighbor_id + year.
# First, get the unique years to iterate over (avoids a massive cross-join).
all_years <- sort(unique(attr_dt$year))

# Pre-set keys for fast joins
setkey(edges_dt, neighbor_id)

# Function to compute neighbor stats for one variable across all years
compute_neighbor_features_fast <- function(attr_dt, edges_dt, var_name, all_years) {

  cat(sprintf("  Computing neighbor stats for: %s ...\n", var_name))

  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Subset to just id, year, and the variable of interest
  sub_dt <- attr_dt[, .(id, year, val = get(var_name))]
  setkey(sub_dt, id)

  # For each year, join neighbor values and aggregate
  results_list <- vector("list", length(all_years))

  for (yi in seq_along(all_years)) {
    yr <- all_years[yi]

    # Get attribute values for this year
    yr_vals <- sub_dt[year == yr, .(neighbor_id = id, neighbor_val = val)]
    setkey(yr_vals, neighbor_id)

    # Join: for each edge, attach the neighbor's value in this year
    edge_vals <- edges_dt[yr_vals, on = "neighbor_id", nomatch = 0L, allow.cartesian = FALSE]
    # edge_vals now has columns: cell_id, neighbor_id, neighbor_val

    # Remove NA neighbor values before aggregation
    edge_vals <- edge_vals[!is.na(neighbor_val)]

    # Aggregate by cell_id
    if (nrow(edge_vals) > 0L) {
      agg <- edge_vals[, .(
        nmax  = max(neighbor_val),
        nmin  = min(neighbor_val),
        nmean = mean(neighbor_val)
      ), by = cell_id]
      agg[, year := yr]
      results_list[[yi]] <- agg
    }
  }

  result_dt <- rbindlist(results_list, use.names = TRUE)
  setnames(result_dt, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  setnames(result_dt, "cell_id", "id")

  return(result_dt)
}

# Compute and merge all neighbor features
for (var_name in neighbor_source_vars) {

  # Remove old columns if they exist (idempotent re-runs)
  old_cols <- paste0(c("neighbor_max_", "neighbor_min_", "neighbor_mean_"), var_name)
  old_cols_present <- old_cols[old_cols %in% names(cell_data)]
  if (length(old_cols_present) > 0L) {
    cell_data[, (old_cols_present) := NULL]
  }

  feat_dt <- compute_neighbor_features_fast(attr_dt, edges_dt, var_name, all_years)

  # Merge back onto cell_data by id + year
  cell_data <- merge(cell_data, feat_dt, by = c("id", "year"), all.x = TRUE)
}

cat("All neighbor features computed and merged.\n")

# ==============================================================================
# STEP 3: Predict with the existing trained Random Forest model.
#
#   The model object (e.g., rf_model) is UNCHANGED. We simply feed the updated
#   cell_data (now containing the neighbor_max_*, neighbor_min_*, neighbor_mean_*
#   columns) into predict().
#
#   The numerical estimand is preserved because the neighbor statistics are
#   computed identically: max, min, and mean of non-NA neighbor values per
#   cell-year, exactly as the original code produced.
# ==============================================================================

# Example prediction call (adjust 'rf_model' and predictor column selection
# to match your existing pipeline):
#
# predictor_cols <- setdiff(names(cell_data), c("id", "year", "response_var"))
# cell_data[, prediction := predict(rf_model, newdata = cell_data[, ..predictor_cols])]

cat("Pipeline complete. Trained RF model is preserved; apply predict() as before.\n")
```

---

## Why This Preserves the Original Numerical Estimand

The original code computes, for each cell-year row, the `max`, `min`, and `mean` of each neighbor source variable across that cell's rook neighbors **in the same year**, excluding `NA` values. The optimized code does exactly the same thing:

1. The edge table encodes the identical spatial neighbor pairs derived from `rook_neighbors_unique` and `id_order`.
2. The join on `neighbor_id` within a single year retrieves exactly the same neighbor attribute values.
3. `NA` removal (`!is.na(neighbor_val)`) and the `max`/`min`/`mean` aggregation match the original logic.
4. Cells with no valid neighbors in a given year receive `NA` via the `all.x = TRUE` merge, matching the original `c(NA, NA, NA)` return.

The trained Random Forest model is never modified â€” only the feature-engineering preprocessing is accelerated.

### Expected performance on a 16 GB laptop:

- **Edge table construction:** < 5 seconds (single loop over 344K cells).
- **Neighbor stats for all 5 variables Ã— 28 years:** ~2â€“8 minutes total (vectorized `data.table` joins and grouped aggregations).
- **Peak RAM:** ~3â€“5 GB (well within 16 GB).
- **Total speedup:** roughly **500â€“2,500Ã—** versus the original 86+ hour estimate.
END_RESPONSE>>>
