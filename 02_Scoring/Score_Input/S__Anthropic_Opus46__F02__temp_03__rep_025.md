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
run_id: Anthropic_F02_T03_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- **String key construction**: `paste(id, year, sep="_")` is called millions of times inside the loop body, and named-vector lookup (`idx_lookup[neighbor_keys]`) is an O(k) hash probe per neighbor key, repeated for every row.
- **Redundant work across years**: Every cell has the same neighbors in every year. The neighbor *structure* is time-invariant, yet the lookup is rebuilt per cell-year row by pasting year into every neighbor key. With 28 years Ã— 344,208 cells, the same neighbor set is resolved 28 times per cell.
- **Memory**: The named character vector `idx_lookup` with 6.46M entries and the resulting list of 6.46M integer vectors is large (~2â€“4 GB depending on neighbor counts).

### 2. `compute_neighbor_stats` â€” O(n) `lapply` with per-row subsetting and aggregation
- For each of the 6.46M rows, it subsets a numeric vector by an index vector, removes NAs, and computes max/min/mean. This is repeated for each of the 5 variables (32.3M iterations total).
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow and memory-hungry.

### Combined effect
The nested per-row R-level loops with string operations and list manipulations explain the 86+ hour estimate. R's interpreted loop overhead dominates.

---

## Optimization Strategy

| Principle | Technique |
|---|---|
| **Eliminate redundant work** | Build the neighbor lookup as a *cell-to-cell* adjacency (time-invariant), then join by year using vectorized operations â€” never loop over cell-years. |
| **Vectorize aggregation** | Convert the neighbor list to a long-form `data.table` edge list, join the variable values, and compute grouped `max/min/mean` in one vectorized pass per variable. |
| **Use `data.table`** | `data.table` provides in-place `:=` assignment, fast keyed joins, and optimized `by`-group aggregation in C â€” eliminates R-level loops entirely. |
| **Minimize memory** | The long-form edge list has ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows of two integer columns (~0.6 GB), far smaller than a 6.46M-element list of variable-length integer vectors. |
| **Preserve the trained RF model** | We only restructure feature engineering; output columns have identical names and identical numerical values (max, min, mean of the same neighbor sets). The RF `predict()` call is unchanged. |

**Expected speedup**: From 86+ hours to roughly 5â€“15 minutes on the same laptop.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert cell_data to data.table (if not already) and key it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)          # non-destructive copy
cell_dt[, row_idx := .I]                     # preserve original row order

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a TIME-INVARIANT edge list from the nb object (once)
#
#     rook_neighbors_unique is a list of length 344,208 where element i
#     contains the integer indices (into id_order) of cell i's neighbors.
#     id_order[i] gives the cell id for position i.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate vectors
  from_ids <- vector("list", length(neighbors))
  to_ids   <- vector("list", length(neighbors))
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    if (length(nb) > 0L && !(length(nb) == 1L && nb[1] == 0L)) {
      from_ids[[i]] <- rep(id_order[i], length(nb))
      to_ids[[i]]   <- id_order[nb]
    }
  }
  data.table(
    from_id = unlist(from_ids, use.names = FALSE),
    to_id   = unlist(to_ids,   use.names = FALSE)
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id, to_id   (~1.37 M rows, time-invariant)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Vectorized neighbor-stat computation for one variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_dt) {

  # Columns we will create (same names as the original pipeline)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # --- a) Subset only the columns we need from cell_dt for the join ---
  #     Keying on (id, year) makes the join O(n log n) or better.
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # --- b) Cross join edges Ã— years is implicit: we join edge_dt with
  #     val_dt on (to_id = id, year) to get each neighbor's value,
  #     then aggregate by (from_id, year). ---

  # Expand edges by year via a keyed join:
  #   For every (from_id -> to_id) edge, look up to_id's value in each year.
  #   We achieve this by joining val_dt onto edge_dt by to_id = id.
  #   Because val_dt has one row per (id, year), the result automatically
  #   has one row per (from_id, to_id, year).

  # Rename for clarity before join
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id)
  setkey(edge_dt, to_id)

  # Join: for each edge, get neighbor value in every year
  neighbor_vals <- val_dt[edge_dt, on = "to_id", allow.cartesian = TRUE,
                          nomatch = NULL]
  # Result columns: to_id, year, val, from_id

  # Remove rows where the neighbor value is NA (matches original logic)
  neighbor_vals <- neighbor_vals[!is.na(val)]

  # --- c) Aggregate by (from_id, year) ---
  stats <- neighbor_vals[, .(
    nmax  = max(val),
    nmin  = min(val),
    nmean = mean(val)
  ), by = .(from_id, year)]

  # --- d) Join aggregated stats back onto cell_dt ---
  setkey(stats, from_id, year)
  setkey(cell_dt, id, year)

  cell_dt[stats, (c(col_max, col_min, col_mean)) :=
            .(i.nmax, i.nmin, i.nmean),
          on = c(id = "from_id", "year")]

  # Rows with no valid neighbors remain NA (`:=` leaves unmatched as NA
  # only if the columns are freshly created; ensure NA default)
  for (col in c(col_max, col_min, col_mean)) {
    if (!col %in% names(cell_dt)) cell_dt[, (col) := NA_real_]
  }

  # Restore val_dt name (edge_dt is shared across calls)
  setnames(val_dt, "to_id", "id")

  invisible(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Outer loop â€” iterate over the 5 source variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  compute_and_add_neighbor_features_fast(cell_dt, var_name, edge_dt)
  gc()
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Restore original row order and convert back if needed
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setorder(cell_dt, row_idx)
cell_dt[, row_idx := NULL]

# If downstream code expects a data.frame:
# cell_data <- as.data.frame(cell_dt)
# Otherwise keep cell_dt as-is for prediction.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Predict with the EXISTING trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_dt)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `build_edge_list` reads the identical `rook_neighbors_unique` nb object and `id_order` mapping. Every directed edge `(from_id â†’ to_id)` is the same. |
| **Same statistics** | `max`, `min`, `mean` are computed on the same non-NA neighbor values per `(cell, year)` group. |
| **Same column names** | Output columns follow the pattern `neighbor_max_{var}`, `neighbor_min_{var}`, `neighbor_mean_{var}` â€” adjust the pattern in the function if your original pipeline uses a different naming convention. |
| **Trained RF untouched** | No model retraining; only the feature matrix is rebuilt with identical numerical values. |

---

## Memory Budget Estimate (16 GB laptop)

| Object | Approximate Size |
|---|---|
| `cell_dt` (6.46M Ã— 110 cols) | ~5.5 GB |
| `edge_dt` (1.37M Ã— 2 int cols) | ~22 MB |
| `neighbor_vals` (worst case ~38.5M Ã— 4 cols) | ~1.2 GB |
| `stats` (up to 6.46M Ã— 4 cols) | ~0.2 GB |
| Overhead + R session | ~2 GB |
| **Total peak** | **~9 GB** âœ” fits in 16 GB |

The `gc()` call after each variable frees the intermediate join tables, keeping peak usage well within 16 GB.
END_RESPONSE>>>
