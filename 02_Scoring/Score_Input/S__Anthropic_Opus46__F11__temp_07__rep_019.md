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
run_id: Anthropic_F11_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence points squarely to the **spatial neighbor feature engineering** stage â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** calls `lapply()` over every row in the dataset (~6.46 million iterations). Each iteration performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is **O(n)** hash-table probing per call, and doing this 6.46 million times with string construction each time is extraordinarily expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), and each call iterates over all ~6.46 million rows via `lapply`, subsetting and computing `max/min/mean` on neighbor values. That's ~32.3 million R-level loop iterations just for stats computation.

3. The `do.call(rbind, result)` at the end of `compute_neighbor_stats` binds ~6.46 million small vectors into a matrix â€” another known slow pattern in R.

4. **In contrast**, Random Forest prediction (`predict()`) on a pre-trained model with ~6.46M rows and ~110 predictors is a single vectorized C/C++ call (in `ranger` or `randomForest`). It typically completes in seconds to minutes, not hours. Loading a serialized model (`readRDS`) is also fast. Writing predictions is trivial.

**Conclusion:** The **86+ hour runtime** is dominated by the row-level `lapply` loops with repeated string operations and named-vector lookups across 6.46 million rows Ã— (1 build + 5 stat passes). The Random Forest step is a negligible fraction of total time.

---

## Optimization Strategy

1. **Eliminate string-keyed lookups entirely.** Replace the `paste(id, year, sep="_")` â†’ named-vector lookup with integer-arithmetic indexing. Since we have a panel with known `id_order` (344,208 cells) and known years (1992â€“2019, 28 years), every row's position can be computed as `(id_index - 1) * 28 + (year - 1991)` if the data is sorted by (id, year). This turns O(1)-amortized hash lookups into O(1) true arithmetic lookups.

2. **Vectorize neighbor stats using `data.table` or matrix operations.** Instead of looping row-by-row, "explode" the neighbor relationships into an edge table (source_row â†’ neighbor_row), join on values, and compute grouped aggregates with `data.table` â€” all in vectorized C code.

3. **Build the neighbor-row mapping once as an integer edge list**, not a list-of-lists with string keys.

This reduces the estimated runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0 â€” Ensure data is a data.table sorted by (id, year)
# ============================================================
cell_dt <- as.data.table(cell_data)

# Confirm years are contiguous 1992-2019
stopifnot(all(sort(unique(cell_dt$year)) == 1992:2019))

year_min  <- 1992L
n_years   <- 28L
n_cells   <- length(id_order)  # 344,208

# Create integer id index: position of each id in id_order
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

# Sort data by (id index, year) so row position is deterministic
cell_dt[, id_idx := id_to_idx[as.character(id)]]
setorder(cell_dt, id_idx, year)

# Now row number for (id_idx=i, year=y) is: (i - 1) * n_years + (y - year_min + 1)
# Verify:
cell_dt[, expected_row := (.I)]
cell_dt[, computed_row := (id_idx - 1L) * n_years + (year - year_min + 1L)]
stopifnot(all(cell_dt$expected_row == cell_dt$computed_row))
cell_dt[, c("expected_row", "computed_row") := NULL]

# ============================================================
# STEP 1 â€” Build integer edge list (source_row -> neighbor_row)
#           one entry per (source_cell, neighbor_cell, year)
# ============================================================
# rook_neighbors_unique is an nb object: list of length n_cells,
# each element is an integer vector of neighbor indices into id_order.

message("Building edge list...")

# For each cell i, get its neighbor cell indices
# We need edges: for every year, source_row -> neighbor_row

# Build cell-level edge list first (no year expansion yet)
edge_cell <- rbindlist(lapply(seq_len(n_cells), function(i) {

  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L) return(NULL)
  data.table(src_id_idx = i, nbr_id_idx = as.integer(nb_idx))
}))

# Now expand to all years via integer arithmetic (no joins, no strings)
# source_row = (src_id_idx - 1) * n_years + year_offset
# neighbor_row = (nbr_id_idx - 1) * n_years + year_offset
# where year_offset = 1..28

message("Expanding edge list across years...")

year_offsets <- 1L:n_years

# Use a cross join: each cell-level edge Ã— each year_offset
edge_cell[, dummy := 1L]
yr_dt <- data.table(year_offset = year_offsets, dummy = 1L)

edges <- edge_cell[yr_dt, on = "dummy", allow.cartesian = TRUE]
edges[, dummy := NULL]

edges[, source_row   := (src_id_idx - 1L) * n_years + year_offset]
edges[, neighbor_row := (nbr_id_idx - 1L) * n_years + year_offset]

# Keep only the columns we need
edges <- edges[, .(source_row, neighbor_row)]

message(sprintf("Edge list: %s rows", format(nrow(edges), big.mark = ",")))

# ============================================================
# STEP 2 â€” Vectorized neighbor stats for each variable
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_neighbor_features_fast <- function(dt, edges, var_name) {
  message(sprintf("  Computing neighbor features for: %s", var_name))
  
  # Extract neighbor values via integer indexing (vectorized)
  vals <- dt[[var_name]]
  edges_work <- copy(edges)
  edges_work[, nbr_val := vals[neighbor_row]]
  
  # Drop NAs in neighbor values
  edges_work <- edges_work[!is.na(nbr_val)]
  
  # Grouped aggregation â€” fully vectorized in data.table C code
  stats <- edges_work[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = source_row]
  
  # Initialize result columns with NA
  n <- nrow(dt)
  col_max  <- paste0(var_name, "_nb_max")
  col_min  <- paste0(var_name, "_nb_min")
  col_mean <- paste0(var_name, "_nb_mean")
  
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  # Assign by integer row index
  set(dt, i = stats$source_row, j = col_max,  value = stats$nb_max)
  set(dt, i = stats$source_row, j = col_min,  value = stats$nb_min)
  set(dt, i = stats$source_row, j = col_mean, value = stats$nb_mean)
  
  invisible(dt)
}

for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_fast(cell_dt, edges, var_name)
}

# ============================================================
# STEP 3 â€” Random Forest prediction (unchanged, fast already)
# ============================================================
# The trained model is preserved exactly as-is.
# rf_model <- readRDS("path/to/trained_rf_model.rds")
# cell_dt[, prediction := predict(rf_model, newdata = cell_dt)$predictions]
# fwrite(cell_dt[, .(id, year, prediction)], "predictions.csv")

message("Neighbor feature engineering complete.")

# Convert back to data.frame if downstream code requires it
cell_data <- as.data.frame(cell_dt)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup mechanism** | `paste()` + named-vector string hash, per row | Integer arithmetic `(id_idx-1)*28 + year_offset` |
| **Neighbor stats** | `lapply` over 6.46M rows Ã— 5 vars | `data.table` grouped aggregation on integer edge list |
| **Result binding** | `do.call(rbind, 6.46M-element list)` | Direct `set()` by integer row index |
| **Estimated runtime** | **86+ hours** | **~5â€“15 minutes** on the same laptop |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical results** | Same max/min/mean per row | Identical âœ“ |

The bottleneck was never Random Forest inference â€” it was **millions of R-level loop iterations with string construction and hash lookups** in the neighbor feature engineering step.
END_RESPONSE>>>
