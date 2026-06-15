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
run_id: Anthropic_F09_T00_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor Ã— that row's year (expensive string allocation).
- Matches those keys against a named character vector (`idx_lookup`) of length 6.46M (expensive named-vector lookup â€” R's named vector lookup is O(n) in the worst case or hash-based but still slow at this scale).

This means **~6.46 million iterations**, each doing string construction and hash lookups against a 6.46M-entry table. The string-key approach is the primary bottleneck.

### 2. `compute_neighbor_stats` is efficient but called on a bad foundation
The stats computation itself (subsetting a numeric vector by integer indices) is fast. But it's called on the `neighbor_lookup` list that took hours to build, and the list itself is ~6.46M elements long with duplicated spatial logic (the same cell-to-cell neighbor relationships are re-resolved for every year).

### Root Cause Summary
The neighbor **topology** is purely spatial and identical across all 28 years. Yet the current code rebuilds the row-index mapping for every cell-year combination by string-pasting IDs and years. The correct approach is:

1. Build the **spatial neighbor table once** (344,208 cells â†’ ~1.37M directed neighbor pairs).
2. For each year, **join** cell attributes onto that table and compute grouped statistics.

This reduces the problem from 6.46M string-key lookups to a simple integer-indexed join.

---

## Optimization Strategy

| Step | What | Why |
|------|------|-----|
| **A** | Build a `data.table` of directed neighbor pairs: `(cell_id, neighbor_id)` from the `nb` object â€” done **once**, ~1.37M rows. | Separates topology from time. |
| **B** | Store `cell_data` as a `data.table` keyed on `(id, year)`. | Enables fast keyed joins. |
| **C** | For each variable, join the neighbor table to cell_data by `(neighbor_id, year)` to pull neighbor values, then compute `max`, `min`, `mean` grouped by `(cell_id, year)`. | Vectorized grouped aggregation replaces 6.46M `lapply` iterations. |
| **D** | Join the resulting stats back onto `cell_data`. | Adds the ~15 new columns (5 vars Ã— 3 stats). |

**Expected speedup**: The entire neighbor-feature computation should complete in **minutes** (roughly 2â€“10 minutes depending on disk I/O and RAM pressure), not 86+ hours. The `data.table` grouped join + aggregation is highly optimized in C and operates on contiguous memory.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP A: Build the spatial neighbor-pair table ONCE
#
# Input:
#   id_order             â€” integer/character vector of cell IDs, length 344,208
#                           (same order as the nb object)
#   rook_neighbors_unique â€” spdep nb object (list of integer index vectors)
#
# Output:
#   neighbor_pairs_dt    â€” data.table with columns (cell_id, neighbor_id)
#                           ~1,373,394 rows (directed pairs)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_pairs <- function(id_order, neighbors_nb) {
  # Pre-allocate vectors for speed
  n_cells <- length(id_order)
  # Count total neighbor links
  n_links <- sum(vapply(neighbors_nb, function(x) {
    # spdep nb objects use 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(n_links)
  to_id   <- integer(n_links)
  pos     <- 1L

  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
    k <- length(nb_idx)
    from_id[pos:(pos + k - 1L)] <- id_order[i]
    to_id[pos:(pos + k - 1L)]   <- id_order[nb_idx]
    pos <- pos + k
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

neighbor_pairs_dt <- build_neighbor_pairs(id_order, rook_neighbors_unique)

cat(sprintf(
  "Neighbor pair table: %s rows (expected ~1,373,394)\n",
  format(nrow(neighbor_pairs_dt), big.mark = ",")
))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP B: Convert cell_data to data.table (if not already) and set key
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  setDT(cell_data)
}
# Ensure key columns exist and set key for fast joins
stopifnot(all(c("id", "year") %in% names(cell_data)))
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP C: Compute neighbor stats for all variables via grouped join
#
# For each source variable, we:
#   1. Expand neighbor_pairs_dt Ã— all years (cross join).
#   2. Join neighbor attribute values from cell_data.
#   3. Aggregate max/min/mean grouped by (cell_id, year).
#   4. Join results back onto cell_data.
#
# Memory note: the cross join of ~1.37M pairs Ã— 28 years = ~38.5M rows.
# With a few numeric columns this is ~300-600 MB â€” fits in 16 GB RAM.
# We process one variable at a time to limit peak memory.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Get the unique years present in the data
all_years <- sort(unique(cell_data$year))

# Build the expanded table: every neighbor pair Ã— every year
# This is the "reusable adjacency table" â€” built once, reused per variable.
years_dt <- data.table(year = all_years)
neighbor_expanded <- neighbor_pairs_dt[
  , CJ_dt := TRUE  # placeholder
][
  rep(seq_len(.N), length(all_years))
][
  , year := rep(all_years, each = nrow(neighbor_pairs_dt))
]
# Clean up placeholder
neighbor_expanded[, CJ_dt := NULL]

# More memory-efficient alternative construction:
neighbor_expanded <- CJ(pair_idx = seq_len(nrow(neighbor_pairs_dt)),
                        year = all_years)
neighbor_expanded[, `:=`(
  cell_id     = neighbor_pairs_dt$cell_id[pair_idx],
  neighbor_id = neighbor_pairs_dt$neighbor_id[pair_idx]
)]
neighbor_expanded[, pair_idx := NULL]
setkey(neighbor_expanded, neighbor_id, year)

cat(sprintf(
  "Expanded neighbor table: %s rows\n",
  format(nrow(neighbor_expanded), big.mark = ",")
))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP C (continued): Function to compute and attach neighbor features
#                      for one source variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_and_add_neighbor_features_fast <- function(cell_dt,
                                                    neighbor_exp,
                                                    var_name) {
  # 1. Extract only the columns we need for the join
  #    (neighbor_id matched to cell_data$id, same year)
  lookup_cols <- c("id", "year", var_name)
  lookup_dt   <- cell_dt[, ..lookup_cols]
  setnames(lookup_dt, "id", "neighbor_id")
  setkey(lookup_dt, neighbor_id, year)

  # 2. Join neighbor values onto the expanded neighbor table
  joined <- lookup_dt[neighbor_exp, on = .(neighbor_id, year), nomatch = NA]
  # joined now has columns: neighbor_id, year, <var_name>, cell_id

  # 3. Aggregate by (cell_id, year), dropping NAs in the variable
  stat_names <- paste0("neighbor_", c("max_", "min_", "mean_"), var_name)

  stats <- joined[
    !is.na(get(var_name)),
    .(
      V_max  = max(get(var_name), na.rm = TRUE),
      V_min  = min(get(var_name), na.rm = TRUE),
      V_mean = mean(get(var_name), na.rm = TRUE)
    ),
    by = .(cell_id, year)
  ]
  setnames(stats, c("V_max", "V_min", "V_mean"), stat_names)

  # 4. Merge back onto cell_data
  #    First remove old columns if they exist (idempotent re-runs)
  for (sn in stat_names) {
    if (sn %in% names(cell_dt)) cell_dt[, (sn) := NULL]
  }

  setkey(stats, cell_id, year)
  cell_dt <- stats[cell_dt, on = .(cell_id = id, year = year)]

  # The join above renames cell_id; fix column names
  # Actually, let's use merge for clarity:
  # Revert: use a clean merge approach
  cell_dt <- NULL  # discard the bad join above

  # Clean merge approach:
  cell_dt_out <- merge(cell_dt_input, stats,
                       by.x = c("id", "year"),
                       by.y = c("cell_id", "year"),
                       all.x = TRUE)
  return(cell_dt_out)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cleaner self-contained version (replaces the above):
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

add_neighbor_features <- function(cell_dt, neighbor_exp, var_name) {
  val_col <- var_name
  stat_max  <- paste0("neighbor_max_",  var_name)
  stat_min  <- paste0("neighbor_min_",  var_name)
  stat_mean <- paste0("neighbor_mean_", var_name)

  # Remove old columns if re-running
  for (col in c(stat_max, stat_min, stat_mean)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Build a small lookup: (neighbor_id, year) -> value
  lookup <- cell_dt[, .(neighbor_id = id, year, val = get(val_col))]
  setkey(lookup, neighbor_id, year)

  # Join values onto the expanded neighbor table
  # neighbor_exp has: cell_id, neighbor_id, year
  joined <- merge(neighbor_exp, lookup,
                  by = c("neighbor_id", "year"),
                  all.x = FALSE,   # inner join: drop if no value
                  allow.cartesian = FALSE)

  # Aggregate
  stats <- joined[
    !is.na(val),
    .(nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)),
    by = .(cell_id, year)
  ]
  setnames(stats, c("nmax", "nmin", "nmean"),
           c(stat_max, stat_min, stat_mean))

  # Merge back
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)
  return(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP D: Run for all 5 neighbor source variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s ...\n", var_name))
  t0 <- proc.time()

  cell_data <- add_neighbor_features(cell_data, neighbor_expanded, var_name)

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("  Done in %.1f seconds.\n", elapsed))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP E: Predict with the existing trained Random Forest
#         (model object is unchanged â€” no retraining)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Assuming the trained model is stored in `rf_model` and expects a
# data.frame with the ~110 predictor columns:

cell_data[, predicted := predict(rf_model, newdata = cell_data)]

cat("Pipeline complete.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Trained RF model unchanged** | We only call `predict()` â€” no `train()`, `randomForest()`, or `ranger()` call. The model object is never modified. |
| **Numerical estimand preserved** | The `max`, `min`, `mean` aggregations are computed on exactly the same neighbor sets (rook contiguity) and the same variable values. The `data.table` grouped aggregation uses the same arithmetic as the original `lapply` â€” IEEE 754 double-precision `max`, `min`, and `mean`. Results are numerically identical (or differ only at machine-epsilon level due to summation order in `mean`, which is inconsequential). |
| **Same neighbor topology** | `build_neighbor_pairs` reads the same `rook_neighbors_unique` nb object and `id_order` vector. The directed pairs are identical. |
| **NA handling identical** | Inner join + `!is.na(val)` filter replicates the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]` logic. Cells with zero valid neighbors get `NA` for all three stats via `all.x = TRUE` in the final merge. |

## Expected Performance

| Stage | Estimated Time |
|-------|---------------|
| `build_neighbor_pairs` (344K cells) | < 5 seconds |
| `CJ` expansion (~38.5M rows) | ~10â€“30 seconds |
| `add_neighbor_features` per variable (merge + group-by on 38.5M rows) | ~30â€“90 seconds each |
| All 5 variables | **~3â€“8 minutes total** |
| `predict()` on 6.46M rows | Depends on RF size (typically 5â€“30 min) |
| **Total** | **~10â€“40 minutes** vs. 86+ hours |

The speedup factor is roughly **100â€“500Ã—**, achieved entirely by replacing per-row string-key `lapply` with vectorized `data.table` keyed joins and grouped aggregation.
END_RESPONSE>>>
