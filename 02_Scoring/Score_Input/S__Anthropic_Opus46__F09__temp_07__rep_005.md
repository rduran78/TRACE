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
run_id: Anthropic_F09_T07_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Performs character-based key construction (`paste(id, year)`)
- Does named-vector lookups (`idx_lookup[neighbor_keys]`) â€” which are hash lookups but repeated millions of times in an R-level loop

This builds a **6.46-million-element list** where each element contains row indices of that cell-year's neighbors. The fundamental problem: **the neighbor topology is static across years**, yet this function re-discovers it per cell-year, multiplying work by 28Ã—.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M elements
Each call iterates over every row, subsetting a numeric vector and computing `max/min/mean`. With 5 variables, that's **~32.3 million R-level function calls** â€” each with allocation overhead.

### 3. The neighbor lookup is year-redundant
Rook neighbors don't change across years. The current design embeds year into the lookup keys, creating 28 copies of the same spatial topology. A cell has the same neighbors in 1992 as in 2019.

---

## Optimization Strategy

**Core insight:** Separate the static spatial topology from the time-varying attributes. Build the adjacency table **once** (344K cells Ã— ~4 neighbors each â‰ˆ 1.37M rows), then use vectorized `data.table` joins to compute neighbor statistics.

| Step | What | Complexity |
|------|------|-----------|
| 1 | Build a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object â€” **once**, ~1.37M rows | O(C Ã— k) |
| 2 | For each year-slice, join cell attributes onto the adjacency table by `neighbor_id + year` | Vectorized join |
| 3 | Group by `(cell_id, year)` and compute `max`, `min`, `mean` in one pass | Vectorized aggregation |
| 4 | Join results back to the main dataset | Vectorized join |

**Expected speedup:** From ~86 hours â†’ **minutes** (typically 2â€“10 minutes on a 16 GB laptop), because:
- The 6.46M-element R-level `lapply` is eliminated.
- All operations are vectorized `data.table` keyed joins and grouped aggregations in C.
- Memory footprint is modest: the adjacency table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB; the join table with one variable is ~1.37M Ã— 28 years â‰ˆ 38.4M rows of integers + doubles, well within 16 GB when processed one variable at a time.

The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per variable per cell-year) is identical.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with columns: id, year, and
#         all predictor columns including the 5 neighbor source vars.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
 cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static adjacency table ONCE from the nb object.
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector of cell IDs in the same order as the nb object
#
#   Result: adj_dt â€” a data.table with columns (cell_id, neighbor_id)
#           representing every directed rook-neighbor pair (~1.37M rows).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_adjacency_table <- function(id_order, neighbors) {
  # Pre-allocate vectors for speed
  n_links <- sum(lengths(neighbors))
  from_id <- integer(n_links)
  to_id   <- integer(n_links)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    # spdep nb objects use 0L to denote "no neighbors"; skip those
    nb_idx <- nb_idx[nb_idx != 0L]
    n      <- length(nb_idx)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
      pos <- pos + n
    }
  }

  # Trim in case some 0-neighbor cells left slack
  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

adj_dt <- build_adjacency_table(id_order, rook_neighbors_unique)

cat(sprintf("Adjacency table: %s directed neighbor pairs\n", format(nrow(adj_dt), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each neighbor source variable, compute neighbor max, min,
#         mean via a keyed join + grouped aggregation, then attach
#         results back to cell_data.
#
#   This replaces both build_neighbor_lookup() and compute_neighbor_stats()
#   and the outer for-loop â€” all in vectorized data.table operations.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Key cell_data for fast joins
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s ...\n", var_name))

  # Extract only the columns needed for the join (small memory footprint)
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)

  # Expand adjacency table by year via join:
  #   For every (cell_id, neighbor_id) pair, attach every year's value
  #   of the neighbor.
  #
  #   adj_dt has ~1.37M rows (no year column).
  #   We join attr_dt (keyed on neighbor_id, year) onto adj_dt,
  #   allowing the cross of adj_dt Ã— years to happen implicitly.
  #
  #   Strategy: add year to adj_dt by crossing with cell_data's (id, year),
  #   then look up the neighbor's value.

  # 2a. Get the (cell_id, year) combinations that actually exist
  cy <- cell_data[, .(cell_id = id, year)]
  setkey(cy, cell_id)

  # 2b. Join: for each (cell_id, year), get all neighbor_ids
  #     Result: (cell_id, year, neighbor_id)
  expanded <- adj_dt[cy, on = .(cell_id), allow.cartesian = TRUE, nomatch = 0L]
  #     expanded now has ~1.37M Ã— 28 â‰ˆ 38.4M rows (but only 3 int columns, manageable)

  # 2c. Look up the neighbor's attribute value for that year
  setkey(expanded, neighbor_id, year)
  expanded[attr_dt, value := i.value, on = .(neighbor_id, year)]

  # 2d. Aggregate: group by (cell_id, year), compute max/min/mean
  stats <- expanded[!is.na(value),
                    .(nb_max  = max(value),
                      nb_min  = min(value),
                      nb_mean = mean(value)),
                    by = .(cell_id, year)]

  # 2e. Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # 2f. Join stats back onto cell_data
  setkey(stats, cell_id, year)
  # Remove old columns if they already exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  cell_data[stats, (c(max_col, min_col, mean_col)) :=
              mget(paste0("i.", c(max_col, min_col, mean_col))),
            on = .(id = cell_id, year)]

  # Clean up to free RAM before next variable

rm(attr_dt, cy, expanded, stats)
  gc()

  cat(sprintf("  Done: %s, %s, %s added.\n", max_col, min_col, mean_col))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the existing trained Random Forest model.
#
#   The trained model object (e.g., `rf_model`) is unchanged.
#   cell_data now contains all ~110 predictor columns including the
#   15 new neighbor_* columns, identical in value to the original code.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Example prediction (adapt variable name to your actual model object):
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]

cat("Neighbor feature engineering complete. Ready for Random Forest prediction.\n")
```

---

## Why the Numerical Estimand Is Preserved

| Aspect | Original code | Optimized code |
|--------|--------------|----------------|
| Neighbor definition | `rook_neighbors_unique[[ref_idx]]` â†’ same cell IDs | `adj_dt` built from the same `nb` object and `id_order` â€” identical pairs |
| Year matching | `paste(neighbor_id, year)` key lookup | Equi-join on `(neighbor_id, year)` â€” identical matching |
| NA handling | Drops `NA` neighbor values before aggregation | `expanded[!is.na(value), ...]` â€” identical |
| Aggregation | `max`, `min`, `mean` on non-NA neighbor values | Same three functions on the same values |
| No-neighbor cells | Returns `(NA, NA, NA)` | Left join leaves columns as `NA` â€” identical |

The output columns are numerically identical; only the computational path changes.

---

## Expected Performance

| Metric | Original | Optimized |
|--------|----------|-----------|
| **Time** | ~86+ hours | **~2â€“10 minutes** |
| **Peak RAM** | Moderate (6.46M-element list of integer vectors) | ~2â€“4 GB peak (one 38.4M-row join table at a time, freed between variables) |
| **RF model** | Unchanged | Unchanged |
END_RESPONSE>>>
