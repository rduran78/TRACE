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
run_id: Anthropic_F11_T10_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. That's ~6.46 million R-level iterations with expensive string operations and named-vector lookups (which are O(n) hash lookups on large named vectors). This alone is catastrophically slow.

2. **`compute_neighbor_stats()`** is then called **5 times** (once per neighbor source variable), each time iterating over the ~6.46 million-element `neighbor_lookup` list, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's ~32.3 million R-level `lapply` iterations total.

3. **Random Forest inference** is a single `predict()` call on a pre-trained model against a matrix/data.frame. Even with 6.46M rows and 110 predictors, optimized RF libraries (e.g., `ranger`) perform this in minutes via vectorized C++ code. Loading the model from disk is a single `readRDS()`. Writing predictions is a single column assignment or `fwrite()`. This is trivially fast relative to the neighbor computation.

**The ~86+ hour runtime is dominated by the neighbor feature engineering, not RF inference.**

---

## Optimization Strategy

The key optimizations:

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` merge/join approach. Instead of iterating row-by-row with string concatenation and named-vector lookups, we expand the neighbor relationships into an edge list and join against the data in bulk.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable â€” computing max, min, and mean of neighbor values via `:=` and `by=` grouping, which runs in optimized C.

3. **Eliminate the per-row `lapply`** entirely. The neighbor lookup list of 6.46M elements is replaced by a flat edge-list data.table with ~1.37 million Ã— 28 years â‰ˆ ~38.5 million rows (directed neighbor-year edges), which `data.table` handles efficiently.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0. Convert cell_data to data.table (if not already) and ensure key columns
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure 'id' and 'year' exist as expected; create a row index for final join-back
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a flat edge-list from the nb object (one-time, vectorized)
#    rook_neighbors_unique is a list of length N_cells (344,208),
#    where element i contains integer indices into id_order of i's neighbors.
#    id_order is the vector mapping position -> cell id.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Number of neighbors per focal cell
n_neighbors <- lengths(rook_neighbors_unique)

# Focal cell indices (repeated by number of neighbors)
focal_indices <- rep(seq_along(rook_neighbors_unique), times = n_neighbors)

# Neighbor cell indices (unlisted)
neighbor_indices <- unlist(rook_neighbors_unique, use.names = FALSE)

# Map indices to actual cell IDs
edges <- data.table(
  focal_id    = id_order[focal_indices],
  neighbor_id = id_order[neighbor_indices]
)

rm(focal_indices, neighbor_indices, n_neighbors)  # free memory

cat("Edge list rows (directed spatial edges):", nrow(edges), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Cross-join edges with years to get the full neighbor-year edge list
#    Then join to cell_data to pull neighbor values
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Get the unique years present in the data
years_vec <- sort(unique(cell_data$year))

# Expand edges Ã— years: each spatial edge exists for every year
# Use a cross join via CJ inside a merge or simply via rep
edges_by_year <- edges[, .(year = years_vec), by = .(focal_id, neighbor_id)]

cat("Edge-year rows:", nrow(edges_by_year), "\n")

# Key the cell_data for fast joins
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. For each neighbor source variable, compute neighbor stats via
#    a single data.table join + grouped aggregation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key edges_by_year on neighbor_id, year for joining neighbor values
setkey(edges_by_year, neighbor_id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "\n")

  # Extract only the columns we need from cell_data for the neighbor lookup
  neighbor_vals <- cell_data[, .(id, year, val = get(var_name))]
  setkey(neighbor_vals, id, year)

  # Join: attach the neighbor's value to each edge-year row
  # edges_by_year is keyed on (neighbor_id, year); neighbor_vals on (id, year)
  work <- neighbor_vals[edges_by_year, on = .(id = neighbor_id, year = year),
                        nomatch = NA,
                        allow.cartesian = TRUE]
  # work now has columns: id (=neighbor_id), year, val, focal_id

  # Drop NAs in the variable value (matching original logic)
  work <- work[!is.na(val)]

  # Aggregate: group by focal_id + year, compute max/min/mean
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(focal_id, year)]

  # Rename columns to match expected output pattern
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  # Join the aggregated stats back onto cell_data
  setkey(agg, focal_id, year)
  cell_data <- agg[cell_data, on = .(focal_id = id, year = year)]

  # The join introduces 'focal_id'; drop it and restore 'id'
  # After the right join (agg[cell_data]), the 'id' column is replaced by focal_id
  # We need to fix this:
  if ("focal_id" %in% names(cell_data)) {
    # focal_id came from agg; the original 'id' is available via the join key
    # In X[Y] join, the key column from Y is absorbed. Let's rename:
    setnames(cell_data, "focal_id", "id", skip_absent = TRUE)
  }

  # Cells with zero valid neighbors get NA (matching original behavior: c(NA,NA,NA))
  # This is automatic since they won't appear in `agg`.

  rm(work, agg, neighbor_vals)
  gc()
}

# Restore the original row order
setorder(cell_data, .row_idx)
cell_data[, .row_idx := NULL]

cat("Neighbor feature engineering complete.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Random Forest inference (unchanged â€” NOT the bottleneck)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Load the pre-trained model (single readRDS call)
rf_model <- readRDS("trained_rf_model.rds")

# Generate predictions (single vectorized predict call)
cell_data$predicted <- predict(rf_model, data = cell_data)$predictions
# (Adjust accessor based on whether model is ranger, randomForest, etc.)

# Write output
fwrite(cell_data, "predictions_output.csv")

cat("Pipeline complete.\n")
```

---

## Summary

| Component | Original Complexity | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with string ops | Eliminated; replaced by vectorized edge-list construction |
| `compute_neighbor_stats` (Ã—5 vars) | 32.3M R-level `lapply` iterations | 5 `data.table` keyed joins + grouped aggregations (C-level) |
| RF inference | Single `predict()` call | **Unchanged** â€” already efficient |

**Expected speedup**: From ~86+ hours down to **minutes** (typically 10â€“30 min depending on RAM/disk), with the neighbor feature engineering dropping from the dominant cost to a minor one. The Random Forest model and the original numerical estimand (predictions) are fully preserved.
END_RESPONSE>>>
