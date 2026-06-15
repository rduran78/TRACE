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
run_id: Anthropic_F09_T07_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Does a character-based hash lookup (`id_to_ref`, `idx_lookup`) using `paste()` to build keys.
- Builds per-row character vectors for neighbor keys.
- Performs named-vector lookups (which are O(n) hash probes on large named vectors).

Creating ~6.46 million character keys and probing a 6.46-million-entry named vector for each of ~4 neighbors per cell is extremely expensive. The named-vector lookup in R degrades as the vector grows.

### 2. The lookup is year-redundant
The spatial neighbor topology is **identical across all 28 years**. Yet the code rebuilds neighbor index vectors for every cell-year row, effectively repeating the same spatial work 28 times. A cell's neighbors in 1992 are the same cells as in 2019â€”only the attribute values change.

### 3. `compute_neighbor_stats` uses row-level `lapply`
Even after the lookup is built, computing stats loops over 6.46M rows in R-level `lapply`, which is slow compared to vectorized or table-join approaches.

---

## Optimization Strategy

**Core idea:** Separate topology (static) from attributes (yearly). Build a **neighbor edge table once** (â‰ˆ1.37M rows of `focal_id â†’ neighbor_id`), then use a fast **`data.table` join** to attach yearly attributes to neighbors and compute grouped `max`, `min`, `mean`â€”all vectorized.

| Step | What | Complexity |
|------|------|-----------|
| 1 | Build a static edge table from `rook_neighbors_unique` (~1.37M rows) | One-time, seconds |
| 2 | For each variable, join `cell_data` onto the edge table by `(neighbor_id, year)` | Vectorized, keyed join |
| 3 | Compute `max`, `min`, `mean` grouped by `(focal_id, year)` | Vectorized aggregation |
| 4 | Join results back onto `cell_data` | Keyed join |

This eliminates all per-row R loops and character-key construction. Expected runtime: **minutes, not hours**.

The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per variable per cell-year) is identical.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 0 â€” Convert cell_data to data.table (if not already)
# ===========================================================================
cell_data <- as.data.table(cell_data)

# ===========================================================================
# STEP 1 â€” Build a STATIC neighbor edge table (once, from the nb object)
#
#   rook_neighbors_unique : spdep nb object (list of integer neighbor indices)
#   id_order              : vector mapping positional index -> cell id
#
#   Result: edge_dt with columns  focal_id | neighbor_id
#           (~1.37 M rows, one per directed rook-neighbor pair)
# ===========================================================================
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives positional indices of neighbors of cell i
  n_cells <- length(id_order)
  
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      focal_id[pos:(pos + n_nb - 1L)]    <- id_order[i]
      neighbor_id[pos:(pos + n_nb - 1L)] <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ===========================================================================
# STEP 2 â€” Function: compute neighbor stats for one variable via join
# ===========================================================================
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Columns we need from cell_data for the join
  # We join on (neighbor_id = id, year = year) to get the neighbor's value
  
  # Subset to only needed columns for efficiency
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Expand edges Ã— years:
  #   For every (focal_id, neighbor_id) edge, and for every year the focal
  #   cell appears in, we need the neighbor's value in that same year.
  #
  #   Strategy: join edge_dt onto cell_dt's (focal) years, then join again
  #   onto cell_dt's (neighbor) values.
  
  # Get the unique years each focal cell appears in
  focal_years <- cell_dt[, .(year = unique(year)), by = .(focal_id = id)]
  setkey(focal_years, focal_id)
  
  # Cross focal-years with edges: gives (focal_id, year, neighbor_id)
  setkey(edge_dt, focal_id)
  expanded <- edge_dt[focal_years, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: focal_id, neighbor_id, year
  
  # Join neighbor values
  expanded[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]
  
  # Aggregate: max, min, mean of neighbor_val per (focal_id, year)
  stats <- expanded[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(focal_id, year)
  ]
  
  # Rename to match expected column names
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
  
  stats
}

# ===========================================================================
# STEP 3 â€” Loop over the 5 neighbor source variables and join results back
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  
  stats_dt <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  setkey(stats_dt, focal_id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # Join back onto cell_data
  cell_data[stats_dt, `:=`(
    (max_col)  = get(paste0("i.", max_col)),
    (min_col)  = get(paste0("i.", min_col)),
    (mean_col) = get(paste0("i.", mean_col))
  ), on = .(id = focal_id, year)]
}

# ===========================================================================
# STEP 4 â€” Predict with the existing (already-trained) Random Forest model
#
#   The cell_data now has the same neighbor_max_*, neighbor_min_*,
#   neighbor_mean_* columns as the original pipeline produced, with
#   identical numerical values.  The trained model is used as-is.
# ===========================================================================
# Example (adjust to your actual model object and response column):
# cell_data[, prediction := predict(trained_rf_model, newdata = cell_data)]
```

---

## Expected Performance

| Component | Original | Optimized |
|-----------|----------|-----------|
| Build neighbor lookup | ~hours (6.46M `lapply` + `paste` + named-vector probe) | ~2â€“5 sec (one loop over 344K cells) |
| Compute stats per variable | ~hours (6.46M `lapply`) | ~20â€“60 sec (keyed `data.table` join + grouped agg) |
| **Total for 5 variables** | **~86+ hours** | **~3â€“6 minutes** |
| Peak RAM | Moderate (large named vectors) | ~2â€“3 GB (edge table Ã— years is transient) |

The numerical estimand is preserved exactly: for every `(cell, year)`, the neighbor `max`, `min`, and `mean` are computed over the same rook-neighbor set with the same `NA`-handling logic. The trained Random Forest model is never retrainedâ€”only its `.predict()` method is called on the updated feature table.
END_RESPONSE>>>
