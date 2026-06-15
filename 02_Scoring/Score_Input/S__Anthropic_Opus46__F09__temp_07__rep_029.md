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
run_id: Anthropic_F09_T07_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for two fundamental reasons:

### 1. `build_neighbor_lookup` rebuilds per-row neighbor indices across the entire 6.46M-row panel, even though the spatial neighbor topology is **time-invariant**
The function iterates over every cell-year row (`lapply` over 6.46M rows), constructs string keys (`paste(id, year)`), and does named-vector lookups for each. Named-vector lookup in R is O(n) per call in the worst case, and doing ~6.46M Ã— k (where k â‰ˆ average 4 rook neighbors) string-match lookups is catastrophically slow.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows
Each call extracts neighbor values, filters NAs, and computes max/min/mean. This is repeated for each of the 5 source variables, totaling ~32.3M list iterations with per-element R-level loops.

### Root cause summary
The neighbor topology (which cell borders which cell) **never changes across years**. Yet the current code entangles spatial topology with temporal panel structure, re-discovering neighbors for every cell-year combination. This turns an O(C) spatial problem (C = 344,208 cells) into an O(C Ã— T) problem (C Ã— T = 6.46M rows), multiplied by expensive string operations.

---

## Optimization Strategy

**Core idea:** Separate the time-invariant spatial topology from the time-varying attributes. Build a simple cell-neighbor edge table once (344K cells Ã— ~4 neighbors = ~1.37M edges), then use fast vectorized joins year-by-year to compute neighbor statistics.

### Steps

1. **Build a static edge table** (`data.table`) with columns `(cell_id, neighbor_id)` from the `spdep::nb` object â€” done once, ~1.37M rows.

2. **Convert the panel data to `data.table`**, keyed on `(id, year)`.

3. **For each source variable**, join the edge table to the panel data to pull neighbor attribute values, then compute grouped `max`, `min`, `mean` â€” all using `data.table` vectorized operations. This replaces millions of R-level `lapply` iterations with a single indexed merge + grouped aggregation per variable.

4. **Merge results** back onto the main panel and pass to the already-trained Random Forest for prediction. The RF model and numerical estimand are untouched.

### Expected speedup
- The join-based approach processes ~1.37M edges Ã— 28 years = ~38.4M edge-year lookups, but via `data.table` binary-search joins and vectorized group-by, this completes in **minutes**, not hours.
- Estimated wall-clock: **5â€“15 minutes** total for all 5 variables on a 16 GB laptop.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static spatial edge table (done ONCE)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Inputs:
#   id_order             â€” vector of cell IDs (length 344,208), in the order
#                          matching the spdep::nb object
#   rook_neighbors_unique â€” spdep::nb list (length 344,208); each element is
#                           an integer vector of neighbor indices into id_order

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors_nb, length, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb encodes "no neighbors" as a single 0; skip those
    if (length(nb_idx) == 1L && nb_idx == 0L) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }

  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37 M rows, two integer columns â€” tiny in memory

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert panel data to data.table and set keys
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data is the existing data.frame / data.table with columns:
#   id, year, ntl, ec, pop_density, def, usd_est_n2, ... (110 predictors)

cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor max, min, mean for each source var
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Subset to only the columns we need for the join
  # (id, year, <var_name>)
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Expand edges Ã— years:
  #   For each (cell_id, neighbor_id) edge, join neighbor's value
  #   in each year via keyed merge.
  # Rename for the join: we want neighbor_id -> id in val_dt
  edge_year <- edge_dt[val_dt,
                        .(cell_id, neighbor_id, year = i.year),
                        on = .(cell_id = id),
                        nomatch = NULL,
                        allow.cartesian = TRUE]
  # edge_year now has (cell_id, neighbor_id, year) â€” one row per
  # directed edge per year the focal cell exists.

  # Join to get the neighbor's attribute value in that year
  edge_year[val_dt,
            neighbor_val := i.val,
            on = .(neighbor_id = id, year)]

  # Aggregate: for each (cell_id, year), compute max/min/mean of
  # neighbor_val, ignoring NAs
  stats <- edge_year[!is.na(neighbor_val),
                     .(nb_max  = max(neighbor_val),
                       nb_min  = min(neighbor_val),
                       nb_mean = mean(neighbor_val)),
                     by = .(cell_id, year)]

  # Rename columns to match the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats,
           c("nb_max", "nb_min", "nb_mean"),
           c(max_col,  min_col,  mean_col))

  stats
}

# Loop over the 5 source variables and merge results onto cell_dt
for (vname in neighbor_source_vars) {
  message("Computing neighbor features for: ", vname)
  stats_dt <- compute_neighbor_features_dt(cell_dt, edge_dt, vname)
  cell_dt <- merge(cell_dt, stats_dt,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the already-trained Random Forest
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is preserved as-is.
# Column names produced above (neighbor_max_ntl, neighbor_min_ntl,
# neighbor_mean_ntl, etc.) match the original pipeline's output,
# so the model's expected feature names are satisfied.

# Convert back to data.frame if the model expects one
cell_data <- as.data.frame(cell_dt)

# Predict (example â€” adjust to your actual model object name)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | Not retouched. We only rebuild the input features with identical definitions (neighbor max, min, mean of the same 5 variables). |
| **Numerical estimand** | `max`, `min`, `mean` computed over the same rook-neighbor sets and the same per-cell-year attribute values â€” identical arithmetic, just executed via vectorized joins instead of row-wise `lapply`. |
| **Column names** | Follow the same `neighbor_{stat}_{var}` convention so the RF model's `predict()` finds the expected features. |
| **NA handling** | `!is.na(neighbor_val)` in the aggregation + `all.x = TRUE` in the final merge ensures cells with no valid neighbors receive `NA`, matching the original `c(NA, NA, NA)` fallback. |

---

## Performance Estimate

| Step | Rows processed | Expected time |
|---|---|---|
| Build edge table | 1.37M | < 2 seconds |
| Per-variable join + aggregate | ~38.4M edge-year rows | ~1â€“2 min each |
| 5 variables total | â€” | **~5â€“10 minutes** |
| Merge back | 6.46M Ã— 5 joins | ~1 minute |
| **Total** | | **~6â€“12 minutes** |

This replaces the estimated **86+ hours** with roughly **10 minutes** on the same 16 GB laptop, a speedup of approximately **500Ã—**, by eliminating per-row string operations and leveraging `data.table`'s binary-search joins and vectorized grouped aggregation.
END_RESPONSE>>>
