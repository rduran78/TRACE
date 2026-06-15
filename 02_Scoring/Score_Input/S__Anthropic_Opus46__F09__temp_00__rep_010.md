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
run_id: Anthropic_F09_T00_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). With ~6.46M rows, this creates billions of intermediate character objects and named-vector searches (O(n) or O(n log n) per lookup in R's hashed environments, but with enormous constant overhead from string allocation and garbage collection).

### 2. The neighbor lookup is **year-redundant**
The spatial neighbor structure is **identical across all 28 years** â€” cell A's rook neighbors are always the same cells regardless of year. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year combination, inflating the work by a factor of 28Ã—.

### 3. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via an R-level `lapply` with 6.46M iterations is inherently slow. Each iteration allocates small vectors, subsets, removes NAs, and computes three summary statistics â€” all in interpreted R.

### Summary of waste
| Bottleneck | Scale | Root cause |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string-paste + named lookups | Year-redundant; should be 344K spatial lookups only |
| `compute_neighbor_stats` | 6.46M R-level iterations Ã— 5 vars | Should be vectorized join + grouped aggregation |
| Memory | Millions of small character vectors | GC pressure from string intermediates |

---

## Optimization Strategy

**Core insight:** Build the neighbor table **once at the spatial level** (344K cells Ã— ~4 neighbors each â‰ˆ 1.37M directed edges), then **join yearly attributes** onto both sides of each edge, and compute grouped `max`, `min`, `mean` using `data.table` â€” fully vectorized, zero row-level R loops.

### Steps

1. **Flatten `rook_neighbors_unique`** (the `nb` object) into an edge list `data.table` with columns `(focal_id, neighbor_id)`. This is done once and has ~1.37M rows.

2. **Convert `cell_data` to a `data.table`** keyed on `(id, year)`.

3. **For each neighbor source variable**, join the neighbor's yearly value onto the edge list (by `neighbor_id` and `year`), then aggregate by `(focal_id, year)` to get `max`, `min`, `mean`. This is a keyed `data.table` join + grouped aggregation â€” extremely fast.

4. **Join the resulting neighbor features back** onto `cell_data`.

5. **Predict** with the existing trained Random Forest model (unchanged).

### Expected speedup

| Component | Before | After |
|---|---|---|
| Build lookup | ~hours (6.46M string ops) | <1 sec (344K integer expansion) |
| Compute stats (5 vars) | ~hours (5 Ã— 6.46M lapply) | ~30â€“90 sec (5 Ã— vectorized join+agg on 1.37MÃ—28 â‰ˆ 38M rows) |
| **Total neighbor feature engineering** | **~86+ hours** | **~2â€“5 minutes** |

RAM: The edge list Ã— years is ~38.4M rows Ã— a few columns of doubles â‰ˆ < 2 GB. Well within 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build the spatial edge list ONCE (year-invariant)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, nb_object) {
  # id_order : vector of cell IDs in the same order as the nb object
  # nb_object: spdep nb list (rook_neighbors_unique)
  #
  # Returns a data.table with columns: focal_id, neighbor_id
  
  n <- length(nb_object)
  # Pre-compute total number of edges for memory pre-allocation
  n_edges <- sum(lengths(nb_object))
  
  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- nb_object[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx != 0L]
    len    <- length(nb_idx)
    if (len > 0L) {
      idx_range <- pos:(pos + len - 1L)
      focal_id[idx_range]    <- id_order[i]
      neighbor_id[idx_range] <- id_order[nb_idx]
      pos <- pos + len
    }
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  if (pos - 1L < n_edges) {
    focal_id    <- focal_id[1:(pos - 1L)]
    neighbor_id <- neighbor_id[1:(pos - 1L)]
  }
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute neighbor features for one variable (vectorized)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_features_dt <- function(cell_dt, edges, var_name) {
  # cell_dt : data.table with key (id, year) and column var_name
  # edges   : data.table with columns (focal_id, neighbor_id)
  # var_name: character, name of the variable
  #
  # Returns cell_dt with three new columns appended:
  #   <var_name>_neighbor_max, <var_name>_neighbor_min, <var_name>_neighbor_mean
  
  # Subset to only the columns we need for the join
  vals_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(vals_dt, id, year)
  
  # Cross edges with all years present in the data
  years <- sort(unique(cell_dt$year))
  edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edge_years[, `:=`(
    focal_id    = edges$focal_id[edge_idx],
    neighbor_id = edges$neighbor_id[edge_idx]
  )]
  edge_years[, edge_idx := NULL]
  
  # Join neighbor values onto edge_years
  setkey(edge_years, neighbor_id, year)
  edge_years[vals_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]
  
  # Aggregate by (focal_id, year), dropping NAs
  agg <- edge_years[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(focal_id, year)
  ]
  
  # Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Join aggregated features back onto cell_dt
  setkey(agg, focal_id, year)
  setkey(cell_dt, id, year)
  
  # Remove these columns if they already exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  cell_dt[agg, (c(max_col, min_col, mean_col)) := mget(c(
    paste0("i.", max_col),
    paste0("i.", min_col),
    paste0("i.", mean_col)
  )), on = .(id = focal_id, year)]
  
  cell_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2b. Memory-efficient variant (processes one year at a time)
#     Use this if the full cross of edges Ã— years exceeds RAM.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_features_dt_lowmem <- function(cell_dt, edges, var_name) {
  years    <- sort(unique(cell_dt$year))
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Pre-allocate result columns with NA
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    cell_dt[, (col) := NA_real_]
  }
  
  setkey(cell_dt, id, year)
  
  for (yr in years) {
    # Subset this year's values
    yr_vals <- cell_dt[year == yr, .(id, val = get(var_name))]
    setkey(yr_vals, id)
    
    # Join neighbor values onto edges
    edge_yr <- copy(edges)
    edge_yr[yr_vals, neighbor_val := i.val, on = .(neighbor_id = id)]
    
    # Aggregate
    agg_yr <- edge_yr[
      !is.na(neighbor_val),
      .(
        nb_max  = max(neighbor_val),
        nb_min  = min(neighbor_val),
        nb_mean = mean(neighbor_val)
      ),
      by = .(focal_id)
    ]
    
    # Write back into cell_dt for this year
    idx <- cell_dt[.(agg_yr$focal_id, yr), which = TRUE, on = .(id, year)]
    set(cell_dt, i = idx, j = max_col,  value = agg_yr$nb_max)
    set(cell_dt, i = idx, j = min_col,  value = agg_yr$nb_min)
    set(cell_dt, i = idx, j = mean_col, value = agg_yr$nb_mean)
  }
  
  cell_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Full pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# --- Load pre-existing objects (assumed already in environment) ---
# cell_data               : data.frame / data.table with columns id, year, ntl, ec, ...
# id_order                : integer vector of cell IDs matching nb object order
# rook_neighbors_unique   : spdep nb object (loaded from disk)
# trained_rf_model        : the already-trained Random Forest model (DO NOT retrain)

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Step A: Build spatial edge list once (~1.37M rows, < 1 second)
cat("Building spatial edge list...\n")
edges <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edges), big.mark = ",")))

# Step B: Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  t0 <- proc.time()
  
  # Use the standard variant; switch to _lowmem if RAM is tight
  cell_data <- compute_neighbor_features_dt(cell_data, edges, var_name)
  
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("    Done in %.1f seconds\n", elapsed))
}

# Step C: Predict with the existing trained Random Forest (unchanged)
cat("Generating predictions with trained Random Forest model...\n")

# Ensure predictor columns match what the model expects
predictor_cols <- setdiff(names(trained_rf_model$forest$xlevels),  # for ranger
                          character(0))
# Generic approach: use the model's expected variable names
# For randomForest package:
if (inherits(trained_rf_model, "randomForest")) {
  predictor_cols <- rownames(trained_rf_model$importance)
}
# For ranger package:
if (inherits(trained_rf_model, "ranger")) {
  predictor_cols <- trained_rf_model$forest$independent.variable.names
}

pred_data <- cell_data[, ..predictor_cols]
cell_data[, prediction := predict(trained_rf_model, data = pred_data)$predictions]
# NOTE: adjust the predict() call syntax to match your RF package
# randomForest: predict(trained_rf_model, newdata = pred_data)
# ranger:       predict(trained_rf_model, data = pred_data)$predictions

cat("Pipeline complete.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The `max`, `min`, `mean` aggregations are computed over exactly the same set of rook-neighbor cell-year values as the original code. The edge list is a faithful flattening of the same `nb` object. NA handling (`!is.na(neighbor_val)`) mirrors the original. |
| **Trained RF model** | The model object is never modified or retrained. Only `predict()` is called. |
| **Column naming** | Output columns follow the same `<var>_neighbor_max/min/mean` convention, so the trained model's expected feature names are matched. |

---

## Expected Performance on a 16 GB Laptop

| Step | Time estimate | Peak RAM |
|---|---|---|
| `build_edge_list` | < 1 sec | ~22 MB (1.37M Ã— 2 int cols) |
| `compute_neighbor_features_dt` Ã— 5 vars | ~30â€“120 sec each | ~2.5 GB peak (38.4M row edge-year table) |
| RF prediction (6.46M rows Ã— 110 features) | Depends on forest size | Existing model footprint |
| **Total** | **~3â€“10 minutes** | **< 6 GB** |

If RAM is a concern, substitute `compute_neighbor_features_dt_lowmem`, which processes one year at a time and peaks at ~1.37M rows per iteration instead of 38.4M â€” at the cost of slightly more wall-clock time (still under ~15 minutes total).
END_RESPONSE>>>
