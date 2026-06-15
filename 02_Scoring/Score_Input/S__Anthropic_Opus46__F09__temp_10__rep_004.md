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
run_id: Anthropic_F09_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation is slow (~86+ hours) for two interconnected reasons:

### A. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
`build_neighbor_lookup` iterates row-by-row over every cell-year observation. For each of the ~6.46 million rows, it:
1. Looks up the cell's reference index.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs `paste()`-based string keys for every neighbor Ã— year combination.
4. Matches those keys back into a named character index (`idx_lookup`).

This is extremely expensive because: (a) `paste()` and named-vector lookup via character matching are slow in a tight loop; (b) the neighbor topology is **time-invariant** â€” rook contiguity doesn't change across years â€” yet the function rebuilds neighbor linkages for every single cell-year row rather than reusing a cell-level adjacency table; (c) the resulting `neighbor_lookup` is a list of 6.46 million integer vectors, consuming significant memory.

### B. Row-Level `lapply` in `compute_neighbor_stats`
For each of the 5 source variables, `compute_neighbor_stats` iterates over 6.46 million list elements, subsetting and computing `max/min/mean` one row at a time. This is called 5 times, totaling ~32.3 million R-level function invocations with no vectorization.

### Core Insight
The neighbor topology is a property of **cells**, not cell-years. There are only 344,208 cells and ~1.37 million directed neighbor pairs. The correct approach is to build a **cell-pair edge table once**, then join yearly attributes onto it and compute grouped summaries using vectorized, columnar operations (via `data.table`). This replaces millions of R-level list iterations with a handful of indexed joins and grouped aggregations.

---

## 2. Optimization Strategy

1. **Build a static edge table** (`data.table`) from `rook_neighbors_unique` with columns `(focal_id, neighbor_id)` â€” only ~1.37 million rows, built once.
2. **Key the main dataset** by `(id, year)` in `data.table`.
3. **For each source variable** (and each year implicitly), join the neighbor's attribute value onto the edge table via a keyed join, then compute `max`, `min`, `mean` grouped by `(focal_id, year)` â€” fully vectorized.
4. **Join** the resulting neighbor stats back onto the main cell-year table.
5. **Predict** with the existing trained Random Forest model as before.

This reduces the problem from 6.46M Ã— R-level list operations to a small number of `data.table` keyed joins and `by=` grouped aggregations, which are implemented in C and run orders of magnitude faster. Expected runtime: **minutes, not days**.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€ Step 0: Convert main data to data.table (if not already) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€ Step 1: Build static cell-level edge table (TIME-INVARIANT, built once) â”€
#
# rook_neighbors_unique is an nb object (list of integer vectors),
# indexed in the same order as id_order.
# id_order is the vector of cell IDs corresponding to each nb element.

build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate vectors
  n_edges <- sum(lengths(nb_obj))
  focal   <- integer(n_edges)
  neighbor <- integer(n_edges)
  pos <- 0L
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0 or integer(0) for no-neighbor cells
    if (length(nbrs) == 0 || (length(nbrs) == 1 && nbrs[1] == 0L)) next
    n <- length(nbrs)
    idx <- pos + seq_len(n)
    focal[idx]    <- id_order[i]
    neighbor[idx] <- id_order[nbrs]
    pos <- pos + n
  }
  data.table(focal_id = focal[seq_len(pos)],
             neighbor_id = neighbor[seq_len(pos)])
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
# edges: ~1,373,394 rows Ã— 2 columns â€” tiny relative to the full panel

# â”€â”€ Step 2: Key the main dataset for fast joins â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(cell_data, id, year)

# â”€â”€ Step 3: Vectorized neighbor-stat computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# For each source variable, join neighbor values via the edge table,
# compute grouped stats, and merge back.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(dt, edges, var_name) {
  # Suffixed column names matching original pipeline output
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Subset to only needed columns for the join (saves memory)
  # Neighbor attribute lookup table keyed on (id, year)
  val_dt <- dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Get unique years present in the data
  years <- unique(dt$year)

  # Cross join edges Ã— years, then look up neighbor values

  # To avoid materialising edges Ã— 28 years in one shot (~38 M rows),

  # we process in yearly batches â€” each batch is only ~1.37 M rows.

  stats_list <- vector("list", length(years))
  for (j in seq_along(years)) {
    yr <- years[j]
    # Slice neighbor values for this year
    val_yr <- val_dt[year == yr]  # keyed, so this is fast
    setkey(val_yr, id)

    # Join neighbor values onto edge table
    # edges[val_yr] would look up focal; we need neighbor values
    work <- copy(edges)
    work[, year := yr]
    # Keyed join: get the neighbor's value
    work[val_yr, neighbor_val := i.val, on = .(neighbor_id = id)]

    # Compute grouped stats per focal cell
    stats_yr <- work[!is.na(neighbor_val),
                     .(nmax  = max(neighbor_val),
                       nmin  = min(neighbor_val),
                       nmean = mean(neighbor_val)),
                     by = .(focal_id, year)]
    stats_list[[j]] <- stats_yr
  }
  stats_all <- rbindlist(stats_list)
  setnames(stats_all,
           c("nmax", "nmin", "nmean"),
           c(col_max, col_min, col_mean))

  # Merge back onto the main table
  dt[stats_all, (c(col_max, col_min, col_mean)) :=
       mget(c(col_max, col_min, col_mean)),
     on = .(id = focal_id, year = year)]

  invisible(dt)
}

# â”€â”€ Step 4: Run for all source variables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  compute_neighbor_features_dt(cell_data, edges, var_name)
}

# â”€â”€ Step 5: Predict with the already-trained Random Forest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (Model object and prediction call are unchanged.)
# cell_data$predicted <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | Not retouched. We only recompute the same input features and call `predict()`. |
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per cell-year as the original code. The edge table encodes the identical rook-neighbor topology from `rook_neighbors_unique`. The yearly keyed join ensures each neighbor's value is drawn from the correct year. |
| **Column names** | Output columns follow the same `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` naming convention. |

## 5. Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build topology | ~6.46M `paste` + named lookups â†’ hours | ~1.37M-row edge table â†’ **< 5 seconds** |
| Neighbor stats (per var) | 6.46M `lapply` iterations â†’ ~17 hours | 28 keyed joins + grouped agg â†’ **~15â€“40 seconds** |
| Total (5 vars) | **~86+ hours** | **~3â€“5 minutes** |
| Peak RAM | Large 6.46M-element list | ~1.37M edge table + yearly working copies â‰ˆ **< 2 GB** |

The key transformation: exploit the **time-invariant topology** by building the adjacency structure once at the cell level, and delegate per-year attribute computation to vectorized `data.table` joins and grouped aggregations instead of R-level row iteration.
END_RESPONSE>>>
