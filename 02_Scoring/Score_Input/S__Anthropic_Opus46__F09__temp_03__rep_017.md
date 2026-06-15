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
run_id: Anthropic_F09_T03_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

1. **`build_neighbor_lookup` operates at the cell-year level (6.46M rows), not the cell level (344K cells).** The neighbor topology is *time-invariant*â€”cell 42's rook neighbors are the same in 1992 as in 2019. Yet the function builds a lookup of length 6.46 million, repeating the same spatial neighbor resolution 28 times. The `paste()`/`match()` keying through `idx_lookup` for every row is O(n) string hashing over millions of entries.

2. **`compute_neighbor_stats` uses an R-level `lapply` over 6.46M rows**, each calling `max`, `min`, `mean` on small vectors. The per-element R interpreter overhead (function dispatch, NA handling, memory allocation, `rbind`) dominates. For 5 variables Ã— 6.46M rows, this is ~32.3 million R-level loop iterations.

3. **No vectorization or join-based strategy is used.** The entire computation is expressible as a single table join + grouped aggregation, which `data.table` can execute in seconds.

### Core Insight

The neighbor graph is purely spatial. Build it **once at the cell level** (344K cells, ~1.37M directed edges), store it as an edge list, then for each year join the cell attributes onto both ends of every edge, and compute grouped `max`, `min`, `mean` by `(cell, year)`. This replaces millions of R-level list lookups with vectorized `data.table` grouped operations.

---

## Optimization Strategy

| Step | What | Complexity |
|------|------|------------|
| 1 | Convert `rook_neighbors_unique` (spdep nb) to a two-column edge table: `(cell_id, neighbor_id)`. ~1.37M rows. **Done once.** | O(E) |
| 2 | Convert `cell_data` to a `data.table`, keyed on `(id, year)`. | O(N) |
| 3 | For each of the 5 variables, join yearly attribute values onto the edge table by `(neighbor_id, year)`, then compute `max`, `min`, `mean` grouped by `(cell_id, year)`. | O(E Ã— Y) â‰ˆ 38.5M rows, fully vectorized |
| 4 | Join the resulting stats back onto `cell_data`. | O(N) |
| 5 | Predict with the existing trained Random Forest model (unchanged). | unchanged |

**Expected speedup:** From ~86 hours to **minutes** (typically 2â€“10 minutes on a 16 GB laptop), because the inner loop is replaced by `data.table` vectorized joins and grouped aggregations.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a time-invariant edge table from the nb object
#         (done ONCE, reusable across all years and variables)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, nb_obj) {
  # id_order: vector of cell IDs in the same order as nb_obj
  # nb_obj:   spdep nb object (list of integer neighbor indices)
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nbrs <- nb_obj[[i]]
    # spdep uses 0L to denote "no neighbors"
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nbrs])
  }))
  edges
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has columns: cell_id, neighbor_id
# ~1,373,394 rows (directed rook edges)

cat("Edge table rows:", nrow(edge_table), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor stats for all variables via joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_table, vars) {
  # Subset only the columns we need for the join: id, year, + source vars
  join_cols <- c("id", "year", vars)
  # This is the "attribute table" we join onto the neighbor side of each edge

  attr_dt <- cell_data[, ..join_cols]
  setnames(attr_dt, "id", "neighbor_id")
  
  # Key for fast join
  setkey(attr_dt, neighbor_id, year)
  
  # Expand edges Ã— years: cross join edge_table with unique years
  years <- sort(unique(cell_data$year))
  edge_year <- CJ_dt(edge_table, years)
  
  # Join neighbor attributes onto edge_year
  setkey(edge_year, neighbor_id, year)
  edge_year <- attr_dt[edge_year, on = .(neighbor_id, year), nomatch = NA]
  
  # Compute grouped stats: max, min, mean per (cell_id, year) for each var
  agg_exprs <- unlist(lapply(vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  # Build the aggregation call
  stats <- edge_year[,
    setNames(lapply(vars, function(v) {
      vv <- get(v)
      vv <- vv[!is.na(vv)]
      if (length(vv) == 0L) list(NA_real_, NA_real_, NA_real_)
      else list(max(vv), min(vv), mean(vv))
    }), vars),
    by = .(cell_id, year)
  ]
  
  # The above is elegant but let's use a more straightforward and
  # performant approach â€” direct aggregation per variable:
  
  result_list <- vector("list", length(vars))
  
  for (vi in seq_along(vars)) {
    v <- vars[vi]
    max_name  <- paste0("neighbor_max_", v)
    min_name  <- paste0("neighbor_min_", v)
    mean_name <- paste0("neighbor_mean_", v)
    
    # Aggregate
    agg <- edge_year[!is.na(get(v)),
      .(
        V_max  = max(get(v)),
        V_min  = min(get(v)),
        V_mean = mean(get(v))
      ),
      by = .(cell_id, year)
    ]
    setnames(agg, c("V_max", "V_min", "V_mean"),
                  c(max_name, min_name, mean_name))
    result_list[[vi]] <- agg
    cat("  Done:", v, "\n")
  }
  
  result_list
}

# Helper: cross join edge_table with years vector
CJ_dt <- function(edge_table, years) {
  # Repeat each edge for every year
  n_edges <- nrow(edge_table)
  n_years <- length(years)
  idx <- rep(seq_len(n_edges), times = n_years)
  yr  <- rep(years, each = n_edges)
  out <- edge_table[idx]
  out[, year := yr]
  out
}

cat("Computing neighbor features...\n")
t0 <- proc.time()

stats_list <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

cat("Aggregation time:", (proc.time() - t0)[3], "seconds\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Join stats back onto cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

setkey(cell_data, id, year)

for (agg_dt in stats_list) {
  setnames(agg_dt, "cell_id", "id")
  setkey(agg_dt, id, year)
  # Merge new columns onto cell_data
  new_cols <- setdiff(names(agg_dt), c("id", "year"))
  # Remove old columns if they exist (idempotent re-runs)
  for (cc in new_cols) {
    if (cc %in% names(cell_data)) cell_data[, (cc) := NULL]
  }
  cell_data <- agg_dt[cell_data, on = .(id, year)]
}

cat("All neighbor features attached. Columns:", ncol(cell_data), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the EXISTING trained Random Forest
#         (model object is unchanged / not retrained)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Ensure column names match what the trained model expects.
# If your trained model was built on a data.frame, convert:
predict_df <- as.data.frame(cell_data)

# Generate predictions (preserves original numerical estimand)
cell_data[, predicted := predict(trained_rf_model, newdata = predict_df)]

cat("Prediction complete.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and the same variable values. The only change is *how* the grouping is executed (vectorized join vs. R-level list indexing). |
| **Trained RF model** | The model object (`trained_rf_model`) is never modified or retrained. Only `predict()` is called. |
| **Edge fidelity** | `build_edge_table` reads the same `rook_neighbors_unique` nb object and the same `id_order`, producing identical directed edges. |
| **NA handling** | Edges with `NA` attribute values are excluded before aggregation (`!is.na(get(v))`), and cells with zero valid neighbors receive `NA` via the left joinâ€”identical to the original `compute_neighbor_stats` logic. |

---

## Expected Performance

| Component | Original | Optimized |
|-----------|----------|-----------|
| Neighbor lookup construction | ~hours (6.46M string keys) | ~2 sec (344K-row edge table) |
| Edge Ã— year expansion | N/A | ~30 sec, ~38.5M rows |
| Join + aggregation (5 vars) | ~80+ hours (R-level lapply) | ~2â€“5 min (data.table vectorized) |
| **Total** | **~86+ hours** | **~3â€“8 minutes** |

The key transformation: **separate the time-invariant spatial topology from the time-varying attributes**, join them with `data.table`, and let vectorized C-level grouped aggregation replace millions of R interpreter iterations.
END_RESPONSE>>>
