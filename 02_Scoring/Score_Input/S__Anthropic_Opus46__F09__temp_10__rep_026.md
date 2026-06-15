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
run_id: Anthropic_F09_T10_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### A. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With 6.46M rows, this creates billions of intermediate character objects and hash-table probes. Named vector lookup (`idx_lookup[neighbor_keys]`) is O(n) per probe against a 6.46M-element named vector, making total complexity roughly O(n Ã— k) where k is mean neighbor countâ€”catastrophically expensive.

### B. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via an R-level loop over 6.46M list elements is slow. Each iteration has R interpreter overhead, memory allocation for subsetting, and NA checks.

### C. The lookup mixes spatial topology with temporal identity unnecessarily
The rook-neighbor relationships are **purely spatial**â€”they never change across years. Yet the current code rebuilds a lookup that is indexed by (cell, year) pairs, duplicating the same spatial adjacency structure 28 times and doing string-matching across all 6.46M rows to find temporal matches.

**Key insight:** The neighbor table should be built **once** over 344,208 cells (spatial only), then joined onto the yearly panel via a vectorized merge/join. This reduces the lookup problem from 6.46M rows to 344K cells and moves all computation into vectorized operations.

---

## 2. Optimization Strategy

1. **Build a static spatial edge list once** from `rook_neighbors_unique` (the `nb` object): ~1.37M directed (cell, neighbor) pairs. This is year-invariant.
2. **Convert the panel data to `data.table`** for fast keyed joins.
3. **For each variable, join yearly attributes onto the edge list** by (neighbor_id, year), then compute grouped max/min/mean by (cell_id, year) using `data.table` aggregationâ€”fully vectorized, no R-level row loops.
4. **Merge results back** onto the main panel.

Expected speedup: The ~6.46M-row `lapply` is replaced by a ~38.4M-row (1.37M edges Ã— 28 years) `data.table` join + grouped aggregation, which runs in seconds to minutes rather than days.

---

## 3. Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a static spatial edge list ONCE
#         from the precomputed spdep::nb object
# ==============================================================
build_spatial_edge_list <- function(id_order, neighbors) {
  # neighbors is a list of integer index vectors (spdep::nb object)
  # id_order is the vector mapping position -> cell id
  edges <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx == 0L)) {
      return(NULL)
    }
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  return(edges)
}

# Build it once â€” ~1.37M rows, takes seconds
edge_list <- build_spatial_edge_list(id_order, rook_neighbors_unique)

# ==============================================================
# STEP 2: Convert panel to data.table (if not already)
# ==============================================================
cell_dt <- as.data.table(cell_data)

# ==============================================================
# STEP 3: For each variable, compute neighbor stats via
#         vectorized join + grouped aggregation
# ==============================================================
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_list, var_name) {
  # Columns we will create
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Extract only the columns we need for the join (small footprint)
  # neighbor_id will be matched to "id" in the attribute table
  attr_cols <- c("id", "year", var_name)
  attr_dt   <- cell_dt[, ..attr_cols]

  # Key the attribute table for fast join
  setkey(attr_dt, id, year)

  # Expand edge list Ã— years by joining neighbor attributes
  # For each (cell_id, neighbor_id) pair, attach the neighbor's
  # yearly value by joining on neighbor_id == id, same year.
  #
  # We do this by: merge edge_list with cell_dt years first,
  # then join neighbor attributes.

  # Get the unique years
  years <- sort(unique(cell_dt$year))

  # Cross-join edges with years: ~1.37M Ã— 28 â‰ˆ 38.4M rows
  # Memory: 38.4M Ã— 3 int/numeric cols â‰ˆ ~900 MB (fits in 16 GB)
  edge_years <- CJ(edge_idx = seq_len(nrow(edge_list)), year = years)
  edge_years[, cell_id     := edge_list$cell_id[edge_idx]]
  edge_years[, neighbor_id := edge_list$neighbor_id[edge_idx]]
  edge_years[, edge_idx    := NULL]

  # Join to get the neighbor's value of var_name
  setkey(edge_years, neighbor_id, year)
  edge_years[attr_dt, neighbor_val := get(var_name), on = .(neighbor_id = id, year)]

  # Aggregate: grouped stats by (cell_id, year)
  stats <- edge_years[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(cell_id, year)
  ]

  # Rename for merge
  setnames(stats,
           c("nb_max", "nb_min", "nb_mean"),
           c(col_max,  col_min,  col_mean))

  # Remove old columns if they already exist (idempotency)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Merge back onto the main panel
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)

  return(cell_dt)
}

# ==============================================================
# STEP 3b (memory-friendly alternative): process one variable
#          at a time WITHOUT materializing the full cross-join.
#          This uses a direct equi-join and is more RAM-efficient.
# ==============================================================
compute_and_add_neighbor_features_lean <- function(cell_dt, edge_list, var_name) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Subset to needed columns
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id)

  # For every edge (cell_id, neighbor_id), pull all years of the neighbor
  # by joining edge_list to attr_dt on neighbor_id == id
  # Result: (cell_id, neighbor_id, year, val)  ~38.4M rows
  joined <- attr_dt[edge_list, on = .(id = neighbor_id), allow.cartesian = TRUE,
                    nomatch = NULL]
  # 'joined' now has columns: id (=neighbor_id), year, val, cell_id
  # Rename for clarity
  setnames(joined, "id", "neighbor_id")

  # Aggregate by (cell_id, year), dropping NAs in val
  stats <- joined[
    !is.na(val),
    .(nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)),
    by = .(cell_id, year)
  ]

  setnames(stats,
           c("nb_max", "nb_min", "nb_mean"),
           c(col_max,  col_min,  col_mean))

  # Remove old columns if present
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Merge back
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)

  return(cell_dt)
}

# ==============================================================
# STEP 4: Run for all 5 neighbor source variables
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_and_add_neighbor_features_lean(cell_dt, edge_list, var_name)
  gc()
}

# ==============================================================
# STEP 5: Predict with the EXISTING trained Random Forest
#          (model object is untouched / not retrained)
# ==============================================================
# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(cell_dt)

# The trained model object (e.g., rf_model) is used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Gains

| Aspect | Before (Original) | After (Optimized) |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste`/named-vector probes | One-time `rbindlist` over 344K cells (~seconds) |
| **Neighbor stat computation** | `lapply` over 6.46M rows Ã— 5 vars (R interpreter loop) | Vectorized `data.table` join + `by=` aggregation (~minutes total) |
| **Memory pattern** | 6.46M-element named character vector + 6.46M-element list | ~38.4M-row `data.table` per variable, freed after each merge |
| **Estimated wall time** | ~86+ hours | **~5â€“15 minutes** (5 variables on 16 GB laptop) |
| **Trained RF model** | âœ… Preserved (not retrained) | âœ… Preserved (not retrained) |
| **Numerical estimand** | âœ… Same max/min/mean over rook neighbors | âœ… Identical values produced |

The core principle: **separate the time-invariant spatial topology from the time-varying attributes**, build the edge list once, and let `data.table`'s optimized C-level joins and grouped aggregations replace millions of R-interpreter-level iterations.
END_RESPONSE>>>
