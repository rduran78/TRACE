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
run_id: Anthropic_F09_T10_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of expensive string operations and named-vector indexing (which is O(n) in base R for each lookup).

### 2. Redundant recomputation of static topology
Rook-neighbor relationships are purely spatial â€” they do not change across years. Yet `build_neighbor_lookup` embeds year into the lookup keys, effectively rebuilding the neighbor graph 28 times (once per year per cell). The 1,373,394 directed rook-neighbor relationships are the same in 1992 as they are in 2019.

### 3. Row-level `lapply` over 6.46M rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean one row at a time in R (not vectorized) is painfully slow for 6.46M rows Ã— 5 variables = ~32.3M iterations.

**Root cause summary:** The topology (which cell neighbors which cell) is static, but the code entangles it with yearly data, forcing millions of redundant string operations and preventing vectorized computation.

---

## Optimization Strategy

**Core idea:** Separate topology from attributes. Build the neighbor edge-list once (344K cells Ã— ~4 neighbors each), then use a vectorized `data.table` join-and-aggregate strategy to compute neighbor stats for all years simultaneously.

| Step | What | Complexity |
|------|------|------------|
| 1 | Build a **cell-level edge-list** `(cell_id, neighbor_id)` from `rook_neighbors_unique` â€” done **once**, ~1.37M rows. | O(cells Ã— avg_neighbors) |
| 2 | Represent `cell_data` as a `data.table` keyed on `(id, year)`. | O(n) |
| 3 | For each variable, **join** the edge-list with yearly attributes: left side = `(cell_id, year)`, joined to neighbor attributes via `(neighbor_id, year)`. This produces ~1.37M Ã— 28 â‰ˆ 38.5M rows, but the join is vectorized in C. | O(edges Ã— years) |
| 4 | **Group-by aggregate** `(cell_id, year)` to get `max`, `min`, `mean` â€” fully vectorized in `data.table`. | O(edges Ã— years) |
| 5 | Join the three new columns back onto `cell_data`. | O(n) |

**Expected speedup:** From ~86 hours to **minutes** (typically 2â€“10 minutes on a 16 GB laptop), because all loops are replaced by vectorized C-level `data.table` operations.

**Preserves:**
- The trained Random Forest model (untouched).
- The original numerical estimand (identical max, min, mean values â€” same arithmetic, just computed vectorized).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the static cell-level neighbor edge-list ONCE
# ============================================================
# rook_neighbors_unique : spdep nb object (list of integer vectors)
# id_order              : vector of cell IDs, position i corresponds
#                         to the i-th element of rook_neighbors_unique

build_neighbor_edgelist <- function(id_order, neighbors) {
  # Pre-allocate: count total number of directed neighbor pairs
  n_links <- sum(lengths(neighbors))

  from_id <- integer(n_links)
  to_id   <- integer(n_links)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx != 0L]
    n      <- length(nb_idx)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
      pos <- pos + n
    }
  }

  # Trim in case some 0-neighbor entries existed
  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

cat("Building static neighbor edge-list...\n")
neighbor_edges <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
cat(sprintf("  Edge-list rows: %s\n", format(nrow(neighbor_edges), big.mark = ",")))

# ============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 3: Vectorized neighbor-stat computation
# ============================================================
compute_neighbor_features_dt <- function(dt, edges, var_name) {
  # Build a slim lookup table: (id, year, value)
  lookup <- dt[, .(id, year, value = get(var_name))]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)

  # Cross the edge-list with all years present in the data
  years <- sort(unique(dt$year))
  edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edge_years[, cell_id     := edges$cell_id[edge_idx]]
  edge_years[, neighbor_id := edges$neighbor_id[edge_idx]]
  edge_years[, edge_idx := NULL]

  # Join neighbor attribute values
  setkey(edge_years, neighbor_id, year)
  edge_years <- lookup[edge_years, on = .(neighbor_id, year)]

  # Aggregate: max, min, mean per (cell_id, year)
  agg <- edge_years[!is.na(value),
                    .(nb_max  = max(value),
                      nb_min  = min(value),
                      nb_mean = mean(value)),
                    by = .(cell_id, year)]

  # Rename columns to match expected feature names
  max_col  <- paste0("max_neighbor_",  var_name)
  min_col  <- paste0("min_neighbor_",  var_name)
  mean_col <- paste0("mean_neighbor_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  agg
}

# ============================================================
# STEP 4: Loop over the 5 source variables, join results back
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for '%s'...\n", var_name))
  agg <- compute_neighbor_features_dt(cell_data, neighbor_edges, var_name)
  setkey(agg, cell_id, year)

  # Left-join the three new columns onto cell_data
  max_col  <- paste0("max_neighbor_",  var_name)
  min_col  <- paste0("min_neighbor_",  var_name)
  mean_col <- paste0("mean_neighbor_", var_name)

  # Remove old columns if they exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- agg[cell_data, on = .(cell_id = id, year = year)]
  setnames(cell_data, "cell_id", "id")
  setkey(cell_data, id, year)

  cat(sprintf("  Done â€” added %s, %s, %s\n", max_col, min_col, mean_col))
}

# ============================================================
# STEP 5: Predict with the existing trained Random Forest
# ============================================================
# The model object (e.g., `rf_model`) is unchanged.
# cell_data now has the same neighbor-stat columns as before,
# with identical numerical values, just computed ~500Ã— faster.

cat("Computing predictions with trained Random Forest model...\n")
cell_data[, prediction := predict(rf_model, newdata = cell_data)]
cat("Done.\n")
```

---

## Memory-Optimized Variant (for tight 16 GB RAM)

The cross-join `edges Ã— years` can reach ~38.5M rows. If memory is tight, process one variable at a time (already done above) **and** process in year-chunks:

```r
compute_neighbor_features_chunked <- function(dt, edges, var_name, chunk_size = 7L) {
  years <- sort(unique(dt$year))
  chunks <- split(years, ceiling(seq_along(years) / chunk_size))

  max_col  <- paste0("max_neighbor_",  var_name)
  min_col  <- paste0("min_neighbor_",  var_name)
  mean_col <- paste0("mean_neighbor_", var_name)

  agg_list <- lapply(chunks, function(yr_chunk) {
    lookup <- dt[year %in% yr_chunk, .(neighbor_id = id, year, value = get(var_name))]
    setkey(lookup, neighbor_id, year)

    edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = yr_chunk)
    edge_years[, cell_id     := edges$cell_id[edge_idx]]
    edge_years[, neighbor_id := edges$neighbor_id[edge_idx]]
    edge_years[, edge_idx := NULL]
    setkey(edge_years, neighbor_id, year)

    edge_years <- lookup[edge_years, on = .(neighbor_id, year)]

    edge_years[!is.na(value),
               .(nb_max = max(value), nb_min = min(value), nb_mean = mean(value)),
               by = .(cell_id, year)]
  })

  agg <- rbindlist(agg_list)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  agg
}
```

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Neighbor topology | Rebuilt per cell-year (6.46M string ops) | Built once as edge-list (~1.37M rows) |
| Stat computation | Row-level `lapply` in R | Vectorized `data.table` join + group-by in C |
| Estimated runtime | ~86+ hours | ~2â€“10 minutes |
| Numerical results | max, min, mean | Identical max, min, mean |
| RF model | Unchanged | Unchanged |
| Peak RAM | Moderate (but slow) | ~4â€“6 GB (fits 16 GB) |
END_RESPONSE>>>
