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
run_id: Anthropic_F09_T03_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates over every cell-year row individually, performing string pasting (`paste(id, year, sep="_")`), named-vector lookups (`id_to_ref`, `idx_lookup`), and subsetting for each of the ~6.46 million rows. Named-vector lookups in R are hash-table operations with non-trivial per-call overhead; doing this billions of times (each row looks up multiple neighbor keys) is catastrophic.

### 2. Redundant Recomputation of Spatial Topology Per Year
The rook-neighbor structure is **purely spatial**â€”it is identical across all 28 years. Yet `build_neighbor_lookup` rebuilds neighbor index vectors for every cell-year row, effectively recomputing the same spatial adjacency 28 times and entangling it with the temporal dimension unnecessarily.

### 3. Row-Level `lapply` in `compute_neighbor_stats`
For each of the 6.46 million rows, `compute_neighbor_stats` subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. The per-call overhead of `lapply` + anonymous function + subsetting + three summary functions, repeated 6.46M Ã— 5 variables = ~32.3 million invocations, dominates runtime.

**In summary:** The architecture conflates spatial structure (fixed) with temporal attributes (varying), and uses row-level R loops where vectorized or table-join operations should be used.

---

## Optimization Strategy

**Core insight:** Build the neighbor table **once** as a spatial-only edge list (cell â†’ neighbor_cell), then use a vectorized `data.table` join to bring in yearly attributes and compute grouped statistics.

### Steps:

1. **Build a static edge list** from `rook_neighbors_unique` (the `nb` object). This produces a two-column data.table: `(id, neighbor_id)`. This is done once and has ~1.37M rows.

2. **Join yearly attributes onto the edge list.** For each year, every edge `(id, neighbor_id)` gets the neighbor's attribute value by joining `cell_data` on `(neighbor_id, year)`. This is a keyed `data.table` equi-joinâ€”extremely fast.

3. **Compute grouped statistics** using `data.table`'s `[, .(max, min, mean), by=.(id, year)]`â€”fully vectorized C-level aggregation.

4. **Join results back** to the main data.table.

This eliminates all row-level R loops. The entire pipeline becomes a sequence of keyed joins and grouped aggregations, reducing runtime from ~86 hours to **minutes**.

### Complexity comparison:

| Step | Current | Proposed |
|---|---|---|
| Neighbor lookup | O(6.46M) R-level iterations with string ops | O(1.37M) edge list built once |
| Stat computation | O(6.46M Ã— 5) R-level `lapply` calls | O(5) vectorized `data.table` group-bys over ~38.4M edge-year rows |
| Estimated time | ~86 hours | ~5â€“15 minutes |

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.table(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build static spatial edge list ONCE from nb object
#
#   rook_neighbors_unique : an nb object (list of integer vectors)
#   id_order              : vector mapping position â†’ cell id
#
#   Output: edge_dt with columns (id, neighbor_id)
#           ~1,373,394 rows (directed rook edges)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains integer indices into id_order
  # for the neighbors of cell id_order[i].
  # spdep::nb objects use 0L to denote "no neighbors" for islands.
  
  n <- length(neighbors_nb)
  
  # Pre-allocate by computing total number of edges
  edge_counts <- vapply(neighbors_nb, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  total_edges <- sum(edge_counts)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
    k <- length(nb_idx)
    from_id[pos:(pos + k - 1L)] <- id_order[i]
    to_id[pos:(pos + k - 1L)]   <- id_order[nb_idx]
    pos <- pos + k
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

cat(sprintf("Edge list built: %d directed edges\n", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each neighbor source variable, compute neighbor
#          max, min, mean via keyed join + grouped aggregation,
#          then join back to cell_data.
#
#   This replaces build_neighbor_lookup AND
#   compute_neighbor_stats AND the outer for-loop.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))
  
  # --- 2a. Extract the (id, year, value) subset for the neighbor lookup side
  #         We rename 'id' to 'neighbor_id' so we can join on the neighbor's id.
  val_dt <- cell_data[, .(neighbor_id = id, year, nbr_val = get(var_name))]
  setkey(val_dt, neighbor_id, year)
  
  # --- 2b. Expand edges Ã— years: join neighbor attribute onto edge list
  #         For every (id, neighbor_id) edge and every year, get the
  #         neighbor's value of var_name.
  #
  #         We do this by joining edge_dt with val_dt on (neighbor_id, year).
  #         But edge_dt has no year columnâ€”we need the Cartesian product
  #         edge Ã— year. Instead of materializing that (~38.4M rows),
  #         we join cell_data's (id, year) with edge_dt to get
  #         (id, year, neighbor_id), then join val_dt to get nbr_val.
  
  # Get unique (id, year) pairs from cell_data
  id_year <- cell_data[, .(id, year)]
  
  # Join: for each (id, year), attach all neighbor_ids
  # This produces the full (id, year, neighbor_id) table
  edges_by_year <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # edges_by_year now has columns: id, neighbor_id, year
  
  # Join: attach the neighbor's variable value
  edges_by_year <- val_dt[edges_by_year, on = .(neighbor_id, year), nomatch = NA]
  # edges_by_year now has columns: neighbor_id, year, nbr_val, id
  
  # --- 2c. Compute grouped statistics
  stats_dt <- edges_by_year[
    !is.na(nbr_val),
    .(
      nbr_max  = max(nbr_val),
      nbr_min  = min(nbr_val),
      nbr_mean = mean(nbr_val)
    ),
    by = .(id, year)
  ]
  
  # --- 2d. Rename columns to match original naming convention
  #         Original code used: {var_name}_neighbor_max, etc.
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  setnames(stats_dt, c("nbr_max", "nbr_min", "nbr_mean"),
           c(max_col, min_col, mean_col))
  
  setkey(stats_dt, id, year)
  
  # --- 2e. Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # --- 2f. Join stats back to cell_data
  cell_data <- stats_dt[cell_data, on = .(id, year)]
  
  # Clean up to free RAM
  rm(val_dt, id_year, edges_by_year, stats_dt)
  gc()
  
  cat(sprintf("  Done: %s\n", var_name))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the existing trained Random Forest
#
#   The trained model object is assumed to be in memory (e.g.,
#   `rf_model`). No retraining occurs. The numerical estimand
#   is preserved because the neighbor statistics are computed
#   with identical logic (max, min, mean of non-NA neighbor
#   values, NA where no valid neighbors exist).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Ensure cell_data is a data.frame if the RF model expects one
cell_data_df <- as.data.frame(cell_data)

# Predict (using whatever predict method matches the trained model)
cell_data_df$prediction <- predict(rf_model, newdata = cell_data_df)

cat("Prediction complete.\n")
```

---

## Memory Considerations (16 GB Laptop)

The largest intermediate object is `edges_by_year` at ~38.4M rows Ã— 4 columns â‰ˆ **1.2 GB** per variable iteration. With `cell_data` itself at ~6.46M Ã— 110 columns â‰ˆ **5.7 GB**, peak usage is roughly **8â€“9 GB**, well within 16 GB. The `rm()` + `gc()` calls after each variable keep memory bounded.

## Numerical Equivalence

The aggregation logic is identical to the original:
- `max(neighbor_vals)`, `min(neighbor_vals)`, `mean(neighbor_vals)` over non-NA neighbor values.
- Rows with no valid neighbors receive `NA` (via the left join in step 2f: cells absent from `stats_dt` get `NA` for all three columns).

The trained Random Forest model is used as-is with `predict()`â€”no retraining occurs.

## Expected Runtime

| Phase | Estimated Time |
|---|---|
| Edge list construction | ~2 seconds |
| Per-variable join + aggregation (Ã—5) | ~1â€“2 minutes each |
| Total neighbor feature engineering | **~5â€“10 minutes** |
| RF prediction (~6.46M rows, 110 features) | ~5â€“30 minutes (model-dependent) |
| **Total** | **~10â€“40 minutes** |

This represents a **~130â€“500Ã— speedup** over the original 86+ hour estimate.
END_RESPONSE>>>
