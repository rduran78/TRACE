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
run_id: Anthropic_F09_T03_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

1. **`build_neighbor_lookup` operates at the cell-year level (6.46M rows), not the cell level (344K cells).** The neighbor topology is *time-invariant*â€”cell ð‘–'s rook neighbors are the same in every year. Yet the function builds a lookup list of length 6.46 million, performing string-paste key construction and hash-table lookups for every single cell-year row. This is ~28Ã— more work than necessary.

2. **`compute_neighbor_stats` uses an R-level `lapply` over 6.46M elements**, each calling `max`, `min`, `mean` on small vectors. The per-element R interpreter overhead (function dispatch, NA checks, vector allocation, `c()` concatenation) dominates. With 5 source variables, that's ~32.3 million R-level iterations, each allocating temporary vectors.

3. **The final `do.call(rbind, result)` on a 6.46M-element list** is itself an expensive operation that creates millions of intermediate row vectors before binding.

**In short:** the code re-discovers the same spatial topology 6.46M times and then loops over 6.46M rows in pure R for each variable. Both steps should be vectorized.

---

## Optimization Strategy

### Core Idea: Separate Topology from Attributes

Build a **cell-level adjacency table once** (344K cells Ã— their neighbors â‰ˆ 1.37M directed edges), then **join yearly attributes** onto that edge table and compute grouped summaries using vectorized `data.table` operations. This eliminates all R-level row loops.

### Steps

| Step | What | Complexity |
|------|------|------------|
| 1 | Convert `rook_neighbors_unique` (spdep nb object) into a two-column edge table: `(cell_id, neighbor_id)`. ~1.37M rows. **Done once.** | O(E) |
| 2 | For each year, join cell attributes onto the edge table by `(neighbor_id, year)` to get each neighbor's variable values. | O(E) per year via keyed join |
| 3 | Group by `(cell_id, year)` and compute `max`, `min`, `mean` in one vectorized pass per variable. | O(E) per variable |
| 4 | Join the resulting neighbor-stat columns back onto `cell_data`. | O(N) |

**Expected speedup:** The edge table has ~1.37M rows Ã— 28 years = ~38.5M edge-year rows, but all operations are vectorized C-level `data.table` group-by aggregationsâ€”no R-level loops. Estimated wall time: **2â€“10 minutes** on a 16 GB laptop, down from 86+ hours.

**Memory:** The edge-year table for one variable is ~38.5M rows Ã— 3 columns â‰ˆ ~900 MB. We process one variable at a time and discard intermediates, staying well within 16 GB.

**Preserves:** The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per variable per cell-year) is identical.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert cell_data to data.table if not already
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a time-invariant cell-level edge table ONCE
#
#   rook_neighbors_unique : spdep nb object (list of integer vectors)
#   id_order              : vector mapping list index -> cell id
#
#   Result: edges_dt with columns (cell_id, neighbor_id), ~1.37M rows
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for cell i's neighbors
  n_cells <- length(id_order)
  
  # Pre-compute total number of edges for pre-allocation
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    nb_len <- length(nb_idx)
    if (nb_len == 0L) next
    # Remove the "no-neighbor" sentinel (0) that spdep uses
    nb_idx <- nb_idx[nb_idx != 0L]
    nb_len <- length(nb_idx)
    if (nb_len == 0L) next
    idx_range <- pos:(pos + nb_len - 1L)
    from_id[idx_range] <- id_order[i]
    to_id[idx_range]   <- id_order[nb_idx]
    pos <- pos + nb_len
  }
  
  # Trim if any zero-neighbor cells caused over-allocation
  if (pos - 1L < n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

cat("Building cell-level edge table...\n")
edges_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed edges\n", format(nrow(edges_dt), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each neighbor source variable, join yearly attributes
#         onto the edge table and compute grouped neighbor stats
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast joins
setkey(cell_data, id, year)

# We will also need a key on the edge table for the cross-join with years
# Strategy: cross-join edges Ã— years, then keyed-join neighbor attributes

# Get unique years
all_years <- sort(unique(cell_data$year))

cat("Computing neighbor features for each variable...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # Extract only the columns we need for the join: id, year, <var_name>
  # This keeps memory low
  attr_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)
  
  # Cross-join edge table with all years to get edge-year table
  # edges_dt has (cell_id, neighbor_id); we expand by year
  # To avoid materializing the full cross-join in memory at once,
  # we join directly:
  #   For each edge (cell_id, neighbor_id), for each year,
  #   look up the neighbor's attribute value, then aggregate.
  #
  # Efficient approach: 
  #   1. Create edge-year table by CJ on edges and years
  #   2. Keyed join to get neighbor values
  #   3. Aggregate by (cell_id, year)
  
  # 1. Expand edges by year (~38.5M rows for 1.37M edges Ã— 28 years)
  edge_year <- edges_dt[, .(year = all_years), by = .(cell_id, neighbor_id)]
  
  # 2. Join neighbor attribute values
  #    Join key: neighbor_id == id, year == year
  setkey(edge_year, neighbor_id, year)
  edge_year[attr_dt, neighbor_val := i.val, on = .(neighbor_id = id, year = year)]
  
  # 3. Aggregate: compute max, min, mean of neighbor_val by (cell_id, year)
  #    Exclude NAs to match original behavior
  stats <- edge_year[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(cell_id, year)
  ]
  
  # Name the output columns to match the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # 4. Join stats back onto cell_data
  #    Remove old columns if they exist (in case of re-run)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  setkey(stats, cell_id, year)
  cell_data[stats, (c(max_col, min_col, mean_col)) := mget(c(max_col, min_col, mean_col)),
            on = .(id = cell_id, year = year)]
  
  # Free memory
  rm(edge_year, stats, attr_dt)
  gc()
  
  cat(sprintf("    Done: added %s, %s, %s\n", max_col, min_col, mean_col))
}

cat("All neighbor features computed.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Predict with the existing trained Random Forest model
#         (model object unchanged â€” no retraining)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example (adjust to your actual model object and predict call):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Key Changes

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup granularity | 6.46M cell-year rows | 344K cells (1.37M edges), built once |
| Core loop | `lapply` over 6.46M Ã— 5 vars in R | Vectorized `data.table` grouped aggregation |
| Key construction | `paste(id, year)` string hashing 6.46M times | Integer-keyed joins on `(id, year)` |
| Result binding | `do.call(rbind, 6.46M-element list)` | In-place `data.table` column assignment |
| Estimated runtime | ~86+ hours | **~2â€“10 minutes** |
| Trained RF model | Preserved âœ“ | Preserved âœ“ |
| Numerical output | Original estimand | Identical âœ“ |

The fundamental insight is that **spatial topology is time-invariant**: build the adjacency structure once at the cell level, then let `data.table`'s vectorized C backend handle the year-wise attribute joins and grouped aggregations.
END_RESPONSE>>>
