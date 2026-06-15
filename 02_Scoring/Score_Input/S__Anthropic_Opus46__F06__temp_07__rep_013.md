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
run_id: Anthropic_F06_T07_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine â€” dictionary lookup).
2. Constructs `paste(neighbor_id, year)` key strings for each neighbor.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) into a 6.46M-element named character vector.

Named vector lookup in R is **hash-based but slow at scale** because `paste` allocates millions of small character vectors and the named lookup has per-call overhead. With ~6.46M rows Ã— ~4 neighbors each â‰ˆ 25.8M string constructions and lookups, this alone can take many hours.

### Bottleneck B: `compute_neighbor_stats` â€” `lapply` over 6.46M rows, repeated 5Ã—

Each call extracts neighbor values via index subsetting and computes `max/min/mean`. The `lapply` loop is called 5 times (once per variable). The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive.

### Why raster focal/kernel operations don't directly apply

Focal operations assume a regular complete grid with a fixed rectangular kernel. The panel has:
- Irregular boundaries (not all cells present every year, NA handling).
- Rook contiguity neighbors that may not map to a simple 3Ã—3 kernel if the grid has missing cells or irregular shape.

However, the **analogy is useful**: focal operations are fast because they operate column-wise on matrices. We can replicate this by converting the neighbor structure into a **sparse adjacency matrix** and using **sparse matrixâ€“dense matrix multiplication** to compute neighbor sums and counts, then derive max/min/mean.

**Caveat for max and min**: Matrix multiplication gives sums, not max/min. For max/min we need a different approach. We can use a **data.table join** strategy that is far faster than the per-row `lapply`.

---

## 2. Optimization Strategy

### Step 1: Replace `build_neighbor_lookup` entirely
Instead of building a per-row lookup list of 6.46M elements, build an **edge table** (data.table) of `(row_i, row_j)` pairs â€” i.e., for each cell-year row `i`, list all row indices `j` that are its rook neighbors in the same year. This edge table has ~25.8M rows (6.46M rows Ã— ~4 neighbors), which is very manageable.

Construction: use `data.table` keyed joins â€” merge the spatial neighbor pairs with year to get row-index pairs. This replaces millions of `paste` + named-vector lookups with a single vectorized join.

### Step 2: Replace `compute_neighbor_stats` with vectorized group-by
Using the edge table, for each variable:
- Join the variable's values onto the edge table by `row_j`.
- Group by `row_i` and compute `max`, `min`, `mean` in one `data.table` aggregation.

This replaces 6.46M R-level loop iterations with a single vectorized `data.table` grouped aggregation over ~25.8M rows â€” typically seconds, not hours.

### Step 3: Compute all 5 variables in one pass (or 5 fast passes)

### Expected speedup
- From ~86+ hours â†’ **~2â€“10 minutes** on a 16 GB laptop.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with a row-index column
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.table(cell_data)
cell_data[, row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the edge table (replaces build_neighbor_lookup)
#
# Inputs:
#   id_order             â€” vector of spatial cell IDs in the order used by
#                          the nb object (length = 344,208)
#   rook_neighbors_unique â€” spdep::nb object (list of length 344,208,
#                           each element = integer vector of neighbor
#                           positions in id_order)
#   cell_data            â€” data.table with columns: id, year, row_idx,
#                          plus all predictor columns
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(cell_data, id_order, neighbors) {
  # --- 1a. Build spatial edge list (cell-id to cell-id) ---------------
  n_cells <- length(id_order)
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  
  for (k in seq_len(n_cells)) {
    nb_idx <- neighbors[[k]]
    if (length(nb_idx) == 0L) next
    from_list[[k]] <- rep(id_order[k], length(nb_idx))
    to_list[[k]]   <- id_order[nb_idx]
  }
  
  spatial_edges <- data.table(
    id_from = unlist(from_list, use.names = FALSE),
    id_to   = unlist(to_list,   use.names = FALSE)
  )
  # spatial_edges has ~1,373,394 rows (directed rook pairs)
  
  # --- 1b. Join with cell_data to get (row_i, row_j) for same year ---
  # Map: for each cell-year row, find its row_idx
  id_year_idx <- cell_data[, .(id, year, row_idx)]
  
  # Join "from" side: get row_idx of the focal cell and its year
  setkey(id_year_idx, id)
  # We need to cross spatial_edges with years.
  # But a full cross (1.37M edges Ã— 28 years) = 38.4M rows â€” still fine.
  #
  # More efficient: join via cell_data directly.
  
  # Create a lookup: for each (id, year) â†’ row_idx
  setkey(id_year_idx, id, year)
  
  # For each spatial edge (id_from, id_to), for each year that id_from
  # appears in, find the row_idx of id_from and id_to in that year.
  
  # Get all (id_from, year, row_idx_from)
  from_dt <- id_year_idx[, .(id_from = id, year, row_i = row_idx)]
  
  # Merge with spatial edges to get id_to for each (id_from, year)
  setkey(spatial_edges, id_from)
  setkey(from_dt, id_from)
  
  # This is the key join: for each (id_from, year) expand by all
  # neighbors of id_from
  edge_year <- spatial_edges[from_dt, on = "id_from",
                              allow.cartesian = TRUE,
                              nomatch = NULL]
  # edge_year has columns: id_from, id_to, year, row_i
  # ~1.37M Ã— 28 â‰ˆ 38.5M rows (but many id_from may not appear all years;
  # actual count â‰ˆ 25.8M based on problem statement)
  
  # Now find row_idx for the neighbor (id_to, year)
  setnames(id_year_idx, c("id_to", "year", "row_j"))
  setkey(id_year_idx, id_to, year)
  setkey(edge_year, id_to, year)
  
  edge_year <- id_year_idx[edge_year, on = c("id_to", "year"),
                            nomatch = NA]
  # Keep only edges where the neighbor actually exists in that year
  edge_year <- edge_year[!is.na(row_j)]
  
  # Return minimal columns
  edge_year[, .(row_i, row_j)]
}

cat("Building edge table...\n")
system.time({
  edge_dt <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~20â€“60 seconds, ~25.8M rows, two integer columns â‰ˆ 200 MB

setkey(edge_dt, row_i)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor stats for all 5 variables
#         (replaces compute_neighbor_stats + outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    
    cat("  Processing:", var_name, "\n")
    
    # Extract the variable values indexed by row_idx
    vals <- cell_data[[var_name]]
    
    # Attach neighbor values to the edge table
    edge_dt[, nval := vals[row_j]]
    
    # Remove edges where the neighbor value is NA
    edge_valid <- edge_dt[!is.na(nval)]
    
    # Grouped aggregation: max, min, mean by focal row
    agg <- edge_valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = row_i]
    
    # Create full-length result columns (NA for rows with no valid neighbors)
    n <- nrow(cell_data)
    col_max  <- rep(NA_real_, n)
    col_min  <- rep(NA_real_, n)
    col_mean <- rep(NA_real_, n)
    
    col_max[agg$row_i]  <- agg$nb_max
    col_min[agg$row_i]  <- agg$nb_min
    col_mean[agg$row_i] <- agg$nb_mean
    
    # Add to cell_data with the same column naming convention
    # (adjust names to match whatever compute_and_add_neighbor_features used)
    set(cell_data, j = paste0(var_name, "_neighbor_max"),  value = col_max)
    set(cell_data, j = paste0(var_name, "_neighbor_min"),  value = col_min)
    set(cell_data, j = paste0(var_name, "_neighbor_mean"), value = col_mean)
  }
  # Clean up temporary column
  edge_dt[, nval := NULL]
})
# Expected: ~1â€“3 minutes total for all 5 variables

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Remove helper column and proceed to prediction
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, row_idx := NULL]

# The trained Random Forest model is unchanged.
# Predict as before (example):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor definition** | We use the identical `rook_neighbors_unique` nb object and `id_order`. The spatial edge list is a direct transcription of the same adjacency structure. |
| **Same year matching** | The `(id_to, year)` join ensures neighbors are matched within the same year, exactly as the original `paste(id, year)` key did. |
| **Same NA handling** | `!is.na(nval)` mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Rows with zero valid neighbors get `NA` for all three stats, matching `return(c(NA, NA, NA))`. |
| **Same aggregation functions** | `max`, `min`, `mean` are identical base-R functions. No approximation. |
| **Random Forest untouched** | The pre-trained model is loaded and used for `predict()` only. No retraining. |

**Result**: Bit-identical outputs (up to floating-point associativity of `mean` in different element orderings, which is negligible and within machine epsilon). The optimization is purely computational â€” replacing R-level loops and string operations with vectorized `data.table` joins and grouped aggregations.

---

## 5. Summary of Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~40â€“60 hours | ~30â€“60 seconds |
| Neighbor stats (5 vars Ã— 6.46M rows) | ~26â€“30 hours | ~1â€“3 minutes |
| **Total** | **86+ hours** | **~2â€“5 minutes** |
| Peak RAM | ~6â€“10 GB (list overhead) | ~3â€“4 GB (edge table + data.table) |
END_RESPONSE>>>
