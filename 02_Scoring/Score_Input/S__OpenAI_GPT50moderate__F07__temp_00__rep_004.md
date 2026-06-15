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
run_id: OpenAI_F07_T00_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F07_OpenAI_temp_00_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, creating millions of small vectors and repeated lookups in R lists.  
- Neighbor lookups are recomputed for every row and every variable, causing redundant work.  
- Pure R loops and list operations are inefficient for this scale.  
- Memory overhead is high due to repeated object creation.  

**Optimization Strategy**  
- Precompute a **flat neighbor index matrix** (or compressed sparse row structure) for all rows once.  
- Use **vectorized operations** or **data.table** joins instead of per-row `lapply`.  
- Compute all neighbor stats in a single pass per variable using fast aggregation.  
- Avoid repeated string concatenation and lookups; map IDs to row indices once.  
- Use `data.table` for efficient grouping and joining.  
- Keep everything in memory-friendly structures (integer indices, numeric vectors).  

---

### **Optimized Approach**
1. Precompute a `data.table` of `(row_id, neighbor_id)` pairs for all cell-year rows.  
2. Join this table with the main data for each variable and compute `max`, `min`, `mean` by `row_id`.  
3. Merge results back into the main dataset.  

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data (data.table), columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs
# rook_neighbors_unique: list of neighbor indices (spdep::nb)
# Precompute row index for each (id, year)
setDT(cell_data)
cell_data[, row_id := .I]

# Map cell_id -> position in id_order
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# Build neighbor pairs for all rows
years <- unique(cell_data$year)
neighbor_pairs_list <- vector("list", length(years))

for (y in seq_along(years)) {
  yr <- years[y]
  # rows for this year
  rows_this_year <- cell_data[year == yr, .(id, row_id)]
  idx_lookup <- setNames(rows_this_year$row_id, as.character(rows_this_year$id))
  
  # For each cell in id_order, get neighbors
  pairs <- lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    src_id <- id_order[i]
    src_row <- idx_lookup[as.character(src_id)]
    if (is.na(src_row)) return(NULL)
    neigh_ids <- id_order[rook_neighbors_unique[[i]]]
    neigh_rows <- idx_lookup[as.character(neigh_ids)]
    neigh_rows <- neigh_rows[!is.na(neigh_rows)]
    if (length(neigh_rows) == 0) return(NULL)
    data.table(row_id = src_row, neighbor_id = neigh_rows)
  })
  neighbor_pairs_list[[y]] <- rbindlist(pairs)
}

neighbor_pairs <- rbindlist(neighbor_pairs_list)
setkey(neighbor_pairs, neighbor_id)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_dt <- function(var_name) {
  tmp <- cell_data[, .(neighbor_id = row_id, value = get(var_name))]
  res <- neighbor_pairs[tmp, on = "neighbor_id"]
  res <- res[!is.na(value), .(
    max_val = max(value),
    min_val = min(value),
    mean_val = mean(value)
  ), by = row_id]
  setnames(res, c("row_id", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  res
}

# Compute for all variables and merge
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_dt(var_name)
  cell_data <- stats_dt[cell_data, on = "row_id"]
}

# Drop helper column
cell_data[, row_id := NULL]
```

---

### **Why This Is Fast**
- `neighbor_pairs` is built once (~6.46M rows Ã— avg neighbors).  
- Joins and aggregations use `data.table`'s optimized C backend.  
- No per-row `lapply` or string concatenation in the main loop.  
- Memory footprint is controlled by using integer indices and merging results efficiently.  

**Expected Performance**:  
- Precomputation: a few minutes.  
- Each variable aggregation: seconds to a few minutes.  
- Entire pipeline: **under 1 hour** on a 16 GB laptop (vs. 86+ hours).  

**Preserves**:  
- Original rook-neighbor relationships.  
- Exact numerical estimand (max, min, mean).  
- Trained Random Forest model remains unchanged.
END_RESPONSE>>>
