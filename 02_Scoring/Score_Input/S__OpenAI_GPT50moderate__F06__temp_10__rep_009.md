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
run_id: OpenAI_F06_T10_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` structure applied over **6.46 million rows Ã— 5 variables**. For each row, the `compute_neighbor_stats` function iterates through an individual lookup list and recomputes max, min, and mean using R loops, causing heavy overhead. This design scales poorly because:

- It repeatedly processes each cell-year rather than computing statistics over the adjacency structure in bulk.
- Memory usage is inefficient due to millions of small list objects.
- Pure R loops cannot effectively utilize vectorization or parallelization here.

The operation is conceptually a **focal statistics computation on a panel grid**, which naturally maps to a **graph-based aggregation** problem. Using **vectorized joins or matrix-based reductions** can cut runtime by orders of magnitude while preserving original results.

---

**Optimization Strategy**  
1. **Vectorize neighbor aggregation**:
   - Expand neighbor relationships into a long â€œedgesâ€ table: `(source_row, neighbor_row)`.
   - Use **data.table** for fast joins and grouped aggregations to compute `max`, `min`, `mean`.
2. Compute all five variables in one pass rather than iterating five times.
3. Reattach aggregated stats to the original dataset.
4. (Optional) Use **parallelization** with `data.table` or `future` for scalability.

This avoids per-row list operations and leverages highly optimized C-backed group aggregations.

---

**Working R Code**

```r
library(data.table)

# Convert your data.frame to data.table for efficiency
setDT(cell_data)

# Step 1: Build neighbor lookup into a long edge list
id_to_ref <- setNames(seq_along(id_order), id_order)
idx_lookup <- setNames(seq_len(nrow(cell_data)),
                       paste(cell_data$id, cell_data$year, sep = "_"))

expand_neighbors <- function(data, neighbors, id_order) {
  # For each row in data, find its neighbors in the same year
  src_ids <- seq_len(nrow(data))
  # Get reference for cell -> rook neighbors
  ref_idx <- id_to_ref[as.character(data$id[src_ids])]
  
  edge_list <- vector("list", length(src_ids))
  
  for (i in src_ids) {
    neigh_cells <- id_order[neighbors[[ref_idx[i]]]]
    if (length(neigh_cells) > 0) {
      neigh_keys <- paste(neigh_cells, data$year[i], sep = "_")
      neigh_rows <- idx_lookup[neigh_keys]
      neigh_rows <- neigh_rows[!is.na(neigh_rows)]
      if (length(neigh_rows) > 0) {
        edge_list[[i]] <- data.table(src = i, nbr = neigh_rows)
      }
    }
  }
  rbindlist(edge_list)
}

# Build edge table
edges <- expand_neighbors(cell_data, rook_neighbors_unique, id_order)
# edges: src (row in cell_data), nbr (row in cell_data)

# Step 2: Aggregate all neighbor variables efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt neighbor values for join
agg_list <- lapply(neighbor_source_vars, function(var) {
  dt <- cell_data[, .(nbr_row = .I, val = get(var))]
  setnames(dt, "val", var)
  dt
})

neighbor_vals <- Reduce(function(x, y) merge(x, y, by = "nbr_row", all = TRUE), agg_list)

# Merge edges with neighbor values
edges <- merge(edges, neighbor_vals, by.x = "nbr", by.y = "nbr_row", all.x = TRUE)

# Step 3: Compute stats by src row
agg_stats <- edges[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = src]

# Replace NaN for groups with no neighbors
for (j in names(agg_stats)) {
  set(agg_stats, which(is.nan(agg_stats[[j]])), j, NA)
}

# Step 4: Merge aggregated features back to cell_data
cell_data <- merge(cell_data, agg_stats, by.x = ".I", by.y = "src", all.x = TRUE)

# Now cell_data has new neighbor-based features
```

---

**Expected Performance Gain:**  
- Original: ~86 hours (pure R looping over millions of rows).
- Optimized: minutes to under an hour on 16 GB RAM laptop using `data.table` and vectorized merges.

**Key Points:**  
- Preserves trained Random Forest model (no retraining).
- Preserves original estimands: max, min, and mean for each neighbor variable.
- Scales well for 6.5M rows and 1.3M edges.
END_RESPONSE>>>
