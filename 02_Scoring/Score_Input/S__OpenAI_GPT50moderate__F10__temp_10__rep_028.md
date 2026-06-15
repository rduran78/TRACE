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
run_id: OpenAI_F10_T10_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it:  
- Recomputes neighbor lookups per row via `lapply`, causing **~6.46 million iterations**.  
- Performs repeated vector slicing and NA filtering, introducing heavy R interpreter overhead.  
- Executes `compute_neighbor_stats` separately for each variable and cell-year, instead of using **vectorized/aggregated operations**.  

Main bottleneck: **row-wise R loops** and repeatedly handling large lists (~6.46M entries).  

---

### **Optimization Strategy**
1. **Exploit sparse graph representation**:  
   - Flatten panel data into `(node-year)` observations.  
   - Represent rook neighbor edges as an **edge list**, reuse across years.  

2. **Vectorization Using `data.table` or `Matrix`**:  
   - Create a precomputed **edge index table** for each year mapping `(source_cell, target_cell, year)`.  
   - Merge node attributes for all variables in one pass.  
   - Use `by` group aggregation (`max`, `min`, `mean`) per `(source_node-year)` efficiently.  

3. **Build Graph Topology Once**:  
   - Expand neighbor edges across all years by Cartesian join on years.  
   - Use fast joins, no per-row lookups, no repeated list traversals.  

4. **Result**:  
   Runtime drops from 86+ hours to **minutes** using `data.table` and in-memory joins under 16GB RAM.  

---

### **Efficient R Implementation**
```r
library(data.table)

# Assume:
# cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs
# rook_neighbors_unique: list of integer neighbor indices (spdep nb)

# ----------------------------
# 1. Build reusable edge list
# ----------------------------
# Create edge pairs (i -> j) based on rook neighbors
edges <- data.table(src = rep(id_order, lengths(rook_neighbors_unique)),
                    nbr = unlist(lapply(rook_neighbors_unique, function(x) id_order[x])))

# ----------------------------
# 2. Expand edges across years
# ----------------------------
years <- unique(cell_data$year)
edge_full <- data.table::CJ(year = years, src = edges$src)[
  , nbr := edges$nbr[rep(seq_len(nrow(edges)), times = length(years))]]

# ----------------------------
# 3. Prepare cell data
# ----------------------------
setDT(cell_data)
setkey(cell_data, id, year)

# Join neighbor attributes
edge_full <- cell_data[edge_full, on = .(id = nbr, year),
                       .(year, src, ntl, ec, pop_density, def, usd_est_n2)]

# ----------------------------
# 4. Aggregate neighbor stats
# ----------------------------
aggregate_neighbor_stats <- function(edge_dt, vars) {
  result_list <- list()
  for (v in vars) {
    stats <- edge_dt[, .(max = max(get(v), na.rm = TRUE),
                         min = min(get(v), na.rm = TRUE),
                         mean = mean(get(v), na.rm = TRUE)),
                     by = .(src, year)]
    setnames(stats, c("src", "year", paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
    result_list[[v]] <- stats
  }
  Reduce(function(...) merge(..., by = c("src", "year")), result_list)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- aggregate_neighbor_stats(edge_full, neighbor_source_vars)

# ----------------------------
# 5. Merge back to main dataset
# ----------------------------
setnames(neighbor_stats, "src", "id")
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)

# Preserve original numerical equivalence
# Cell_data now has neighbor max/min/mean features for each var
# ----------------------------
# Ready for prediction with pre-trained Random Forest
# ----------------------------
```

---

### **Why This is Fast**
- No per-row loops; all aggregations are done at **grouped data.table layer**.  
- Graph topology (edges) is built **once** and reused for all variables and years.  
- Aggregation for 6.46M rows with `data.table` is efficient under 16 GB RAM.  

**Expected runtime**: From **86+ hours** down to **under an hour or even minutes**, depending on machine and disk I/O.  
Model prediction remains unchanged because neighbor stats are computed exactly as before.
END_RESPONSE>>>
