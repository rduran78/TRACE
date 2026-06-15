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
run_id: OpenAI_F06_T07_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach repeatedly performs expensive list-based lookups and per-row `lapply` loops over ~6.46 million rows. For each variable, it recomputes neighbor summaries in pure R lists, causing severe overhead. The key bottlenecks are:  

- `build_neighbor_lookup` returns a list of integer vectors for every row, consuming huge memory and requiring repeated indexing.  
- `compute_neighbor_stats` iterates over all rows for each variable, repeatedly scanning lists, which is computationally prohibitive.  
- No vectorization or efficient matrix operations; everything is done with R loops.  

**Optimization Strategy**  
- **Precompute a sparse adjacency structure** (e.g., via `Matrix::dgCMatrix`) for neighbors to make aggregation vectorized.  
- **Use matrix operations or `rowsum` grouping** rather than per-row loops.  
- **Process by year** to keep memory within 16 GB RAM and exploit temporal independence.  
- Use `data.table` for fast joins and updates.  
- Avoid building massive lookup lists; instead, use neighbor pairs in a long format and aggregate.  

**Optimized Approach**  
1. Convert neighbor relationships into a long edge list `(i, j)` where `i` is the focal cell, `j` is the neighbor.  
2. Join neighbor pairs with variable values by year.  
3. Aggregate by focal cell-year using `max`, `min`, `mean`.  
4. Merge aggregated results back.  
5. Do this for each variable in one pass or variable-by-variable efficiently.  

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors with neighbor indices corresponding to id_order
# id_order: vector of ids in the same order used in rook_neighbors_unique

# Convert neighbors to edge list
edges <- data.table(
  focal = rep(id_order, lengths(rook_neighbors_unique)),
  neigh = unlist(rook_neighbors_unique)
)

# Create a long key for joining with year
setkey(cell_data, id, year)

# Variables to process
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Process by year to limit memory
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  # Subset data for this year
  dt_year <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  
  # Join edges with neighbor values
  setkey(dt_year, id)
  edge_dt <- edges[dt_year, on = .(neigh = id)]
  
  # edge_dt now has: focal, neigh, year, var values
  # Remove rows with NA in focal or required columns
  edge_dt <- edge_dt[!is.na(focal)]
  
  # Compute stats for each variable
  for (v in neighbor_vars) {
    stats <- edge_dt[!is.na(get(v)), .(
      max_val = max(get(v), na.rm = TRUE),
      min_val = min(get(v), na.rm = TRUE),
      mean_val = mean(get(v), na.rm = TRUE)
    ), by = .(focal)]
    
    # Rename columns
    setnames(stats, c("focal", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
    
    # Merge back into main cell_data
    cell_data[year == yr, (names(stats)[-1]) := stats[.SD, on = .(focal = id), .SDcols = names(stats)[-1]]]
  }
}

# At this point, cell_data has neighbor-based features added for all years and variables
```

---

**Why Itâ€™s Faster:**  
- Avoids per-row `lapply`; uses vectorized grouping in `data.table`.  
- Processes data year by year, keeping joins and aggregations manageable in memory.  
- Handles all neighbors via an edge list instead of building an enormous lookup list.  

**Expected Performance:**  
- From 86+ hours down to a few hours on a 16 GB machine, depending on I/O.  
- Scales linearly with years and neighbor edges.  

**Preserves:**  
- Original numerical estimand (same max, min, mean logic).  
- Pre-trained Random Forest model remains untouched.  
- Panel structure and cell-year association remain intact.
END_RESPONSE>>>
