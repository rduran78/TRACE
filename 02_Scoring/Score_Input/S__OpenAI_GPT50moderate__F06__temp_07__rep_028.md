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
run_id: OpenAI_F06_T07_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current approach is **pure R loops with `lapply` over ~6.46M rows**, which is extremely inefficient for large datasets.  
- Each row computes neighbor stats by repeatedly indexing into `vals`, causing **millions of small vector allocations**.  
- Memory overhead and interpretation layer in R make this **O(N Ã— avg_neighbors)** approach (â‰ˆ6.46M Ã— ~4â€“6 neighbors) prohibitively slow.  

**Optimization Strategy**  
- Avoid repeated lookups and R-level loops by **vectorizing using a long-format edge list** and aggregating with `data.table`.  
- Convert neighbor relationships into a **cell-year edge table**, join with variable values, then compute `max`, `min`, `mean` using **fast group-by**.  
- This approach is analogous to **focal raster operations** but implemented as **grouped aggregation**, preserving the exact estimand.  
- Operates in **linear time** relative to edges, avoids millions of function calls, and runs in **minutes instead of 86+ hours** on 16 GB RAM.  

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep::nb), same order as id_order
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)

# Build lookup for neighbors as edge list (static, no year yet)
edges <- data.table(
  src = rep(id_order, lengths(rook_neighbors_unique)),
  dst = unlist(rook_neighbors_unique)
)

# Expand to cell-year relationships
years <- unique(cell_data$year)
edges <- edges[, .(year = years), by = .(src, dst)]

# Join with source variable values
# Prepare a fast key on cell_data
setkey(cell_data, id, year)

compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  edges[, value := cell_data[.(dst, year), get(var_name)]]
  
  # Aggregate by source cell-year
  agg <- edges[!is.na(value), .(
    max_val = max(value),
    min_val = min(value),
    mean_val = mean(value)
  ), by = .(src, year)]
  
  # Merge back to cell_data
  setnames(agg, c("src", "year"), c("id", "year"))
  new_cols <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  setnames(agg, c("max_val", "min_val", "mean_val"), new_cols)
  
  cell_data[agg, on = .(id, year), (new_cols) := mget(new_cols)]
}

# Apply for each neighbor source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) compute_neighbor_stats_dt(v)
```

---

**Why This Works**  
- `edges` table size â‰ˆ `1.37M neighbors Ã— 28 years` â‰ˆ 38M rows, which fits in memory on 16 GB with `data.table`.  
- Joins and aggregations are **highly optimized in C** via `data.table`.  
- Eliminates per-row loops and repeated indexing, preserving **exact same logic** (max, min, mean over rook neighbors in same year).  
- Fully compatible with your existing Random Forest pipeline (model remains unchanged).  

**Performance Gain**  
- From **86+ hours to a few minutes** on a modern laptop for ~6.46M rows and ~5 features.  
- Memory footprint is manageable since only relevant columns are materialized during joins.  

This preserves the trained model and the numerical estimand exactly while making the computation tractable.
END_RESPONSE>>>
