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
run_id: OpenAI_F02_T10_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-heavy because:  
1. `build_neighbor_lookup` materializes a 6.46M-length list of integer vectors using **`lapply`**, repeatedly creating strings and performing environment lookups â†’ very high overhead.  
2. `compute_neighbor_stats` loops through the same 6.46M elements across multiple variables using **`lapply`** and `do.call(rbind)` â†’ repeated traversals of large data.  
3. Essentially, you have a dense 6.46M Ã— k loops happening **in R**, which is single-threaded and not vectorized.  

**Optimization Strategy**  
- **Avoid repeated neighbor lookups per variable**: Create a single long table of `(focal_idx, neighbor_idx)` using the precomputed neighbor relationships and a join keyed by `(id, year)` â†’ eliminates repeated string pasting/lookup.  
- Convert the pipeline to a **data.table** join-based aggregation, which is memory-efficient and orders of magnitude faster in R.  
- Use **grouped aggregation** (max, min, mean) over neighbors with one pass.  
- **Reuse results for all variables** by reshaping to long and performing grouped stats in one step.  
- Avoid building large lists of lists: store neighbor expansions in numeric vectors.  

---

### **Proposed Approach**
- Step 1: Expand your nb object into a data.table mapping `id` â†’ `neighbor_id`.
- Step 2: Cross join this map with years â†’ `(id, year, neighbor_id, year)`.
- Step 3: Join to `cell_data` to get neighbor values; then `data.table` `by=.(id, year)` summarization for all variables.
- Step 4: Merge summaries back to `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Ensure keys
# cell_data: columns: id, year, ntl, ec, pop_density, def, usd_est_n2, etc.
setkey(cell_data, id, year)

# Step 1: Expand rook_neighbors_unique to pairs
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Step 2: Create panel for years by repeating neighbor pairs for each year
years <- sort(unique(cell_data$year))
neighbor_expanded <- neighbor_pairs[, .(year = years), by = .(id, neighbor_id)]
setkey(neighbor_expanded, neighbor_id, year)

# Step 3: Join neighbor values for multiple variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_expanded[cell_data, (paste0("n_", vars)) := mget(vars), on = .(neighbor_id = id, year)]

# Step 4: Compute max, min, mean for each var by id-year
agg_exprs <- lapply(vars, function(v) {
  list(
    as.name(paste0(v, "_nb_max")) = substitute(max(x, na.rm = TRUE), list(x = as.name(paste0("n_", v)))),
    as.name(paste0(v, "_nb_min")) = substitute(min(x, na.rm = TRUE), list(x = as.name(paste0("n_", v)))),
    as.name(paste0(v, "_nb_mean")) = substitute(mean(x, na.rm = TRUE), list(x = as.name(paste0("n_", v))))
  )
})
agg_exprs <- do.call(c, agg_exprs)

neighbor_stats <- neighbor_expanded[, eval(agg_exprs), by = .(id, year)]

# Step 5: Merge back to cell_data
cell_data <- neighbor_stats[cell_data, on = .(id, year)]

```

---

### **Why This Works**
- Eliminates expensive per-row `lapply`.
- Performs all neighbor feature computations in **one grouped aggregation**.
- Uses `data.table` joins to leverage fast in-memory keyed lookups.
- Scales linearly in the number of neighbor relationships, not in `nrow(cell_data) Ã— neighbors`.

**Expected Gain**:  
- Memory: ~ tens of millions rows in `neighbor_expanded` (id-year-neighbor triplets) but processed in chunks by `data.table`.
- Runtime: From 86+ hours down to **minutes** (typical speed-up of 1000Ã— compared to unoptimized R loops on ~6M rows).  

**Key Invariants Preserved**:  
- Original numeric estimand unchanged.
- Trained Random Forest model remains untouched.
- Neighbor stats computed for exact same relationships.
END_RESPONSE>>>
