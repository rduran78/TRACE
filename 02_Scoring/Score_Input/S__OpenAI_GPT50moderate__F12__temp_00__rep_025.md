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
run_id: OpenAI_F12_T00_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F12_OpenAI_temp_00_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is **incorrect**. While `rbind` on millions of rows is non-trivial, the deeper bottleneck is the **nested `lapply` over 6.46M rows for each of 5 variables**, performing repeated neighbor lookups and vector operations in pure R. This results in ~32M iterations and heavy R interpreter overhead. The real issue is lack of vectorization and redundant passes through the data.

---

### **Correct Optimization Strategy**
- Precompute neighbor indices once (already done via `neighbor_lookup`).
- Avoid per-row `lapply` in `compute_neighbor_stats()`. Instead, **vectorize aggregation** using `data.table` or `matrix` operations.
- Compute all neighbor-based stats for all variables in a **single pass** over neighbor relationships, not per variable.
- Use efficient joins and group aggregation rather than millions of small R function calls.

---

### **Optimized Approach**
1. Flatten neighbor relationships into a long table: `(cell_id, year, neighbor_id, neighbor_year)`.
2. Join with `data` to get neighbor values for all variables.
3. Aggregate `max`, `min`, `mean` by `(cell_id, year)` for each variable in one grouped operation.
4. Merge results back to `cell_data`.

---

### **Working R Code (data.table solution)**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Add a unique key for each cell-year
dt[, cell_year := paste(id, year, sep = "_")]

# Build neighbor lookup as a long table
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(dt)), dt$cell_year)

# Flatten neighbor relationships
neighbor_list <- lapply(seq_along(id_order), function(i) {
  src_id <- id_order[i]
  neighbors <- rook_neighbors_unique[[i]]
  if (length(neighbors) == 0) return(NULL)
  data.table(
    src = src_id,
    nbr = id_order[neighbors]
  )
})
neighbor_dt <- rbindlist(neighbor_list)

# Expand across years
years <- unique(dt$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(src, nbr)]
neighbor_dt[, `:=`(
  src_key = paste(src, year, sep = "_"),
  nbr_key = paste(nbr, year, sep = "_")
)]

# Map to row indices
neighbor_dt[, src_idx := idx_lookup[src_key]]
neighbor_dt[, nbr_idx := idx_lookup[nbr_key]]
neighbor_dt <- neighbor_dt[!is.na(src_idx) & !is.na(nbr_idx)]

# Join neighbor values for all variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_dt <- neighbor_dt[, .(src_idx, nbr_idx)]
long_dt <- merge(long_dt, dt[, c("row_id" = .I, vars), with = FALSE],
                 by.x = "nbr_idx", by.y = "row_id", all.x = TRUE)

# Melt for aggregation
melted <- melt(long_dt, id.vars = "src_idx", measure.vars = vars,
               variable.name = "var", value.name = "val", na.rm = TRUE)

# Compute stats by src_idx and var
agg <- melted[, .(
  max_val = max(val, na.rm = TRUE),
  min_val = min(val, na.rm = TRUE),
  mean_val = mean(val, na.rm = TRUE)
), by = .(src_idx, var)]

# Reshape wide and merge back
agg_wide <- dcast(agg, src_idx ~ var, value.var = c("max_val", "min_val", "mean_val"))
setnames(agg_wide, "src_idx", "row_id")
dt <- cbind(dt, agg_wide[match(seq_len(nrow(dt)), row_id), -1])

# Result: dt now has neighbor stats for all variables
```

---

### **Why This Is Faster**
- Eliminates 6.46M `lapply` calls Ã— 5 variables.
- Uses `data.table` joins and grouped aggregation in compiled C code.
- Processes all variables in one pass.
- Memory-efficient: avoids repeated intermediate lists.

**Expected runtime improvement:** From 86+ hours to **minutes** on a 16 GB laptop.

**Preserves:**  
- Original numerical estimand.  
- Trained Random Forest model (no retraining).  

**Summary:** The bottleneck is the per-row R loops, not `rbind`. The correct fix is full vectorization and grouped aggregation using `data.table`.
END_RESPONSE>>>
