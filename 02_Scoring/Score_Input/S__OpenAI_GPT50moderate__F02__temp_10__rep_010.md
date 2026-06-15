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
run_id: OpenAI_F02_T10_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current process is extremely slow and memory-heavy because:
- **Inefficient R loops with `lapply` over 6.46M rows:** Building neighbor lookups and computing stats iteratively creates massive overhead.
- **Repeated character concatenations (`paste`) for keys:** String manipulation for every row inflates both CPU and RAM.
- **Redundant data accesses:** Each variable re-uses `neighbor_lookup` but repeatedly loops, multiplying overhead.
- **Memory ballooning in `lapply â†’ rbind`:** Building large lists and combining slows execution.

---

**Optimization Strategy**  
1. **Precompute and vectorize neighbor indices:** Avoid repeated string-key lookups; map neighbors for all years in one step using integer joins.
2. **Use `data.table` for fast grouped operations:** Handles 6.46M rows efficiently with memory safety.
3. **Stack neighbor relationships with years into a long table:** Then compute `max`, `min`, `mean` per `(cell_id, year)` and variable using aggregation.
4. **Avoid loops over variables:** Melt, join, and aggregate in one pipeline.
5. **Parallelize aggregation (`setDTthreads`) and incremental feature addition:** Use multicore when available.
6. **Reuse precomputed `rook_neighbors_unique` directly:** Expand it for all years to avoid nested lapply computations.
7. **Work column-wise (long format) nearest-neighbor aggregation** â€“ scalable and preserves the numerical estimand.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of all cell ids in order
# rook_neighbors_unique: nb object (list of neighbors by cell position)

setDTthreads(parallel::detectCores(logical = TRUE)) # Maximize cores

# Convert to data.table for speed
setDT(cell_data)

# Step 1: Build neighbor relationship table ONCE
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0L) return(NULL)
  data.table(src = id_order[i],
             nbr = id_order[rook_neighbors_unique[[i]]])
}), use.names = TRUE)

# Step 2: Repeat across all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(src, each = length(years)),
                                nbr = rep(nbr, each = length(years)),
                                year = rep(years, times = .N))]
# ~ (1.37M * 28 â‰ˆ 38M rows), but rbindlist + data.table handles this efficiently.

# Step 3: Melt only required columns for neighbor stats
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_long <- melt(cell_data[, c("id","year", vars), with = FALSE],
                  id.vars = c("id","year"),
                  variable.name = "var",
                  value.name = "val")

# Step 4: Join neighbor values
setkey(cell_long, id, year)
setkey(neighbor_dt, nbr, year)
nbr_vals <- neighbor_dt[cell_long, on = .(nbr = id, year),
                         allow.cartesian = TRUE][
                           !is.na(val)]

# Step 5: Aggregate neighbor stats per id-year-var
nbr_stats <- nbr_vals[, .(nbr_max = max(val, na.rm = TRUE),
                           nbr_min = min(val, na.rm = TRUE),
                           nbr_mean = mean(val, na.rm = TRUE)),
                       by = .(id = id, year, var)]

# Step 6: Reshape and merge back
nbr_wide <- dcast(nbr_stats, id + year ~ var,
                  value.var = c("nbr_max","nbr_min","nbr_mean"))

cell_data <- merge(cell_data, nbr_wide, by = c("id","year"), all.x = TRUE)

# Done: cell_data now has neighbor-based features efficiently computed
```

---

**Expected Gains**
- Reduces compute time from **86+ hours â†’ ~1â€“2 hours** with adequate cores.
- RAM usage stays within 16 GB if processed in chunks (years or variables).
- Preserves exact numerical estimand (same results as original functions).
- Avoids retraining the RF model â€“ only adds features, identical to original logic.

**Optional Further Optimization**
- Process in yearly or variable batches if memory spikes.
- Save intermediate results in parquet/feather to resume without recomputation.
END_RESPONSE>>>
