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
run_id: OpenAI_F02_T07_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck comes from:  
1. **Inefficient per-row `lapply` in `build_neighbor_lookup`**: It loops over ~6.46M rows, repeatedly building neighbor keys and indexing into a large hash table.  
2. **`compute_neighbor_stats` is also per-row**: Computing max/min/mean for each row individually is slow and memory-heavy.  
3. **Redundant work across years**: Neighbor relationships are static across years, but the lookup is recomputed for every row.  
4. **Pure R loops** on millions of rows are not feasible for 16 GB RAM.  

---

### **Optimization Strategy**
- **Precompute neighbor indices once per cell** (not per row) and reuse.
- **Vectorize aggregation by grouping** instead of looping over rows (use `data.table` or `collapse`).
- **Process by year in chunks** to avoid loading all 6.46M rows into memory at once.
- **Store neighbor relationships as integer vectors**, not lists of character keys.
- **Use fast join/merge operations** with `data.table`.

---

### **Optimized Workflow**
1. Convert `cell_data` to `data.table` keyed by `(id, year)`.
2. Precompute a mapping from each cell to its neighbors (`id` â†’ neighbor `id`s).
3. For each year:
   - Subset that year's data.
   - Join to its neighbors (self-join) using precomputed neighbor pairs.
   - Compute `max`, `min`, `mean` by `(id, year)` group.
4. Combine results back into the main table.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (id-neighbor_id)
# rook_neighbors_unique: list where each element = neighbors of id_order[i]
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Key for fast join
setkey(cell_data, id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_feature <- function(dt, neighbor_pairs, var_name) {
  # Subset columns needed
  dt_subset <- dt[, .(id, year, value = get(var_name))]
  
  # Join neighbor values: create all (id, year, neighbor_id)
  # Merge on neighbor_id + year to bring neighbor's value
  merged <- merge(
    neighbor_pairs[dt_subset, on = .(id), allow.cartesian = TRUE],
    dt_subset, by.x = c("neighbor_id", "year"), by.y = c("id", "year"),
    suffixes = c("", "_neighbor"), allow.cartesian = TRUE
  )
  
  # Compute stats by (id, year)
  stats <- merged[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Rename columns
  setnames(stats, c("id", "year", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  stats
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Process in chunks by year to reduce memory footprint
years <- unique(cell_data$year)
result_list <- vector("list", length(years))

for (yr in years) {
  dt_year <- cell_data[year == yr]
  features_list <- lapply(neighbor_source_vars, function(v) compute_neighbor_feature(dt_year, neighbor_pairs, v))
  
  # Merge all features back
  features <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE), features_list)
  
  # Merge into dt_year
  dt_year <- merge(dt_year, features, by = c("id", "year"), all.x = TRUE)
  
  result_list[[as.character(yr)]] <- dt_year
}

# Combine all years
cell_data_final <- rbindlist(result_list)
```

---

### **Why This Works**
- No per-row loops over 6.46M rows.
- Joins and aggregations are handled by `data.table` in optimized C code.
- Memory load is reduced by chunking by year.
- Preserves trained Random Forest model (same feature names, same estimand).
- Runs in **hours instead of 86+ hours**, feasible on 16 GB RAM.

**Further speed-up:**  
- Use `fread`/`fwrite` for I/O.
- Parallelize the yearly loop with `future.apply` or `parallel::mclapply`.
- If possible, compute all variables in a single melt-join-cast cycle for even better performance.
END_RESPONSE>>>
