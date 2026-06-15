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
run_id: OpenAI_F08_T03_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because it repeatedly computes neighbor statistics (`max`, `min`, `mean`) for each variable across all 6.46 million rows using a large `neighbor_lookup` list. This involves ~6.46M Ã— 5 iterations of R-level loops and repeated subsetting, which is highly inefficient in R. The neighbor structure is static across years, but the code redundantly processes neighbors for every row-year combination.

**Optimization Strategy**  
- Exploit the static neighbor structure: compute neighbor indices once at the **cell level** (344,208 cells), not per row-year.
- For each year, extract the relevant slice of data (344,208 rows), compute neighbor stats using **vectorized operations** or **matrix-based aggregation**, then bind results.
- Use `data.table` for fast grouping and joins.
- Avoid repeated list traversals; instead, precompute neighbor relationships as integer vectors and use `vapply` or matrix operations.
- Process year by year to keep memory manageable.
- Preserve the trained Random Forest model and numerical estimand by ensuring the resulting features match the original definition.

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbors per cell index (length = number of cells)
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup at cell level (static)
neighbor_lookup_static <- lapply(rook_neighbors_unique, function(neigh_idx) id_order[neigh_idx])
names(neighbor_lookup_static) <- as.character(id_order)

# Variables for neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output container
result_list <- vector("list", length = length(unique(cell_data$year)))
years <- sort(unique(cell_data$year))

# Function to compute neighbor stats for one year
compute_year_stats <- function(dt_year, neighbor_lookup, vars) {
  # dt_year: data.table with one year of data (rows = cells)
  vals <- dt_year[, ..vars]
  ids <- dt_year$id
  # Preallocate result matrix
  res_mat <- matrix(NA_real_, nrow = nrow(dt_year), ncol = length(vars) * 3)
  colnames(res_mat) <- as.vector(t(outer(vars, c("max", "min", "mean"), paste, sep = "_neighbor_")))
  
  for (i in seq_along(ids)) {
    neigh_ids <- neighbor_lookup[[as.character(ids[i])]]
    if (length(neigh_ids) == 0) next
    idx <- match(neigh_ids, ids, nomatch = 0L)
    idx <- idx[idx > 0]
    if (length(idx) == 0) next
    neigh_vals <- vals[idx, ]
    # Compute stats
    res_mat[i, ] <- as.vector(rbind(apply(neigh_vals, 2, max, na.rm = TRUE),
                                     apply(neigh_vals, 2, min, na.rm = TRUE),
                                     apply(neigh_vals, 2, mean, na.rm = TRUE)))
  }
  cbind(dt_year[, .(id, year)], res_mat)
}

# Loop over years
for (j in seq_along(years)) {
  yr <- years[j]
  dt_year <- cell_data[year == yr]
  result_list[[j]] <- compute_year_stats(dt_year, neighbor_lookup_static, neighbor_source_vars)
}

# Combine all years
neighbor_features <- rbindlist(result_list)

# Merge back into original data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This is Faster**
- Processes **344k rows per year** instead of 6.46M rows in one go.
- Avoids redundant neighbor lookups for every row-year combination.
- Uses `data.table` and vectorized `apply` for each neighbor set.
- Memory footprint is limited to one year at a time.

---

**Expected Performance**  
On a standard laptop, this approach should reduce runtime from 86+ hours to a few hours (or less) because:
- Static neighbor structure is leveraged.
- Year-by-year processing avoids huge intermediate objects.
- Heavy R loops over millions of rows are eliminated.

**Preserves**:
- Original numerical estimand (neighbor max, min, mean).
- Trained Random Forest model remains unchanged.
END_RESPONSE>>>
