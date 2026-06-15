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
run_id: OpenAI_F08_T07_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor statistics (`max`, `min`, `mean`) for every row (cell-year) by iterating through all 6.46M rows and repeatedly pulling neighbor indices. This is extremely inefficient because:  
- The neighbor structure is static across years, but the computation redundantly reuses it for each year-row.  
- The algorithm is row-wise, not vectorized, so it performs millions of small operations instead of grouped operations.  
- It repeatedly subsets vectors inside nested loops, causing memory and time overhead.  

**Optimization Strategy**  
1. **Exploit static neighbor relationships**: Precompute neighbor IDs once.  
2. **Group by year**: For each year and variable, compute neighbor stats in a vectorized way using the static lookup.  
3. Use **matrix operations or `data.table` joins** instead of per-row lapply.  
4. **Chunk processing** to fit in memory while leveraging fast aggregation.  
5. Preserve the trained Random Forest model and original estimand (same stats: neighbor `max`, `min`, `mean`).  

---

### **Optimized R Implementation**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute static neighbor lookup: list of neighbor ids per cell id
# rook_neighbors_unique: list of integer vectors aligned to id_order
neighbor_list <- rook_neighbors_unique   # already loaded
names(neighbor_list) <- as.character(id_order)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Function to compute neighbor stats for one variable across all years
compute_neighbor_stats_fast <- function(dt, var_name, neighbor_list) {
  years <- unique(dt$year)
  
  # Prepare result storage
  res_list <- vector("list", length(years))
  
  for (yr_idx in seq_along(years)) {
    yr <- years[yr_idx]
    sub <- dt[year == yr, .(id, val = get(var_name))]
    setkey(sub, id)
    
    # For each id, compute stats from neighbor ids
    stats_mat <- matrix(NA_real_, nrow = nrow(sub), ncol = 3)
    
    for (i in seq_len(nrow(sub))) {
      nid <- neighbor_list[[as.character(sub$id[i])]]
      if (length(nid) > 0) {
        neigh_vals <- sub[J(nid), val, nomatch = 0L]
        if (length(neigh_vals) > 0) {
          stats_mat[i, ] <- c(max(neigh_vals, na.rm = TRUE),
                              min(neigh_vals, na.rm = TRUE),
                              mean(neigh_vals, na.rm = TRUE))
        }
      }
    }
    
    res <- data.table(id = sub$id,
                      year = yr,
                      max = stats_mat[, 1],
                      min = stats_mat[, 2],
                      mean = stats_mat[, 3])
    res_list[[yr_idx]] <- res
  }
  
  rbindlist(res_list)
}

# Apply for all variables and merge back
for (var_name in neighbor_source_vars) {
  message("Processing ", var_name, " ...")
  stats_dt <- compute_neighbor_stats_fast(cell_data, var_name, neighbor_list)
  setnames(stats_dt, c("id", "year", "max", "min", "mean"),
           c("id", "year",
             paste0(var_name, "_nbr_max"),
             paste0(var_name, "_nbr_min"),
             paste0(var_name, "_nbr_mean")))
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"))
}
```

---

### **Why This Is Faster**
- **Static neighbor structure** is reused without rebuilding per row.
- **Per-year subsetting** reduces working set size from 6.46M rows to ~344k rows per iteration.
- **Single pass for each variable-year**, no redundant lookups.
- Uses **data.table** for efficient joins and memory handling.
- Complexity drops from ~6.46M Ã— neighbors operations to ~28 Ã— 344k Ã— neighbors.

**Expected Runtime**  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to a few hours (depending on CPU), while preserving model compatibility and estimands.  

---

âœ… **Preserves original RF model, numerical estimand, and neighbor stats.**  
âœ… **Memory-safe and CPU-efficient given constraints.**
END_RESPONSE>>>
