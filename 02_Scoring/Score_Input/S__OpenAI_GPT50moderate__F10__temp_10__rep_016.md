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
run_id: OpenAI_F10_T10_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current approach recomputes neighbor statistics for ~6.46M rows individually using `lapply`, which is highly inefficient for this scale.  
- Building neighbor keys per row and year repeatedly causes massive overhead.  
- No vectorization; operations done row-by-row with repeated lookups.  
- Neighbor topology (rook graph) is rebuilt for every year instead of reused.  

**Optimization Strategy**  
1. **Precompute graph topology once:** Leverage `rook_neighbors_unique` for adjacency.  
2. **Use matrix-based operations:** Avoid `lapply` over millions of rows; compute stats by grouping years and applying matrix multiplications or fast aggregations.  
3. **Split by year:** For each year, extract node attributes into an array, then compute neighbor stats via adjacency lists efficiently.  
4. **Reuse adjacency lists:** Apply them repeatedly on yearly slices without string concatenation.  
5. **Store output efficiently:** Use `data.table` for fast join and column binding.  

---

### **Efficient Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: a list (spdep::nb object) where each element is neighbor IDs for each cell

# Convert to data.table
setDT(cell_data)

# Step 1: Build reusable adjacency index list
id_order <- sort(unique(cell_data$id))  # ensure alignment
id_to_pos <- setNames(seq_along(id_order), id_order)
adj_list <- lapply(rook_neighbors_unique, function(nb_ids) id_to_pos[nb_ids])

# Step 2: Prepare container for neighbor stats
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stat_names <- c("max", "min", "mean")

# Step 3: Efficient computation by year
result_list <- vector("list", length(vars) * length(stat_names))

# Create placeholder for final columns
col_names <- c()
for (var in vars) {
  for (stat in stat_names) {
    col_names <- c(col_names, paste(var, stat, sep = "_"))
  }
}
cell_data[, (col_names) := NA_real_]

# Split computation by year for memory efficiency
years <- unique(cell_data$year)

for (yr in years) {
  slice <- cell_data[year == yr]
  vals <- slice[, ..vars]
  
  # For each variable, compute neighbor stats for all nodes
  for (j in seq_along(vars)) {
    vname <- vars[j]
    vvals <- vals[[j]]
    
    max_v <- numeric(length(vvals))
    min_v <- numeric(length(vvals))
    mean_v <- numeric(length(vvals))
    
    # Loop over nodes, but only indexing neighbors - adjacency reused
    for (i in seq_along(adj_list)) {
      nbs <- adj_list[[i]]
      if (length(nbs) == 0) {
        max_v[i] <- NA
        min_v[i] <- NA
        mean_v[i] <- NA
      } else {
        nvals <- vvals[nbs]
        nvals <- nvals[!is.na(nvals)]
        if (length(nvals) == 0) {
          max_v[i] <- NA
          min_v[i] <- NA
          mean_v[i] <- NA
        } else {
          max_v[i] <- max(nvals)
          min_v[i] <- min(nvals)
          mean_v[i] <- mean(nvals)
        }
      }
    }
    
    # Write back to cell_data
    cell_data[year == yr, paste0(vname, "_max") := max_v]
    cell_data[year == yr, paste0(vname, "_min") := min_v]
    cell_data[year == yr, paste0(vname, "_mean") := mean_v]
  }
}

# Preserve numerical equivalence; Random Forest model application unchanged
# Apply trained RF model
predictions <- predict(trained_rf_model, newdata = cell_data)
cell_data[, prediction := predictions]
```

**Key Efficiency Gains**  
- **Adjacency computed once, reused across all years.**  
- Operations grouped per year to keep memory manageable (~6.46M rows processed in chunks).  
- No repeated neighbor key creation; direct integer lookup.  
- Still deterministic, preserves max/min/mean computation logic from original code.  

With proper indexing and chunking, this approach reduces time from **86+ hours to a few hours**, dominated by simple loops over ~344K nodes/year rather than 6.46M rows individually recomputing adjacency. For further speedup, consider:  
- **Parallelizing inner loop** with `parallel::mclapply` or `future.apply`.  
- Using **Rcpp** for neighbor stats to achieve multi-threaded performance.
END_RESPONSE>>>
