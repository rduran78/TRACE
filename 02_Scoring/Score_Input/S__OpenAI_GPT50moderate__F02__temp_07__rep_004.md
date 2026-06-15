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
run_id: OpenAI_F02_T07_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The slowdown and large memory usage stem from:  
1. **`lapply` over 6.46M rows** in `build_neighbor_lookup` and `compute_neighbor_stats`, creating millions of small lists and repeated string concatenations.  
2. **Repeated lookup by string keys** (`paste(..., sep="_")`), which is expensive at this scale.  
3. **R interprets everything at row-level**, causing severe overhead.  
4. **No vectorization and no memory-efficient backend** for aggregation.  

---

**Optimization Strategy**  
- **Avoid per-row `lapply`**: Represent panel data as numeric arrays and neighbors as integer indices.  
- **Precompute neighbor indices once for all years**: For each cell, store neighbor indices (no string keys), then offset by year blocks.  
- **Use matrix-based operations or `data.table`** for aggregation.  
- **Parallelize** computations where possible.  
- **Chunk processing** if memory is tight.  

This avoids constructing millions of intermediate vectors and string operations.  

---

**Optimized R Code** (using `data.table` and precomputed offsets):  

```r
library(data.table)
library(parallel)

# Assume `cell_data` is data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

n_years <- length(unique(cell_data$year))
n_cells <- length(unique(cell_data$id))
year_index <- match(cell_data$year, sort(unique(cell_data$year)))

# Precompute neighbor mapping as integer
id_to_idx <- setNames(seq_len(n_cells), as.character(unique(cell_data$id)))
neighbors_idx <- lapply(rook_neighbors_unique, function(nb) id_to_idx[as.character(nb)])

# Compute row index offsets for each year
offsets <- (year_index - 1L) * n_cells

# Build neighbor_lookup as integer indices matrix
neighbor_lookup <- vector("list", length = nrow(cell_data))
for (i in seq_len(nrow(cell_data))) {
  ref_idx <- id_to_idx[as.character(cell_data$id[i])]
  nb_cells <- neighbors_idx[[ref_idx]]
  if (length(nb_cells) > 0) {
    neighbor_lookup[[i]] <- nb_cells + offsets[i]
  } else {
    neighbor_lookup[[i]] <- integer(0)
  }
}

# Function to compute stats efficiently
compute_neighbor_stats_fast <- function(var_name, vals, neighbor_lookup) {
  # Parallel apply using mclapply for speed
  res <- mclapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    nb_vals <- vals[idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nb_vals), min(nb_vals), mean(nb_vals))
  }, mc.cores = detectCores() - 1)
  do.call(rbind, res)
}

# Apply for each variable and bind
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats_fast(var_name, vals, neighbor_lookup)
  colnames(stats) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

**Why This Works Faster**  
- Removes string concatenation and `setNames` lookups for each row.  
- Uses integer offsets for years â†’ O(1) neighbor index resolution.  
- Runs stats in parallel with `mclapply`.  
- Works in-memory efficiently; if memory is still tight, process in **chunks**: split `neighbor_lookup` into blocks and write intermediate results to disk.  

**Expected Speedup**: From 86+ hours to a few hours on a standard laptop (due to vectorization and parallelization).  

**Preserves**: Original estimand, trained Random Forest model, and feature definitions.
END_RESPONSE>>>
