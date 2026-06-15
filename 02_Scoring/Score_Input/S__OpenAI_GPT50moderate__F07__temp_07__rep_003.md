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
run_id: OpenAI_F07_T07_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) because:

1. **Inefficient neighbor lookup**: `build_neighbor_lookup` loops over 6.46M rows, repeatedly doing vector lookups and string concatenations.  
2. **Repeated lapply per variable**: For each of 5 variables, `compute_neighbor_stats` traverses all rows again, performing multiple loops and `rbind`.  
3. **Pure R loops on huge panel**: 6.46M rows Ã— 5 variables Ã— multiple operations in R lists is memory- and CPU-heavy.  
4. **No vectorization or sparse matrix usage**: The rook-neighbor structure is static, but code does not leverage efficient adjacency representations (e.g., sparse matrices).  

---

### **Optimization Strategy**
- **Represent neighbors as a sparse adjacency matrix** (rows = cell-year rows, columns = cell-year rows). But building a full 6.46M Ã— 6.46M matrix is infeasible â†’ Instead:
  - Use **cell-level adjacency** (344k Ã— 344k) and then apply it year-wise efficiently.
- **Precompute yearly offsets** so neighbor indices can be computed without string concatenation.
- **Vectorize neighbor aggregation** using `rowsum` or `Matrix` operations or `data.table` joins instead of per-row `lapply`.
- **Compute all stats in one pass** per variable without repeated loops.
- **Memory discipline**: Use integer indices, avoid character keys.

---

### **High-Level Approach**
1. Precompute **neighbor list at cell-level** (already have).
2. Expand to year-level by **index arithmetic**, not string concatenation.
3. Use **data.table** for fast grouping and merges.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute annual offsets
years <- sort(unique(cell_data$year))
n_years <- length(years)
id_to_idx <- match(cell_data$id, unique(cell_data$id))
n_cells <- length(unique(cell_data$id))

# Build neighbor list expanded to year-level using integer arithmetic
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
neighbor_lookup_idx <- vector("list", nrow(cell_data))

for (yr_idx in seq_along(years)) {
  year_offset <- (yr_idx - 1) * n_cells
  row_start <- year_offset + 1
  row_end   <- year_offset + n_cells
  
  # For each cell in this year, compute neighbor row indices
  for (cell in seq_len(n_cells)) {
    global_idx <- year_offset + cell
    nbs <- rook_neighbors_unique[[cell]]
    if (length(nbs) > 0) {
      neighbor_lookup_idx[[global_idx]] <- year_offset + nbs
    } else {
      neighbor_lookup_idx[[global_idx]] <- integer(0)
    }
  }
}

compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  maxv <- numeric(length(vals))
  minv <- numeric(length(vals))
  meanv <- numeric(length(vals))
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) {
      maxv[i] <- NA_real_
      minv[i] <- NA_real_
      meanv[i] <- NA_real_
    } else {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0) {
        maxv[i] <- NA_real_
        minv[i] <- NA_real_
        meanv[i] <- NA_real_
      } else {
        maxv[i] <- max(v)
        minv[i] <- min(v)
        meanv[i] <- mean(v)
      }
    }
  }
  cbind(max = maxv, min = minv, mean = meanv)
}

# Apply for all vars in one go
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_lookup_idx)
  cell_data[[paste0(var_name, "_nb_max")]]  <- stats[, "max"]
  cell_data[[paste0(var_name, "_nb_min")]]  <- stats[, "min"]
  cell_data[[paste0(var_name, "_nb_mean")]] <- stats[, "mean"]
}
```

---

### **Why This is Faster**
- Removed all string concatenations and character lookups.
- Precomputed integer offsets for each year â†’ O(n) pass.
- Single pass per variable with tight loop over integer indices.
- Scales linearly with 6.46M rows and uses simple numeric vectors.
- Memory: ~3Ã— number of rows for stats, no extra giant objects.

---

**Expected Improvement**: From 86+ hours to a few hours or less on a 16 GB laptop. Further speedups possible using `Rcpp` or `data.table` parallelization, but this pure R approach is already drastically faster **while preserving the original numerical estimand and the trained Random Forest model**.
END_RESPONSE>>>
