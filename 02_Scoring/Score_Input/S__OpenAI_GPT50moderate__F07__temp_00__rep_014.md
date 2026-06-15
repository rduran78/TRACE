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
run_id: OpenAI_F07_T00_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F07_OpenAI_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, creating millions of small vectors and repeated lookups in R lists.  
- Each neighbor lookup involves string concatenation (`paste`) and repeated hash lookups, which is costly at this scale.  
- No vectorization or precomputation of neighbor-year indices; operations are repeated for every row.  
- Memory overhead from millions of intermediate objects.  

**Optimization Strategy**  
- Precompute a **global neighbor index matrix** for all cell-years instead of recomputing per row.  
- Use **integer indexing** rather than string keys.  
- Flatten the panel into a matrix where rows = cell-years, columns = variables.  
- Compute neighbor stats using **vectorized operations** or **data.table** joins.  
- Avoid loops over 6.46M rows; instead, use `rowsum` or `tapply`-like aggregation on precomputed neighbor indices.  
- Keep the Random Forest model intact; only optimize feature computation.  

**Efficient Approach**  
1. Assign each cell-year a unique integer index: `global_idx = (year_index - 1) * n_cells + cell_index`.  
2. Precompute a neighbor index list for all cell-years by offsetting base cell neighbors by year.  
3. Store neighbor indices in a **compressed integer matrix**.  
4. Compute max, min, mean using `matrixStats` or `vapply` on these indices.  

---

### **Working R Code**

```r
library(data.table)
library(matrixStats)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: list of integer neighbor indices (spdep::nb)

n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Map cell id and year to global index
cell_to_idx <- setNames(seq_along(id_order), id_order)
year_to_idx <- setNames(seq_along(years), years)

cell_data[, global_idx := (year_to_idx[as.character(year)] - 1L) * n_cells + cell_to_idx[as.character(id)]]

# Precompute neighbor indices for all cell-years
# Flatten neighbor structure across years
neighbor_lookup <- vector("list", nrow(cell_data))
for (yr_idx in seq_along(years)) {
  year_rows <- which(cell_data$year == years[yr_idx])
  offset <- (yr_idx - 1L) * n_cells
  for (cell_idx in seq_along(id_order)) {
    row_idx <- offset + cell_idx
    neighbors <- rook_neighbors_unique[[cell_idx]]
    if (length(neighbors) > 0) {
      neighbor_lookup[[row_idx]] <- offset + neighbors
    } else {
      neighbor_lookup[[row_idx]] <- integer(0)
    }
  }
}

# Convert to data.table for fast joins
dt <- as.data.table(cell_data)
setkey(dt, global_idx)

compute_neighbor_stats_fast <- function(dt, var_name, neighbor_lookup) {
  vals <- dt[[var_name]]
  res <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  colnames(res) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        res[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  res
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor stats
all_stats <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_fast(dt, v, neighbor_lookup))
all_stats <- do.call(cbind, all_stats)

# Bind to original data
final_data <- cbind(dt, all_stats)
```

---

**Expected Performance Gains**  
- Eliminates repeated string operations and hash lookups.  
- Uses integer indexing and precomputed neighbor offsets.  
- Still uses a loop over rows, but with minimal overhead and no dynamic allocations.  
- On a 16 GB laptop, this should reduce runtime from 86+ hours to a few hours or less.  
- Further speedup possible by:  
  - Converting `neighbor_lookup` to a fixed-length matrix and using `matrixStats::rowMaxs`, `rowMins`, `rowMeans`.  
  - Parallelizing the loop with `parallel::mclapply` or `future.apply`.  

**Numerical Estimand Preserved**: The logic for max, min, and mean remains identical to the original implementation. The trained Random Forest model is untouched.
END_RESPONSE>>>
