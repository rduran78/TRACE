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
run_id: OpenAI_F05_T03_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(...)`) and named vector lookups (`idx_lookup[...]`) inside the `lapply` in `build_neighbor_lookup` are a **local inefficiency**, but they occur in a loop over **6.46 million rows**, making it a major bottleneck. The root cause is that for every row, you rebuild neighbor keys and perform string-based lookups instead of using numeric indices. This is not just a micro-inefficiencyâ€”itâ€™s symptomatic of an **algorithmic design issue**: the neighbor relationships are static across years, but the code recomputes them for every row and every variable.

**Optimization Strategy**  
- Precompute a **numeric neighbor index matrix** once, avoiding string concatenation and repeated lookups.
- Exploit the panel structure: neighbors are the same across years, so replicate indices across years instead of recomputing.
- Use **vectorized operations** or `matrixStats` instead of per-row `lapply` for neighbor statistics.
- Keep memory in check by storing neighbor indices in a list or sparse structure.

---

### **Reformulated Approach**

1. Map `(id, year)` â†’ row index **once** using a fast join.
2. For each cell id, get its neighbors (static), then expand across all years.
3. Store neighbor indices in a list of integer vectors (one per row).
4. Compute neighbor stats in a vectorized way.

---

### **Working R Code**

```r
library(data.table)
library(matrixStats)

# Assume: data.table with columns id, year, and all variables
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute row index for each (id, year)
cell_data[, row_idx := .I]

# Build neighbor lookup efficiently
build_neighbor_lookup_fast <- function(cell_data, id_order, neighbors) {
  # Map id -> neighbor ids
  id_to_neighbors <- lapply(neighbors, function(nbrs) id_order[nbrs])
  names(id_to_neighbors) <- as.character(id_order)
  
  # Expand to full panel
  years <- sort(unique(cell_data$year))
  n_rows <- nrow(cell_data)
  
  # Preallocate list
  neighbor_lookup <- vector("list", n_rows)
  
  # Build a fast lookup table for (id, year) -> row index
  idx_map <- cell_data[, .(id, year, row_idx)]
  setkey(idx_map, id, year)
  
  # Fill neighbor_lookup
  for (id in id_order) {
    ref_rows <- idx_map[J(id, years), row_idx]
    nbr_ids  <- id_to_neighbors[[as.character(id)]]
    
    if (length(nbr_ids) == 0) next
    
    # For each year, get neighbor row indices
    for (k in seq_along(years)) {
      y <- years[k]
      nbr_rows <- idx_map[J(nbr_ids, y), row_idx]
      neighbor_lookup[[ref_rows[k]]] <- as.integer(nbr_rows[!is.na(nbr_rows)])
    }
  }
  
  neighbor_lookup
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  max_vals <- numeric(n)
  min_vals <- numeric(n)
  mean_vals <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) {
      max_vals[i] <- min_vals[i] <- mean_vals[i] <- NA_real_
    } else {
      nbr_vals <- vals[idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) {
        max_vals[i] <- min_vals[i] <- mean_vals[i] <- NA_real_
      } else {
        max_vals[i] <- max(nbr_vals)
        min_vals[i] <- min(nbr_vals)
        mean_vals[i] <- mean(nbr_vals)
      }
    }
  }
  
  cbind(max_vals, min_vals, mean_vals)
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Is Faster**
- Eliminates repeated `paste` and string-based lookups (O(n) â†’ O(1) per row).
- Precomputes neighbor relationships once, reuses across all variables.
- Uses numeric indexing instead of character keys.
- Avoids redundant work across years.

---

**Expected Performance Gain**  
From 86+ hours â†’ **minutes to a few hours**, depending on I/O and memory, because the algorithm now scales linearly with rows and avoids repeated expensive operations.
END_RESPONSE>>>
