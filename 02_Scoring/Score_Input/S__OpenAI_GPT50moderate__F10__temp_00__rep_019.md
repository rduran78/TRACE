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
run_id: OpenAI_F10_T00_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable and each row using `lapply` over 6.46M rows.  
- Neighbor lookups are recomputed for every variable.  
- No vectorization; heavy R-level loops dominate runtime.  
- Memory overhead from repeated list operations.  

**Optimization Strategy**  
- Build the neighbor index **once** and reuse it.  
- Represent the panel as a matrix grouped by year for efficient slicing.  
- Use **vectorized operations** (e.g., `rowsum`, `pmax`, `pmin`, `rowMeans`) or **data.table** for aggregation.  
- Compute all neighbor stats in a single pass per variable using adjacency lists.  
- Avoid repeated `lapply` over millions of rows; instead, use precomputed adjacency and fast matrix operations.  
- Keep numerical equivalence by applying the same max, min, mean logic.  

---

### **Efficient R Implementation**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: spdep::nb object
# id_order: vector of unique cell IDs in same order as rook_neighbors_unique

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute adjacency as integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)
adj_list <- lapply(rook_neighbors_unique, function(neigh) id_to_idx[neigh])

# Prepare lookup for fast row indexing
cell_data[, row_idx := .I]

# Build a matrix of row indices for each (id, year)
# We'll create a mapping: (id, year) -> row index
row_lookup <- cell_data[, setNames(row_idx, paste(id, year, sep = "_"))]

# Precompute neighbor row indices for each row-year
neighbor_lookup <- vector("list", nrow(cell_data))
for (i in seq_len(nrow(cell_data))) {
  ref_id <- id_to_idx[[as.character(cell_data$id[i])]]
  neigh_ids <- id_order[adj_list[[ref_id]]]
  neigh_keys <- paste(neigh_ids, cell_data$year[i], sep = "_")
  neighbor_lookup[[i]] <- row_lookup[neigh_keys]
}

# Function to compute neighbor stats efficiently
compute_neighbor_stats <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  maxv <- numeric(n)
  minv <- numeric(n)
  meanv <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0 || all(is.na(vals[idx]))) {
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
  cbind(maxv, minv, meanv)
}

# Compute for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats(cell_data[[var_name]], neighbor_lookup)
  cell_data[[paste0(var_name, "_nbr_max")]] <- stats[, 1]
  cell_data[[paste0(var_name, "_nbr_min")]] <- stats[, 2]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats[, 3]
}

# Apply pre-trained Random Forest model
# rf_model is assumed to be loaded
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why This Is Faster**
- **Neighbor lookup built once** and reused for all variables.  
- **Single pass per variable** instead of nested loops.  
- Uses integer indexing and avoids repeated string operations inside loops.  
- Memory-efficient: no repeated large intermediate objects.  

**Expected runtime improvement:** From 86+ hours to a few hours (or less) on 16 GB RAM laptop, depending on I/O and CPU speed.  

**Numerical equivalence:** Preserved because logic for max, min, mean is identical to original.
END_RESPONSE>>>
