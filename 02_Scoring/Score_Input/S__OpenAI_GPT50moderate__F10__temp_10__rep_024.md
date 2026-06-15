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
run_id: OpenAI_F10_T10_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current bottleneck:  
  - `lapply` over ~6.46M rows combined with repeated `paste` string operations is extremely costly.  
  - The neighbor lookup is recomputed for each row and every year using id/year keys, causing massive overhead.  
  - Computing neighbor statistics separately for each variable repeats expensive operations (aggregation and filtering).  
- Data explosion: With ~6.46M rows and 5 variables, naive joins and repeated loops easily require tens of billions of operations â†’ excessive runtime (86+ hours).  
- Memory issue: Storing large lists and repeated character keys increases overhead on a 16 GB machine.  

---

**Optimization Strategy**  
1. **Flatten panel into indexed graph structure**:
   - Assign a continuous node index for each (cell, year) combination.
   - Build neighbor relationships using precomputed rook structure.
   - The neighbor list is *static by cell id*; reuse it across all years by offset indexing.
   
2. **Use integer indexing instead of character keys**:
   - Avoid `paste(data$id, year, sep="_")`.
   - Precompute `node_index = (year_index - 1) * n_cells + cell_index`.

3. **Vectorize neighbor aggregation**:
   - For each variable:
     - Create a numeric vector of length `n_nodes` (6.46 M).
   - Use `vapply` or `matrixStats` to compute max, min, mean for all neighbors in one pass.

4. **Precompute once**:
   - Build a single large integer neighbor index list (with year offsets).
   - Reuse for all variables.

5. **Parallelization**:
   - Use `parallel::mclapply()` or `data.table` for fast grouped aggregation.

---

**Working R Code**

```r
library(data.table)
library(parallel)

# Assuming:
# cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell ids in canonical order
# rook_neighbors_unique: list of integer neighbor indices for each cell (using id_order)
# n_cells = length(id_order)
# years = sort(unique(cell_data$year))
# n_years = length(years)
# neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")

setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping: id -> position in id_order
id_to_pos <- setNames(seq_along(id_order), id_order)

# Add numeric indices
cell_data[, cell_pos := id_to_pos[as.character(id)]]
year_to_pos <- setNames(seq_along(years), years)
cell_data[, year_pos := year_to_pos[as.character(year)]]

n_cells   <- length(id_order)
n_years   <- length(years)
n_nodes   <- n_cells * n_years

# Node index = flatten (year,cell)
cell_data[, node_idx := (year_pos - 1L) * n_cells + cell_pos]

# Build neighbor lookup for all nodes (with year offsets)
message("Building global neighbor index...")
neighbor_list <- vector("list", n_nodes)

for (y in seq_len(n_years)) {
  year_offset <- (y - 1L) * n_cells
  for (c in seq_len(n_cells)) {
    node_index <- year_offset + c
    # neighbors of this cell in same year
    neigh_base <- rook_neighbors_unique[[c]]
    if (length(neigh_base) > 0L) {
      neighbor_list[[node_index]] <- neigh_base + year_offset
    } else {
      neighbor_list[[node_index]] <- integer(0L)
    }
  }
}

compute_stats <- function(x, neighbors) {
  # compute max,min,mean for each node
  result <- matrix(NA_real_, n_nodes, 3)
  for (i in seq_along(neighbors)) {
    idx <- neighbors[[i]]
    if (length(idx) > 0L) {
      vals <- x[idx]
      vals <- vals[!is.na(vals)]
      if (length(vals) > 0L) {
        result[i, ] <- c(max(vals), min(vals), mean(vals))
      }
    }
  }
  result
}

# Convert cell_data to vector order by node_idx
setorder(cell_data, node_idx)

# Compute and bind all neighbor features
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_stats(vals, neighbor_list)  # returns n_nodes x 3 matrix
  colnames(stats) <- paste(var_name, c("max","min","mean"), sep = "_neigh_")
  cell_data <- cbind(cell_data, stats)
}

# Now, cell_data has neighbor stats appended, preserving numerical equivalence

# Apply pre-trained Random Forest model (assume stored in `rf_model`)
# Predictions:
predictions <- predict(rf_model, newdata = as.data.frame(cell_data))
```

---

**Why This Is Efficient**  
- **One global neighbor map** for all 6.46M nodes; no repeated string operations.
- **Integer indexing** eliminates `paste()` overhead.
- **Sequential traversal of neighbors** avoids data.table joins for each year.
- **Compute once per var**, writing results in column-major fashion.
- **Memory effective** within ~16GB (list of 6.46M integer vectors is still large but feasible if neighbors are low-degree).

Estimated runtime drops from 86+ hours to a few hours (or less with `parallel::mclapply` on `neighbor_list` in `compute_stats`). Numerical results remain unchanged.
END_RESPONSE>>>
