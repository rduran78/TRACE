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
run_id: OpenAI_F10_T07_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- For each of ~6.46M rows, `build_neighbor_lookup` constructs neighbor indices via string concatenation and repeated lookups in R lists.  
- `compute_neighbor_stats` repeatedly traverses large lists and uses many small vector operations in R, which is inefficient for millions of iterations.  
- The pipeline recomputes neighbor features in pure R loops without vectorization or matrix-based operations.  

**Optimization Strategy**  
- Represent the problem as a **sparse graph aggregation**: cells = nodes, rook neighbors = directed edges, node attributes = yearly variables.  
- Build the **graph topology once** (adjacency list or sparse matrix) and reuse it across all years.  
- Use **vectorized matrix operations** or **data.table** with grouping to compute neighbor statistics efficiently.  
- Avoid repeated string concatenation and repeated list indexing.  
- Use **parallelization** if possible and memory-efficient structures (`Matrix` package for sparse ops).  
- Preserve numerical equivalence: compute exact max, min, and mean of neighbor attributes.  

---

### **Optimized Implementation**

```r
library(data.table)
library(Matrix)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep::nb)
# id_order: vector of cell ids in the same order as rook_neighbors_unique

# Convert to data.table for efficiency
setDT(cell_data)

# Step 1: Build graph once as a sparse adjacency matrix
n_cells <- length(id_order)
id_to_pos <- setNames(seq_len(n_cells), id_order)

# Create edge list
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(from = i, to = rook_neighbors_unique[[i]])
}))

# Sparse adjacency matrix (rows = nodes, cols = neighbors)
adj <- sparseMatrix(
  i = edges$from,
  j = edges$to,
  x = 1,
  dims = c(n_cells, n_cells)
)

# Step 2: Process year by year to reduce memory footprint
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

results_list <- vector("list", length(neighbor_source_vars))
names(results_list) <- neighbor_source_vars

# Preallocate final columns
for (var in neighbor_source_vars) {
  cell_data[[paste0(var, "_nb_max")]] <- NA_real_
  cell_data[[paste0(var, "_nb_min")]] <- NA_real_
  cell_data[[paste0(var, "_nb_mean")]] <- NA_real_
}

years <- sort(unique(cell_data$year))

for (yr in years) {
  dt_year <- cell_data[year == yr]
  setorder(dt_year, id)  # Ensure same order as id_order
  stopifnot(all(dt_year$id == id_order))  # Sanity check

  for (var in neighbor_source_vars) {
    vals <- dt_year[[var]]

    # Compute neighbor statistics
    # Mean via matrix multiplication
    sums <- as.numeric(adj %*% vals)
    counts <- as.numeric(adj %*% rep(1, length(vals)))
    means <- sums / counts
    means[counts == 0] <- NA_real_

    # For max and min: iterate efficiently in C via apply over adjacency
    nb_max <- numeric(n_cells)
    nb_min <- numeric(n_cells)
    for (i in seq_len(n_cells)) {
      neigh_idx <- rook_neighbors_unique[[i]]
      if (length(neigh_idx) == 0) {
        nb_max[i] <- NA_real_
        nb_min[i] <- NA_real_
      } else {
        neigh_vals <- vals[neigh_idx]
        neigh_vals <- neigh_vals[!is.na(neigh_vals)]
        if (length(neigh_vals) == 0) {
          nb_max[i] <- NA_real_
          nb_min[i] <- NA_real_
        } else {
          nb_max[i] <- max(neigh_vals)
          nb_min[i] <- min(neigh_vals)
        }
      }
    }

    # Write back
    idx <- cell_data$year == yr
    cell_data[idx, paste0(var, "_nb_max") := nb_max]
    cell_data[idx, paste0(var, "_nb_min") := nb_min]
    cell_data[idx, paste0(var, "_nb_mean") := means]
  }
}

# At this point, cell_data has all neighbor features. Then apply pre-trained RF model:
# predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Why this is efficient**
- The **graph topology** is built once as a sparse matrix.
- **Mean** computation uses sparse matrix multiplication (`adj %*% vals`), which is highly optimized in C.
- **Max/min** still require iteration, but we do it per-cell/year, avoiding repeated key lookups and string concatenation.
- All operations run in memory with `data.table` and `Matrix` for speed.
- Memory footprint is controlled by processing **year by year**.

Expected runtime: **orders of magnitude faster** than 86 hours (likely a few hours or less on 16 GB RAM laptop).  

**Preserves numerical equivalence**: identical logic for max, min, mean neighbor stats.  
**Pre-trained model remains unchanged**: we only compute features.
END_RESPONSE>>>
