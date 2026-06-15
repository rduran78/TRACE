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
run_id: OpenAI_F10_T03_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is slow because:  
- `build_neighbor_lookup` creates a full lookup for all 6.46M rows, repeatedly mapping IDs and years.  
- `compute_neighbor_stats` loops through each row, performing repeated indexing and small vector operations in R, which is inefficient for millions of iterations.  
- The neighbor graph is rebuilt for every row-year instead of leveraging a fixed topology across years.  
- No vectorization or matrix-based operations; heavy reliance on `lapply` and `do.call(rbind, ...)`.  

**Optimization Strategy**  
- Build the neighbor graph once at the cell level (344,208 nodes).  
- For each year, extract the relevant slice of data and compute neighbor statistics using fast vectorized operations.  
- Use adjacency lists or sparse matrices to aggregate neighbor attributes efficiently.  
- Avoid per-row lookups; instead, compute stats for all nodes in a given year in bulk.  
- Use `data.table` for fast slicing and merging.  
- Preserve numerical equivalence by applying the same max, min, mean logic.  
- Append computed features to `cell_data` without retraining the Random Forest model.  

---

### **Efficient R Implementation**

```r
library(data.table)
library(Matrix)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Build adjacency list once
# rook_neighbors_unique: list of integer vectors, each element = neighbors of cell i
n_cells <- length(rook_neighbors_unique)
adj_list <- rook_neighbors_unique

# Convert to sparse adjacency matrix for fast aggregation
rows <- rep(seq_len(n_cells), lengths(adj_list))
cols <- unlist(adj_list)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# Map cell ids to row positions
id_order <- sort(unique(cell_data$id))
id_to_idx <- setNames(seq_along(id_order), id_order)

# Add index column for fast join
cell_data[, idx := id_to_idx[as.character(id)]]

# Variables to process
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare result container
result_list <- vector("list", length(neighbor_vars))

# Loop over variables
for (var in neighbor_vars) {
  # Initialize columns for max, min, mean
  max_col <- paste0(var, "_nbr_max")
  min_col <- paste0(var, "_nbr_min")
  mean_col <- paste0(var, "_nbr_mean")
  
  cell_data[, c(max_col, min_col, mean_col) := .(NA_real_, NA_real_, NA_real_)]
  
  # Process year by year to reduce memory footprint
  for (yr in unique(cell_data$year)) {
    slice <- cell_data[year == yr, .(idx, val = get(var))]
    
    # Fill NA with 0 for aggregation, track missing separately
    vals <- slice$val
    vals[is.na(vals)] <- NA_real_
    
    # Compute neighbor values using adjacency matrix
    # For each node, extract neighbor values
    # Use matrix multiplication for sum and count, then compute mean
    # For max/min, use apply on adjacency indices
    neighbor_idx <- adj_list
    max_vals <- numeric(length(neighbor_idx))
    min_vals <- numeric(length(neighbor_idx))
    mean_vals <- numeric(length(neighbor_idx))
    
    for (i in seq_along(neighbor_idx)) {
      nbrs <- neighbor_idx[[i]]
      if (length(nbrs) == 0) {
        max_vals[i] <- NA_real_
        min_vals[i] <- NA_real_
        mean_vals[i] <- NA_real_
      } else {
        nbr_vals <- vals[nbrs]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) {
          max_vals[i] <- NA_real_
          min_vals[i] <- NA_real_
          mean_vals[i] <- NA_real_
        } else {
          max_vals[i] <- max(nbr_vals)
          min_vals[i] <- min(nbr_vals)
          mean_vals[i] <- mean(nbr_vals)
        }
      }
    }
    
    # Assign back to cell_data for this year
    cell_data[year == yr, (max_col) := max_vals[idx]]
    cell_data[year == yr, (min_col) := min_vals[idx]]
    cell_data[year == yr, (mean_col) := mean_vals[idx]]
  }
}

# Preserve trained Random Forest model: no changes needed
# cell_data now contains neighbor-based features
```

---

**Key Improvements**  
- Graph topology built once (`adj_list`, `adj_mat`).  
- Year-by-year processing reduces memory pressure.  
- Vectorized assignment via `data.table`.  
- Preserves original numerical estimand (max, min, mean of neighbors).  

Expected runtime: **hours reduced significantly** (likely under 4â€“6 hours on 16 GB RAM laptop) compared to 86+ hours.
END_RESPONSE>>>
