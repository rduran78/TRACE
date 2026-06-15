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
run_id: OpenAI_F10_T03_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for ~6.46M rows using `lapply` and `do.call(rbind, ...)`, which is highly inefficient in R.  
- Neighbor lookups are recomputed per row and per variable, causing redundant work.  
- No vectorization or efficient data structures (e.g., matrices) are used; everything is list-based.  
- The graph topology is rebuilt or accessed inefficiently for each year-variable combination.  

**Optimization Strategy**  
- Precompute a **sparse adjacency matrix** (or CSR-like structure) for the rook graph (344,208 nodes).  
- Represent the panel as a matrix of size (nodes Ã— years) for each variable.  
- Use **matrix operations** or efficient loops in C++ (via `Rcpp`) or `Matrix` package to aggregate neighbor values.  
- Compute max, min, and mean in a single pass per variable using the adjacency structure.  
- Avoid repeated string concatenation and list indexing; use integer-based indexing.  
- Process by variable in blocks to reduce memory overhead.  
- Append computed features back to the main data frame by row order.  

**Working R Code (Optimized)**  
```r
library(Matrix)
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object

# 1. Build adjacency as sparse matrix (nodes x nodes)
n_nodes <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list, use.names = FALSE)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_nodes, n_nodes))

# 2. Prepare panel data as matrix: rows = nodes, cols = years
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_index <- match(cell_data$year, years)
id_index <- match(cell_data$id, id_order)

# Convert to data.table for fast reshaping
setDT(cell_data)
setkey(cell_data, id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_matrix <- function(var_name) {
  # Create node x year matrix
  mat <- matrix(NA_real_, nrow = n_nodes, ncol = n_years)
  mat[cbind(id_index, year_index)] <- cell_data[[var_name]]
  
  # For each year, aggregate neighbors
  max_mat <- matrix(NA_real_, n_nodes, n_years)
  min_mat <- matrix(NA_real_, n_nodes, n_years)
  mean_mat <- matrix(NA_real_, n_nodes, n_years)
  
  for (y in seq_len(n_years)) {
    vals <- mat[, y]
    # For each node, get neighbor values using adjacency
    # Use adjacency %*% indicator to find neighbors
    # Extract neighbor indices efficiently
    for (i in seq_len(n_nodes)) {
      nbr_idx <- adj_list[[i]]
      if (length(nbr_idx) > 0) {
        nbr_vals <- vals[nbr_idx]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) > 0) {
          max_mat[i, y] <- max(nbr_vals)
          min_mat[i, y] <- min(nbr_vals)
          mean_mat[i, y] <- mean(nbr_vals)
        }
      }
    }
  }
  
  # Return as data.table aligned with cell_data
  res <- data.table(
    max = max_mat[cbind(id_index, year_index)],
    min = min_mat[cbind(id_index, year_index)],
    mean = mean_mat[cbind(id_index, year_index)]
  )
  setnames(res, c(paste0(var_name, "_nbr_max"),
                  paste0(var_name, "_nbr_min"),
                  paste0(var_name, "_nbr_mean")))
  res
}

# 3. Compute for all variables and bind
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_features <- lapply(neighbor_source_vars, compute_neighbor_stats_matrix)
cell_data <- cbind(cell_data, do.call(cbind, all_features))

# cell_data now has neighbor features; ready for prediction with pre-trained RF
# Example: preds <- predict(pretrained_rf_model, newdata = cell_data)
```

**Key Improvements**  
- Graph topology is built once as `adj_list` and `adj_mat`.  
- Node-year data is reshaped into matrices for efficient access.  
- Loops are reduced to node-level per year (still O(NÃ—Y), but far faster than list-based row operations).  
- Memory layout is contiguous, avoiding repeated allocations.  
- Preserves numerical equivalence with original pipeline.  

**Expected Runtime**  
On 16 GB RAM, this approach should reduce runtime from 86+ hours to a few hours (or less if parallelized or ported to Rcpp). Further speedups:  
- Parallelize the inner loop over nodes using `parallel` or `future.apply`.  
- Implement the neighbor aggregation in C++ for an additional 10â€“20Ã— speedup.
END_RESPONSE>>>
