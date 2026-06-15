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
run_id: OpenAI_F10_T10_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) due to repeated high-cost operations for each `var_name` and each row-year (â‰ˆ6.46M iterations). Inefficiencies include:  
- Recomputing neighbor lookups repeatedly inside loops.  
- Using `lapply` and list-processing for millions of rows instead of vectorized operations.  
- Building large intermediate structures repeatedly.  
- No use of efficient matrix operations or sparse graph representations, leading to high memory overhead.  

**Optimization Strategy**  
- **Precompute and reuse neighbor index structure** for all years: build a unified lookup matrix or adjacency list across all `(cell_id, year)` rows just once.  
- Use **fast join or sparse matrix multiplication**: represent rook relationships as adjacency sparse matrix and compute aggregated stats (max/min/mean) using vectorized operations.  
- Combine all years in a single pass: convert data to matrix, run computations for each feature in block operations.  
- Use **data.table** for efficient indexing and grouping, reducing overhead of repeated `lapply`.  
- Avoid building character keys (`paste`) repeatedlyâ€”create integer mappings instead.  

**Efficient Implementation in R**  
Below uses `Matrix` for sparse adjacency and vectorized neighbor-aggregation over all years:

```r
library(data.table)
library(Matrix)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of spatial ids
# rook_neighbors_unique: neighbor list from spdep (list of integer vectors)

# Convert to integer indices
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

n_cells <- length(id_order)
n_years <- length(unique(cell_data$year))
n_rows <- nrow(cell_data) # ~6.46M

# Construct adjacency as sparse dgCMatrix (cells only, static over years)
# rook_neighbors_unique: list with positions matching id_order
i_idx <- rep(seq_along(rook_neighbors_unique), lengths(rook_neighbors_unique))
j_idx <- unlist(rook_neighbors_unique)
adj_base <- sparseMatrix(i = i_idx, j = j_idx, dims = c(n_cells, n_cells), repr = "C")

# Build block-diagonal adjacency for all years
Adj <- kronecker(Diagonal(n_years), adj_base)

# Create feature matrix (rows align with cell-year order)
setorder(cell_data, id, year)
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
X <- as.matrix(cell_data[, ..vars])

# Compute neighbor sums and counts per var using sparse multiply
neighbor_sums <- Adj %*% X
neighbor_counts <- Adj %*% rep(1L, n_rows)

# Compute stats: mean straightforward
neighbor_means <- neighbor_sums / neighbor_counts

# For max/min, do block aggregation efficiently:
compute_max_min <- function(xvec, Adj) {
  # xvec numeric length n_rows
  # Return matrix [n_rows, 2] of max and min
  res_max <- res_min <- numeric(length(xvec))
  for (i in seq_len(nrow(Adj))) {
    nbr_idx <- Adj[i, ]@i + 1L  # neighbors of row i
    vals <- xvec[nbr_idx]
    if (length(vals) == 0) {
      res_max[i] <- NA
      res_min[i] <- NA
    } else {
      res_max[i] <- max(vals, na.rm = TRUE)
      res_min[i] <- min(vals, na.rm = TRUE)
    }
  }
  cbind(res_max, res_min)
}

# Apply for each var efficiently
neighbor_features <- vector("list", length(vars))
names(neighbor_features) <- vars
for (v in seq_along(vars)) {
  maxmin <- compute_max_min(X[, v], Adj)
  neighbor_features[[v]] <- cbind(
    max = maxmin[, 1],
    min = maxmin[, 2],
    mean = neighbor_means[, v]
  )
}

# Bind new features into data.table
for (v in vars) {
  nm <- paste0(v, "_nbr_")
  cell_data[, paste0(nm, c("max", "min", "mean")) := as.data.table(neighbor_features[[v]])]
}

# Predict using pre-trained RF model (model_rf)
preds <- predict(model_rf, newdata = cell_data)
```

**Key Gains**  
- **Single adjacency build** using Kronecker product for time expansion.  
- **Sparse matrix multiplication** for neighbor sums and counts enables fast mean calculation for millions of rows.  
- **Sequential block max/min computation** avoids expensive repeated list-lookupsâ€”O(E) complexity (edges).  
- Memory efficient: adjacency stored sparsely; no massive intermediate lists.  

**Expected runtime improvement**: From 86+ hours to a few hours or less (dominated by max/min loop), scalable on 16 GB RAM. Parallelize `compute_max_min` with `parallel` or `future.apply` for further speedup.
END_RESPONSE>>>
