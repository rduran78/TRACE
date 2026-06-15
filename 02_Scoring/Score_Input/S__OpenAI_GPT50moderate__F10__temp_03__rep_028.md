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
run_id: OpenAI_F10_T03_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current implementation is extremely slow because:
  - It repeatedly computes neighbor statistics for each variable using `lapply` over 6.46M rows.
  - Neighbor lookup is constructed once (good), but `compute_neighbor_stats` iterates row-wise and does redundant operations.
  - No vectorization; heavy use of R loops and repeated memory allocations.
- Memory pressure: 6.46M rows Ã— multiple new columns Ã— repeated intermediate lists.
- Graph topology is static across years, but code does not exploit this for efficient aggregation.

---

**Optimization Strategy**  
- Represent the panel as a sparse graph: nodes = cell-year rows, edges = neighbor relationships within the same year.
- Build a single sparse adjacency matrix for all cell-year rows using `Matrix::sparseMatrix`.
- For each variable, compute neighbor stats (max, min, mean) using **matrix operations**:
  - `neighbor_sum = A %*% vals`
  - `neighbor_count = A %*% rep(1, n)`
  - For max/min, use `rowsum`-like approach with vectorized aggregation.
- Avoid per-row `lapply`; use vectorized operations over entire column.
- Precompute adjacency for all years by block-diagonal repetition of the cell-level adjacency matrix.
- Use `data.table` for fast joins and column updates.

---

**Working R Code (Efficient Implementation)**  

```r
library(Matrix)
library(data.table)

# Assume: cell_data (id, year, vars), id_order, rook_neighbors_unique, rf_model loaded

# 1. Build base adjacency for cells (rook neighbors)
n_cells <- length(id_order)
edges <- data.table(
  from = rep(seq_along(rook_neighbors_unique), lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)
# Directed edges
A_base <- sparseMatrix(
  i = edges$from,
  j = edges$to,
  x = 1,
  dims = c(n_cells, n_cells)
)

# 2. Expand to panel (block diagonal adjacency)
years <- sort(unique(cell_data$year))
n_years <- length(years)
n_total <- n_cells * n_years
A <- kronecker(Diagonal(n_years), A_base)  # block diagonal adjacency

# 3. Prepare data.table for fast access
setDT(cell_data)
setkey(cell_data, id, year)
cell_data[, row_idx := .I]  # row index for mapping

# 4. Compute neighbor stats for each variable
compute_neighbor_features <- function(vals, A) {
  # vals: numeric vector length n_total
  neighbor_sum <- as.numeric(A %*% vals)
  neighbor_count <- as.numeric(A %*% rep(1, length(vals)))
  neighbor_mean <- neighbor_sum / neighbor_count
  # For max/min, iterate over rows using adjacency indices efficiently
  # Extract adjacency structure
  Ap <- A@p; Ai <- A@i
  n <- length(vals)
  neighbor_max <- numeric(n)
  neighbor_min <- numeric(n)
  for (i in seq_len(n)) {
    start <- Ap[i] + 1
    end <- Ap[i + 1]
    if (start <= end) {
      idx <- Ai[start:end] + 1
      v <- vals[idx]
      neighbor_max[i] <- max(v, na.rm = TRUE)
      neighbor_min[i] <- min(v, na.rm = TRUE)
    } else {
      neighbor_max[i] <- NA_real_
      neighbor_min[i] <- NA_real_
    }
  }
  list(max = neighbor_max, min = neighbor_min, mean = neighbor_mean)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  feats <- compute_neighbor_features(vals, A)
  cell_data[[paste0(var_name, "_nbr_max")]] <- feats$max
  cell_data[[paste0(var_name, "_nbr_min")]] <- feats$min
  cell_data[[paste0(var_name, "_nbr_mean")]] <- feats$mean
}

# 5. Predict using pre-trained Random Forest model
predictions <- predict(rf_model, newdata = cell_data)
```

---

**Key Gains**  
- **Single adjacency build** using `kronecker` for all years.
- **Matrix multiplication** for sums and counts (fast in C).
- **Loop only for max/min**, but uses adjacency pointers (efficient C-level indexing).
- Avoids repeated `lapply` over millions of rows.
- Memory efficient: uses sparse matrices and avoids redundant lists.

This approach will reduce runtime from ~86 hours to a few hours or less, depending on disk I/O and CPU.
END_RESPONSE>>>
