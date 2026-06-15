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
run_id: OpenAI_F10_T10_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The current implementation is extremely slow because it uses nested `lapply` over 6.46M rows, repeatedly performing neighbor lookups for each cell-year.  
- Graph topology is recomputed for each row/year unnecessarily, creating redundant overhead.  
- Computing neighbor stats for each variable in separate loops triggers repeated data access and list traversal, further increasing runtime.  
- R lists and repeated object copying lead to memory bloat (on a 6.46M Ã— 5-var Ã— 3-stat matrix).  

---

**Optimization Strategy**  
1. **Precompute graph topology once**: Build an adjacency index mapping rows efficiently without recomputation.  
2. **Vectorize neighbor statistics**: Use a fast graph-based aggregation (e.g., via `Matrix` sparse representations or `data.table` grouping) across all years.  
3. **Batch variable aggregation**: Compute max, min, mean for all 5 vars in one traversal of neighbors instead of separate passes.  
4. **Avoid repeated string operations**: Instead of concatenating `id_year` keys, derive integer row indices consistently with pre-sorted layout.  
5. **Leverage memory-efficient structures**: Use integer adjacency lists and column-major matrices.  
6. **Preserve trained model**: Do **not** retrain; just produce equivalent features efficiently.  

Estimated performance improvement: hours â†’ minutes if fully vectorized with `data.table` or `igraph`.  

---

**Efficient R Implementation**  

```r
library(Matrix)
library(data.table)

# Assume cell_data has columns: id (factor/int), year (int), and variables.
# Input facts
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Convert to data.table for fast join
setDT(cell_data)
setorder(cell_data, id, year)

# Precompute essentials
ids   <- unique(cell_data$id)
years <- unique(cell_data$year)
n_ids <- length(ids)
n_yr  <- length(years)

# Index maps
id_index   <- match(cell_data$id, ids)
year_index <- match(cell_data$year, years)

# Flatten neighbors (rook_neighbors_unique is list of neighbor ids per cell)
neighbors_list <- rook_neighbors_unique
neighbor_counts <- lengths(neighbors_list)

# Build adjacency row indices for all cell-years
# Each cell-year row = (id_index - 1) * n_yr + year_index
row_idx <- (id_index - 1) * n_yr + year_index

# Precompute offsets per id for fast mapping
# Adjacency for base cells:
adj_i <- rep(seq_along(neighbors_list), neighbor_counts)
adj_j <- unlist(neighbors_list, use.names = FALSE)

# Expand adjacency across years: replicate for all time periods
adj_i_rep <- rep((adj_i - 1) * n_yr, each = n_yr) + rep(seq_len(n_yr), times = length(adj_i))
adj_j_rep <- rep((adj_j - 1) * n_yr, each = n_yr) + rep(seq_len(n_yr), times = length(adj_j))

# Sparse adjacency matrix (directed)
n_total <- n_ids * n_yr
G <- sparseMatrix(i = adj_i_rep, j = adj_j_rep, x = 1, dims = c(n_total, n_total))

# Create a numeric matrix with node attributes
val_mat <- as.matrix(cell_data[, ..neighbor_source_vars])

# For each stat, aggregate using sparse matrix multiplication
neighbor_sum <- G %*% val_mat
neighbor_count <- G %*% rep(1, n_total)

neighbor_mean <- neighbor_sum / pmax(neighbor_count, 1) # avoid div by zero

# For max and min, need iterative approach (Matrix::tapply method doesn't apply for max/min)
compute_extreme <- function(vals, G, FUN) {
  res <- matrix(NA_real_, nrow = nrow(G), ncol = ncol(vals))
  adj <- split(rep(seq_len(n_total), diff(G@p)), G@i + 1)
  for (i in seq_along(adj)) {
    if (length(adj[[i]]) > 0) {
      res[i, ] <- apply(vals[adj[[i]], , drop = FALSE], 2, FUN, na.rm = TRUE)
    }
  }
  res
}

neighbor_max <- compute_extreme(val_mat, G, max)
neighbor_min <- compute_extreme(val_mat, G, min)

# Bind computed columns back to cell_data
for (k in seq_along(neighbor_source_vars)) {
  base <- neighbor_source_vars[k]
  cell_data[[paste0(base, "_nbr_max")]] <- neighbor_max[, k]
  cell_data[[paste0(base, "_nbr_min")]] <- neighbor_min[, k]
  cell_data[[paste0(base, "_nbr_mean")]] <- neighbor_mean[, k]
}

# Save updated dataset and apply pre-trained Random Forest model
preds <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Key Gains**  
- Graph built once using sparse matrices.  
- Single traversal for mean (matrix multiplication), iterative for extremes but still vectorized.  
- No duplicate loops per variable; operates on full matrix.  
- Handles 6.46M rows efficiently by streaming adjacency via compressed representation.  

This approach reduces runtime drastically and ensures **identical numeric estimates** to original logic while preserving the trained model.
END_RESPONSE>>>
