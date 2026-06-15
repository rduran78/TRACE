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
run_id: OpenAI_F10_T10_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current implementation is extremely slow because:
- `build_neighbor_lookup` constructs neighbor indices repeatedly for every row across **6.46 million rows**, causing large memory churn.
- `compute_neighbor_stats` uses `lapply` row-wise, repeatedly scanning neighbors and extracting valuesâ€”this is inherently `O(N Ã— k)` with high R loop overhead.
- Neighbor relationships do **not change across years**, but code rebuilds neighbor structures per cell-year.
- No vectorization: every operation iterates row-wise instead of leveraging matrix operations.
- Writing 15 new columns (max/min/mean for five variables) via repeated `compute_and_add_neighbor_features` multiplies overhead.

---

### **Optimization Strategy**
1. **Precompute a global sparse graph adjacency list** once at the **cell level** (344,208 nodes).
2. Exploit the repeated panel years: replicate adjacency for each year logically, **without physically copying neighbors NÃ—T times**.
3. Convert `cell_data` into a **wide matrix by variable**, grouped by year for faster block access.
4. Implement neighbor aggregation via:
   - **Sparse matrix multiplications** using the `Matrix` package.
   - For each year, build an adjacency matrix **A** (row normalization if needed), compute max/min/mean using fast vector ops per row.
5. **Chunk processing** to manage memory (process year by year).
6. Append features efficiently using `data.table` or `dplyr` joins with precomputed results.
7. **Parallelization:** Use `parallel::mclapply` or `future.apply` across variables or years.
8. No conversion of Random Forest modelâ€”just append features as before for prediction.

---

### **Efficient R Implementation**

```r
library(data.table)
library(Matrix)
library(pbapply)

# ---- Prepare Data ----
setDT(cell_data)
setkey(cell_data, id, year)

ids <- unique(cell_data$id)
n_cells <- length(ids)
years <- sort(unique(cell_data$year))

# Convert neighbor list to adjacency structure
# rook_neighbors_unique is an nb object
adj_list <- rook_neighbors_unique
# Build adjacency matrix in sparse form (cells only)
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list)
A <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Helper function for neighbor aggregates
compute_year_stats <- function(dt_year, A, vars) {
  res_list <- list()
  # Convert variables to matrix
  X <- as.matrix(dt_year[, ..vars])
  # For mean: sum and count
  neighbor_sum <- A %*% X
  neighbor_count <- Matrix::rowSums(A > 0)
  neighbor_mean <- sweep(neighbor_sum, 1, neighbor_count, `/`)
  
  # For max/min
  # We'll loop variable-wise using fast apply
  for (j in seq_along(vars)) {
    v <- X[, j]
    # Compute max/min over neighbors
    # We'll manually aggregate using pblapply for speed
    max_vec <- pbsapply(seq_len(n_cells), function(i) {
      nb <- adj_list[[i]]
      if (length(nb) == 0) return(NA_real_)
      max(v[nb], na.rm = TRUE)
    })
    min_vec <- pbsapply(seq_len(n_cells), function(i) {
      nb <- adj_list[[i]]
      if (length(nb) == 0) return(NA_real_)
      min(v[nb], na.rm = TRUE)
    })
    
    res_list[[paste0(vars[j], "_nb_max")]] <- max_vec
    res_list[[paste0(vars[j], "_nb_min")]] <- min_vec
    res_list[[paste0(vars[j], "_nb_mean")]] <- neighbor_mean[, j]
  }
  as.data.table(res_list)
}

# ---- Main loop: process by year to avoid 6.4M overhead ----
result_list <- list()
for (yr in years) {
  dt_year <- cell_data[year == yr]
  stats_dt <- compute_year_stats(dt_year, A, neighbor_source_vars)
  stats_dt[, id := dt_year$id]
  stats_dt[, year := yr]
  result_list[[as.character(yr)]] <- stats_dt
}

neighbor_features <- rbindlist(result_list)
setkey(neighbor_features, id, year)

# Merge back with original
cell_data <- neighbor_features[cell_data]

# ---- cell_data now has neighbor aggregates ----
# Next step: Apply pre-trained Random Forest model
predictions <- predict(pretrained_rf_model, newdata = cell_data)

```

---

### **Why This Is Fast**
- Builds graph topology **once** (O(n_cells + edges)).
- Processes per-year slices, not per-row iterations.
- Uses **sparse matrix multiplication** for mean aggregates and **vectorized** loops for min/max.
- Avoids row-wise `lapply` over 6.4M rows â†’ replaced with `pbsapply` over 344K nodes per year Ã— 28 years with progress bar (parallelizable).
- Memory-friendly: only one year (~344K rows) loaded at a time.

---

**Expected speed-up:** From **86+ hours â†’ a few hours** on a standard laptop (even less with parallelization). Numerical results remain equivalent to original implementation, preserving the Random Forest estimand.
END_RESPONSE>>>
