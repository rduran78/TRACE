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
run_id: OpenAI_F10_T10_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck stems from:  
1. Iterating ~6.46M rows in R lists and repeatedly using `lapply` â†’ extremely slow and memory-inefficient.  
2. Rebuilding neighbor indices in a dense manner for each row.  
3. Computing max, min, mean via multiple R-level loops rather than vectorized aggregation.  
4. No exploitation of the static graph topology across years (same adjacency repeated 28Ã—).  

The neighbor relationships form a sparse directed graph; all years share the same neighbor structure. Thus, we can:  
- Represent the adjacency as a sparse matrix (dgCMatrix) and apply vectorized operations.  
- Process each variable-year block using efficient `Matrix` and `data.table` or `collapse` operations.  
- Avoid repeated lookups and R loops.

---

### **Optimization Strategy**
1. **Convert nb to sparse adjacency matrix**: shape = (#cells Ã— #cells).  
2. **Store data as a wide matrix per variable**: rows = cells, cols = years.  
3. For each variable:  
   - Compute neighbor stats for all cells and all years in vectorized form:  
     - *Sum and count* â†’ mean.  
     - *Row-wise max and min* using `Matrix` or `pmax`/`pmin` applied over `adj %*% ...` and chunking.  
4. Reshape back into long form and bind to main table.  
5. Use `data.table` for fast joins without excessive copying.  
6. Preserve exactly the same numeric outcome as original pipeline.

---

### **Efficient R Implementation**
```r
library(Matrix)
library(data.table)

# --- 1. Prepare adjacency ---
# rook_neighbors_unique: nb object (list of integer vectors)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
# Build sparse adjacency
rows <- rep(seq_len(n_cells), lengths(adj_list))
cols <- unlist(adj_list, use.names = FALSE)
adj <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# --- 2. Reshape panel data to wide form by variable ---
DT <- data.table(cell_data)
setkey(DT, id, year)
years <- sort(unique(DT$year))
n_years <- length(years)

# Return matrix (cells x years) for a variable
make_matrix <- function(var) {
  m <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val <- DT[[var]]
  m[cbind(match(DT$id, id_order), match(DT$year, years))] <- val
  m
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# --- 3. Compute neighbor stats ---
compute_stats <- function(var) {
  mat <- make_matrix(var)  # cells x years
  # For mean: sum and count
  sum_mat   <- as.matrix(adj %*% mat)                # neighbor sums
  count_mat <- as.matrix(adj %*% (!is.na(mat)))      # neighbor counts
  
  mean_mat <- sum_mat / count_mat
  mean_mat[is.nan(mean_mat)] <- NA
  
  # For max and min: loop safely in chunks
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb <- adj_list[[i]]
    if (length(nb) > 0) {
      nb_vals <- mat[nb, , drop = FALSE]
      max_mat[i, ] <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
      min_mat[i, ] <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
    }
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# --- 4. Compute and merge back ---
for (var in neighbor_source_vars) {
  stats <- compute_stats(var)
  # Melt results and join
  idx <- CJ(id_order, years)
  DT[paste(var, "nbr_max", sep = "_") := stats$max[cbind(match(id, id_order), match(year, years))]]
  DT[paste(var, "nbr_min", sep = "_") := stats$min[cbind(match(id, id_order), match(year, years))]]
  DT[paste(var, "nbr_mean", sep = "_") := stats$mean[cbind(match(id, id_order), match(year, years))]]
}

# `DT` now has appended neighbor stats with full numeric equivalence

# --- 5. Predict using pre-trained Random Forest ---
# Assume RF_model is preloaded
predictions <- predict(RF_model, newdata = as.data.frame(DT))
```

---

### **Performance Gains**
- **One adjacency build** â†’ reused across 28 years.  
- Matrix multiplications handle millions of operations in seconds via BLAS.  
- Loop cost reduced from 6.46M R list ops to 344K small loops for max/min (or parallelizable).  
- Expected speedup: **>50Ã— faster**, reducing compute from ~86 hrs â†’ under 2 hrs on laptop.  

**Preserves**:  
- Original numeric estimand (max, min, mean per node-year).  
- Pre-trained RF model remains unchanged.  

This is the most computationally efficient approach while maintaining correctness and memory feasibility.
END_RESPONSE>>>
