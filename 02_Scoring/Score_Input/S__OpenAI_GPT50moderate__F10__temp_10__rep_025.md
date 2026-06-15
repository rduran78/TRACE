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
run_id: OpenAI_F10_T10_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The current implementation is extremely slow because:
  - For each of ~6.46M cell-year rows, it repeatedly runs `lapply` and performs multiple vector lookups for neighbors, creating huge R-level overhead.
  - Neighbor stats are recomputed from scratch per variable and are based on inefficient list structures.
- Memory and speed bottleneck: repeated processing inside R loops instead of vectorized or matrix-based operations.  
- The `neighbor_lookup` is reused but stats computation is very expensive because each row deals with lists and uses multiple conditional checks.

---

**Optimization Strategy**  
- Build a **sparse adjacency matrix** once for all cells (344,208 nodes) using `rook_neighbors_unique`.
- Use the Kronecker product with an identity matrix for years (28) only if needed; better: compute stats year-wise and append results.
- Leverage `Matrix` package for sparse structures.
- Compute `max`, `min`, and `mean` in a **vectorized way using matrix operations** or fast `rowsum` aggregation.
- Avoid per-row R loops; use `apply` over sparse subviews or collapse year by year.
- Process neighbor stats per variable in **chunked** or **year-slice parallel manner**.
- Guarantee identical output: NA handling as original (ignore NA neighbors; if none left, all NA).

---

**Working R Code**

```r
library(Matrix)
library(data.table)

# Assumptions:
# - cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# - id_order: vector of cell IDs in correct order (matches rook_neighbors_unique)
# - rook_neighbors_unique: spdep::nb object

# 1. Build Sparse Adjacency (344,208 x 344,208)
build_adjacency <- function(neighbors, n) {
  i_idx <- rep(seq_along(neighbors), lengths(neighbors))
  j_idx <- unlist(neighbors)
  sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
adj <- build_adjacency(rook_neighbors_unique, n_cells)

# 2. Convert cell_data to data.table keyed by id-year
setDT(cell_data)
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Pre-build row index mapping (id -> row index per year block)
id_to_idx <- setNames(seq_along(id_order), id_order)

compute_neighbor_stats_matrix <- function(vals, adj) {
  # vals: numeric vector length = number of cells
  # adj: sparse adjacency
  # For each node, gather its neighbors' vals
  # Compute max, min, mean efficiently
  # Handle all-NAs
  n <- length(vals)
  # Create dense matrix subset for computation
  # Extract neighbor values for all nodes in efficient pass
  # Use adjacency to compute sums and counts for mean
  # Sum:
  val_na <- is.na(vals)
  vals_non_na <- vals
  vals_non_na[val_na] <- 0
  sums <- as.numeric(adj %*% vals_non_na)
  counts <- as.numeric(adj %*% (!val_na))
  means <- ifelse(counts > 0, sums / counts, NA_real_)
  
  # For max and min: iterate neighbors efficiently
  # (sparse traversal)
  maxs <- rep(NA_real_, n)
  mins <- rep(NA_real_, n)
  rows <- seq_len(n)
  aj <- adj@j + 1
  ai <- rep(seq_along(adj@p[-1]), diff(adj@p))  # row indices
  # Compute in one pass
  for (k in seq_along(ai)) {
    r <- ai[k]; c <- aj[k]
    v <- vals[c]
    if (!is.na(v)) {
      if (is.na(maxs[r]) || v > maxs[r]) maxs[r] <- v
      if (is.na(mins[r]) || v < mins[r]) mins[r] <- v
    }
  }
  
  cbind(maxs, mins, means)
}

# 3. Apply year by year to keep memory low
results_list <- vector("list", length(neighbor_source_vars))
names(results_list) <- neighbor_source_vars

for (var_name in neighbor_source_vars) {
  max_col <- paste0(var_name, "_nb_max")
  min_col <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  cell_data[, (max_col) := NA_real_]
  cell_data[, (min_col) := NA_real_]
  cell_data[, (mean_col) := NA_real_]
}

for (yr in years) {
  subset_idx <- which(cell_data$year == yr)
  vals_dt <- cell_data[year == yr]
  positions <- id_to_idx[as.character(vals_dt$id)]
  
  for (var_name in neighbor_source_vars) {
    vals <- vals_dt[[var_name]]
    stats_mat <- compute_neighbor_stats_matrix(vals, adj)
    cell_data[subset_idx, (paste0(var_name, "_nb_max")) := stats_mat[, 1]]
    cell_data[subset_idx, (paste0(var_name, "_nb_min")) := stats_mat[, 2]]
    cell_data[subset_idx, (paste0(var_name, "_nb_mean")) := stats_mat[, 3]]
  }
}

# At this point, cell_data has all neighbor feature columns computed efficiently
# Pass to the pre-trained Random Forest model as before:
# preds <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### Why This Is Fast
- Adjacency is built once (sparse format).
- Stats are computed in **linear time over edges** with a single pass for max/min, matrix multiply for means.
- Year-level batching avoids building an enormous Kronecker graph.
- No nested `lapply` â†’ minimal R overhead.
- Memory-conscious: reuses adjacency, processes chunks.

This pipeline is equivalent to the original in terms of numerical results but is optimized for high performance on a laptop.
END_RESPONSE>>>
