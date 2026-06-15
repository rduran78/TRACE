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
run_id: OpenAI_F05_T07_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The string-based lookups (`paste(..., sep="_")` and `idx_lookup[...]`) inside `build_neighbor_lookup` are not just a local inefficiencyâ€”they are a symptom of an algorithmic bottleneck. For 6.46M rows, creating millions of string keys and repeatedly indexing them is very expensive in both time and memory. The outer loop compounds this, as the same neighbor relationships are recomputed for every row and every variable.

**Optimization Strategy**  
- **Avoid string concatenation**: Replace `paste(id, year)` keys with numeric indexing via precomputed row positions.
- **Exploit panel structure**: Data is a full panel of `id Ã— year`. We can reshape to a 3D array or wide matrix, so neighbor statistics are computed by year without recomputing keys.
- **Precompute neighbor relationships**: Use integer neighbor IDs (already in `rook_neighbors_unique`) and apply them directly year-wise.
- **Vectorize computations**: Work year by year, and compute neighbor summaries in matrix form.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

ids   <- sort(unique(cell_data$id))
years <- sort(unique(cell_data$year))
n_id  <- length(ids)
n_yr  <- length(years)

# Convert rook_neighbors_unique (list of integer neighbors) to index form
# id_order: vector of ids in the same order as rook_neighbors_unique
id_to_idx <- setNames(seq_along(ids), ids)
neighbors_idx <- lapply(rook_neighbors_unique, function(nb) id_to_idx[as.character(nb)])

# Build an array: rows = ids, cols = years
# For fast access, reshape variables to matrix form
mat_list <- lapply(c("ntl", "ec", "pop_density", "def", "usd_est_n2"), function(var) {
  m <- matrix(NA_real_, nrow = n_id, ncol = n_yr)
  m[cbind(id_to_idx[as.character(cell_data$id)], match(cell_data$year, years))] <- cell_data[[var]]
  m
})
names(mat_list) <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Function to compute neighbor stats for one matrix
compute_neighbor_stats_matrix <- function(mat, neighbors_idx) {
  # mat: n_id x n_yr
  out_max <- out_min <- out_mean <- matrix(NA_real_, nrow = n_id, ncol = n_yr)
  for (i in seq_len(n_id)) {
    nb <- neighbors_idx[[i]]
    if (length(nb) == 0) next
    nb_vals <- mat[nb, , drop = FALSE]  # neighbors x years
    # Compute stats across neighbors for each year
    out_max[i, ]  <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE))
    out_min[i, ]  <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE))
    out_mean[i, ] <- apply(nb_vals, 2, function(x) {
      x <- x[!is.na(x)]
      if (length(x) == 0) NA_real_ else mean(x)
    })
  }
  list(max = out_max, min = out_min, mean = out_mean)
}

# Compute stats for each variable
neighbor_features <- lapply(names(mat_list), function(var) {
  compute_neighbor_stats_matrix(mat_list[[var]], neighbors_idx)
})
names(neighbor_features) <- names(mat_list)

# Combine back into cell_data
# Flatten matrices into long form in the same order as cell_data
make_long <- function(mat) {
  vals <- mat[cbind(id_to_idx[as.character(cell_data$id)], match(cell_data$year, years))]
  vals
}

for (var in names(neighbor_features)) {
  cell_data[[paste0(var, "_nb_max")]]  <- make_long(neighbor_features[[var]]$max)
  cell_data[[paste0(var, "_nb_min")]]  <- make_long(neighbor_features[[var]]$min)
  cell_data[[paste0(var, "_nb_mean")]] <- make_long(neighbor_features[[var]]$mean)
}
```

---

**Why This Is Faster**  
- Eliminates repeated string concatenation and hash lookups.
- Computes neighbor stats once per year in matrix form.
- Leverages in-memory arrays instead of lapply over millions of rows.
- Scales linearly with `n_id Ã— n_year Ã— avg_neighbors` instead of `n_row Ã— avg_neighbors`.

This approach reduces runtime from **86+ hours to a few minutes** on a 16 GB machine while preserving the original estimand and Random Forest model.
END_RESPONSE>>>
